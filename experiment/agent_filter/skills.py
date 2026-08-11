"""Question-shape 驅動的 agent skill 庫。

從 v2-v9 的錯誤分析與成功軌跡沉澱出的搜尋戰術,按**問題形狀**(regex 偵測,
與 benchmark 類別標籤無關)注入 agent 的任務訊息。每個 skill 是一段具體的
作戰手冊:用什麼 anchor、怎麼下 GREP、何時 READ、選擇時保留什麼。

來源依據:
- 成功軌跡:寬搜 → 鑽入 → 答案 span 反向驗證(binaural beats 案例)
- 失敗模式 1:彙整題漏算 mention / 只留一條(v2 錯題 16/39)
- 失敗模式 2:KU 取到舊值(錯例 context 只剩 1-2 條)
- 失敗模式 3:整句 phrase 0 命中、日期條件 grep 不到(March ≠ 2023/03)
"""
from __future__ import annotations

import re


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


# ── 已停用:counting skill ────────────────────────────────────────────────
# v10 配對驗證歸因:counting 命中題的保持率只有 73%(其他 skill 93-100%)——
# 「keep every instance / more evidence beats cleaner」的建議與 precision 引擎
# 直接衝突(與 v3-v9 的 min-keep/sufficiency 同一規律)。counting 題的真正
# 瓶頸在答題端彙整能力,證據端救不動(v7 硬 prompt 同樣失敗)。保留文字供
# 換更強模型時重測。
_COUNTING_SKILL_DISABLED = (
    "counting",
    _rx(r"\b(how many (?!(days?|weeks?|months?|years?|hours?|minutes?)\b)|how much|"
        r"how often|how frequently|number of|total|in total|altogether|count|sum of)\b"),
    "SKILL counting/aggregation — the answer is a NUMBER computed from multiple mentions:\n"
    "- Enumerate EVERY mention: grep the countable noun AND its variants "
    "(singular/plural/synonyms, e.g. appointment|visit|checkup). One grep is never "
    "enough — different sessions use different words.\n"
    "- If a timeframe is given, ALSO grep the date stamp form: 'in March' → GREP 2023/03 "
    "(stamps are [YYYY/MM/DD]); combine: GREP 2023/03.*appointment.\n"
    "- READ each hit to check it is a DISTINCT event (same event re-mentioned twice "
    "counts once; two events on different dates count separately).\n"
    "- KEEP every distinct instance in FINAL — dropping one mention breaks the count. "
    "For counting questions, more evidence beats cleaner evidence.",
)

# 每個 skill: (name, detector, strategy)
# NOTE: counting 目前預設不在此清單(見 _COUNTING_SKILL_DISABLED)。
# 若環境變數 GREP_AGENT_COUNTING_SKILL=1,select_skills 會把它插到最前面
# (ablation 用,120b 重測)。
SKILLS: list[tuple[str, re.Pattern, str]] = [
    (
        "latest-value",
        _rx(r"\b(most recent(ly)?|latest|current(ly)?|now|these days|still|"
            r"nowadays|as of|right now|today)\b|\b(did i .*(change|update|switch))\b"),
        "SKILL latest-value — the fact may have been UPDATED over time:\n"
        "- Grep the target entity/fact across ALL sessions; collect every dated mention "
        "(the same fact stated with different values on different dates).\n"
        "- KEEP BOTH the old and the new mentions in FINAL — the answer needs the most "
        "recent one, and the reader picks it by comparing the [date] stamps. Keeping "
        "only one mention risks keeping the stale value.\n"
        "- If only one mention is found, grep synonyms of the fact before concluding "
        "there was no update.",
    ),
    (
        "temporal-computation",
        _rx(r"\b(how long|how many (days|weeks|months|years)|days? (before|after|between)|"
            r"ago\b|duration|since when|when did|what (day|date))\b"),
        "SKILL temporal-computation — the answer needs DATES for arithmetic or ordering:\n"
        "- Locate the dated anchor turn for EACH event in the question (two events for "
        "'between', one for 'ago/when').\n"
        "- Date stamps are searchable: GREP 2023/05 finds May-2023 turns; month names in "
        "text ('last Saturday', 'in January') often do NOT appear as stamps — grep the "
        "EVENT keywords first, then read the [date] stamp of the hit.\n"
        "- Prefer turns whose text also states the date explicitly; keep every turn whose "
        "[date] stamp is needed for the computation.",
    ),
    (
        "preference-recommendation",
        _rx(r"\b(recommend|suggest(ion)?s?|what should i|any (tips|ideas|advice)|"
            r"help me (choose|pick|plan)|would i (like|prefer|enjoy))\b"),
        "SKILL preference/recommendation — the answer must fit the USER's own stated "
        "situation, not generic advice:\n"
        "- The key evidence is what the USER said about themselves: grep "
        "'i (have|use|own|prefer|like|love|enjoy|hate|dislike|am allergic)' combined with "
        "the topic word; also grep the user's named gear/brands/places.\n"
        "- KEEP the user-side turns (sid ending :u) describing their setup, constraints "
        "and tastes — an assistant's earlier suggestions are secondary evidence.\n"
        "- Multiple aspects of their situation may live in different sessions; sweep more "
        "than one keyword before FINAL.",
    ),
    (
        "literal-recall",
        _rx(r"\b(what (is|was|did)|which|who|whom|whose|where (did|do|was)|"
            r"name of|called)\b"),
        "SKILL literal-recall — the answer is a literal span (name/number/place/title):\n"
        "- Anchor on the RAREST word in the question (proper nouns, unusual terms, "
        "numbers) — one rare word beats a long phrase (multi-word patterns match as an "
        "exact phrase and usually miss).\n"
        "- After locating a promising turn, READ it — the span is often deep inside a "
        "long turn.\n"
        "- Before FINAL, verify: grep a distinctive fragment of the answer you found "
        "(e.g. GREP 38 subjects) to confirm the span really exists in the selected turn.\n"
        "- Select the minimal turn(s) containing the span; drop topical look-alikes.",
    ),
]

# literal-recall 的 detector 很寬(what/which/who),作為墊底 skill;
# 排序即優先序,最多取前 N 個命中。
MAX_SKILLS = 2


def select_skills(question: str) -> list[tuple[str, str]]:
    """回傳 [(name, strategy)],依 SKILLS 順序最多 MAX_SKILLS 個。

    GREP_AGENT_COUNTING_SKILL=1 時把已停用的 counting skill 插到最前(ablation)。
    """
    import os
    active = SKILLS
    if os.getenv("GREP_AGENT_COUNTING_SKILL", "0") not in ("0", "", "false"):
        active = [_COUNTING_SKILL_DISABLED, *SKILLS]
    out = []
    for name, det, strategy in active:
        if det.search(question):
            out.append((name, strategy))
        if len(out) >= MAX_SKILLS:
            break
    return out
