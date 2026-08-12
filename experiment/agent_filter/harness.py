"""Grep agent mini-harness (inline delivery, plain-text command protocol).

流程:
  seed = Evidence Summary 內的 16 個 sid(vector+rerank 粗篩結果)
  → agent 用 GREP / READ 驗證候選 + 補找漏掉的字面證據
  → FINAL sids → 用 raw turn text 重組 Evidence Summary block
安全網:agent 失敗 / 輸出無效 / 超出預算 → 原 context 原封不動退回。

不用 function-calling API:local 模型(gpt-oss-20b via LM Studio)對 plain-text
單行指令協議最穩,指令用 regex 解析。
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path

from experiment.agent_filter.corpus import Corpus, load_corpus
from experiment.agent_filter.prompts import (
    ABSTENTION_HINT,
    CATEGORY_HINTS,
    GAP_HINT_TEMPLATE,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)

_EVIDENCE_HEADER = "### Evidence Summary"
_SID_RE = re.compile(r"\[sid=([^\]\s]+)\]")

_GREP_RE = re.compile(r"^\s*GREP\s+(.+?)\s*$", re.IGNORECASE)
# 彙整/最新值型問題偵測(問題驅動的保留策略觸發器,與資料集類別標籤無關)
_AGG_QUESTION_RE = re.compile(
    r"\b(how many|how much|how often|how long|how frequently|total|count|sum|"
    r"number of|most recent(ly)?|latest|currently|current|in total|altogether)\b",
    re.IGNORECASE,
)
_READ_RE = re.compile(r"^\s*READ\s+(\S+)(?:\s+(\d+))?\s*$", re.IGNORECASE)
_VECTOR_RE = re.compile(r"^\s*VECTOR\s+(.+?)\s*$", re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\s*[::]?\s*(.*?)\s*$", re.IGNORECASE)
_COVERAGE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{2,}", re.IGNORECASE)
_COVERAGE_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "were", "have", "has",
    "had", "for", "her", "his", "their", "they", "them", "she", "you",
    "your", "what", "when", "where", "which", "does", "did", "about",
    "into", "also", "just", "very", "more", "some", "user", "assistant",
}


def seed_sids_from_context(context: str) -> list[str]:
    """依出現順序取出 Evidence Summary block 的 sid(保序去重)。"""
    idx = context.find(_EVIDENCE_HEADER)
    if idx == -1:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _SID_RE.finditer(context[idx:]):
        s = m.group(1).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# Evidence Summary 每條的 rerank 分數:`[sid=...][score=0.615]`(前端候選表的 Score 欄)
_SID_SCORE_RE = re.compile(r"\[sid=([^\]\s]+)\][^\[]*?\[score=([-\d.]+)\]")


def seed_scores_from_context(context: str) -> dict[str, float]:
    """取出每個 seed sid 的 rerank 分數(保留第一次出現);解析失敗略過。"""
    idx = context.find(_EVIDENCE_HEADER)
    scan = context[idx:] if idx != -1 else context
    out: dict[str, float] = {}
    for m in _SID_SCORE_RE.finditer(scan):
        sid = m.group(1).strip()
        if sid in out:
            continue
        try:
            out[sid] = float(m.group(2))
        except ValueError:
            continue
    return out


def graph_context_from_context(context: str, *, max_chars: int = 12000) -> str:
    """Return the retrieved graph prefix (Entities/Relationships) if present."""
    idx = context.find(_EVIDENCE_HEADER)
    prefix = context[:idx] if idx != -1 else context
    if "=== Entities ===" not in prefix and "=== Relationships ===" not in prefix:
        return ""
    prefix = prefix.strip()
    return prefix if len(prefix) <= max_chars else prefix[:max_chars] + "\n…(graph context truncated)"


def _resp_diag(resp) -> dict:
    """回應層診斷欄位:區分「模型真的空回覆」vs「輸出落在 reasoning channel /
    tool_calls,content 讀不到」(gpt-oss harmony 常見)。附在每步 trace 上。"""
    try:
        choice = resp.choices[0]
        msg = choice.message
    except (AttributeError, IndexError):
        return {"diag": "no_choices"}
    d: dict = {"finish_reason": getattr(choice, "finish_reason", None)}
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        d["reasoning"] = str(reasoning)[:1200]
    tc = getattr(msg, "tool_calls", None)
    if tc:
        d["tool_calls"] = [
            {"name": getattr(getattr(t, "function", None), "name", None),
             "arguments": str(getattr(getattr(t, "function", None), "arguments", ""))[:300]}
            for t in tc
        ]
    content = getattr(msg, "content", None)
    if not (content or "").strip():
        # content 空:把 message 實際帶了哪些欄位也記下來,供排除 adapter 丟欄位
        d["content_empty"] = True
        d["message_keys"] = sorted(vars(msg).keys())
    return d


def _response_command_candidates(resp) -> list[tuple[str, str]]:
    """Return possible command-bearing text from an OpenAI-compatible response.

    gpt-oss/Harmony adapters do not agree on where the visible command goes:
    some put it in ``content``, some expose native ``tool_calls``, and some
    (notably older LM Studio adapters) put it in ``reasoning``.  The command
    parser itself is format-agnostic, so normalize all three surfaces here.
    """
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return []

    candidates: list[tuple[str, str]] = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        candidates.append(("content", content.strip()))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not name:
            continue
        name = str(name).split(".")[-1].upper()
        if name not in _CMD_NAMES:
            continue
        if isinstance(arguments, str):
            payload = arguments
        else:
            payload = json.dumps(arguments or {}, ensure_ascii=False)
        candidates.append(("tool_calls", f"to={name} <|message|>{payload}"))

    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        candidates.append(("reasoning", reasoning.strip()))
    return candidates


# gpt-oss(harmony template)有時會用原生 tool-call 語法回覆:
#   <|channel|>commentary to=READ <|constrain|>json<|message|>{"id": "...", "k": 2}
_HARMONY_RE = re.compile(
    r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b.*?<\|message\|>\s*(\{.*?\})\s*(?:<\|\w+\|>|$)",
    re.IGNORECASE | re.DOTALL,
)
# 更亂的變體:to=GREP <|constrain|>="pattern"(沒有 <|message|> JSON)
# namespace 前綴放寬:functions./tool./任何 <ns>. 都剝掉(92 機模型用 to=tool.GREP)。
_HARMONY_LOOSE_RE = re.compile(r"to=(?:\w+\.)?(GREP|READ|VECTOR|FINAL)\b(.*)$", re.IGNORECASE)


_CMD_NAMES = {"GREP", "READ", "VECTOR", "FINAL"}


def _flatten_json(obj) -> tuple[list[str], list[int]]:
    """遞迴收集 JSON payload 內的字串與整數(模型的 schema 不可預測)。"""
    strings: list[str] = []
    ints: list[int] = []
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, bool):
            continue
        elif isinstance(cur, int):
            ints.append(cur)
        elif isinstance(cur, str):
            strings.append(cur)
    return strings, ints


def _parse_harmony(reply: str) -> tuple[str, str] | None:
    m = None
    for m in _HARMONY_RE.finditer(reply):
        pass  # 取最後一個 tool call
    if m is None:
        return None
    kind = m.group(1).upper()
    try:
        payload = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None

    strings, ints = _flatten_json(payload)
    # payload 內若自帶指令名(如 {"cmd": ["GREP", ...]}),以它為準
    for s in strings:
        if s.strip().upper() in _CMD_NAMES:
            kind = s.strip().upper()
    args = [s for s in strings if s.strip().upper() not in _CMD_NAMES]

    if kind in ("GREP", "VECTOR"):
        return (kind, _unquote(args[0])) if args else None
    if kind == "READ":
        sid = next((s for s in args if ":" in s), None)
        k = next((i for i in ints if 0 < i <= 10), 2)
        return ("READ", f"{_unquote(sid)} {k}") if sid else None
    return ("FINAL", " ".join(s for s in args if ":" in s))


def _parse_harmony_loose(reply: str) -> tuple[str, str] | None:
    """最後手段:`to=GREP <|constrain|>="pattern"` 這類無 JSON 的變體。
    取 to=CMD 之後的行尾,剝掉 harmony 標記與 constrain/json 雜訊當參數。"""
    m = None
    for line in reply.splitlines():
        for m2 in _HARMONY_LOOSE_RE.finditer(line):
            m = m2
    if m is None:
        return None
    kind = m.group(1).upper()
    tail = re.sub(r"<\|[^|]*\|>", " ", m.group(2))
    tail = re.sub(r"\b(?:json|commentary|response)\b", " ", tail, flags=re.IGNORECASE)
    tail = tail.strip().lstrip("=").strip()
    arg = _unquote(tail)
    if kind in ("GREP", "VECTOR"):
        return (kind, arg) if arg else None
    if kind == "READ":
        sid = next((s for s in re.split(r"[,\s]+", arg) if ":" in s), None)
        return ("READ", f"{_unquote(sid)} 2") if sid else None
    return ("FINAL", arg)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        return s[1:-1]
    return s


def _parse_command(reply: str) -> tuple[str, str] | None:
    """從回覆的最後幾行解析出指令(允許指令前有 reasoning);
    plain-text 協議優先,退而解析 harmony 原生 tool-call 語法。
    harmony 標記(<|channel|> 等)先換成換行,讓塞在 <|message|> 後的
    plain-text 指令也能被行解析抓到。"""
    sanitized = re.sub(r"<\|[^|]*\|>", "\n", reply)
    # 120B(gpt-oss)harmony 雙通道黏連:多個指令擠同一行且整段重複
    # ("GREP MelanieGREP Melanie"、"READ 0__2:0 5FINAL 0__2:0")——行首是
    # GREP 時整行被當 pattern、行尾 FINAL 被吃掉,L1 失靈率 49.6% 的根因
    # (2026-07-06)。小寫/數字/引號後緊跟大寫指令字 → 斷行;20B 逐行輸出不受影響。
    sanitized = re.sub(r"(?<=[a-z0-9\"'\)\].:])((?:GREP|READ|VECTOR)\s|FINAL\b)",
                       r"\n\1", sanitized)
    for line in reversed(sanitized.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        if m := _FINAL_RE.match(line):
            if not m.group(1).strip():
                # 空 FINAL:若同回覆還有其他可執行指令,先執行它們
                # (120B 常把整段計畫 GREP..READ..FINAL 一次吐完)
                continue
            return ("FINAL", m.group(1))
        if m := _GREP_RE.match(line):
            return ("GREP", _unquote(m.group(1)))
        if m := _READ_RE.match(line):
            return ("READ", f"{_unquote(m.group(1))} {m.group(2) or 2}")
        if m := _VECTOR_RE.match(line):
            return ("VECTOR", _unquote(m.group(1)))
    return _parse_harmony(reply) or _parse_harmony_loose(reply)


def _candidates_block(corpus: Corpus, sids: list[str]) -> str:
    lines = []
    for s in sids:
        entry = corpus.display_entry(s, max_chars=400)
        if entry is None:
            lines.append(f"[sid={s}] (raw text unavailable)")
        else:
            t = corpus.resolve(s)[0]
            lines.append(f"[sid={s}] [{t.date}] {entry}")
    return "\n".join(lines) if lines else "(none)"


def _coverage_tokens(text: str) -> set[str]:
    """Extract small, dependency-free lexical evidence signatures.

    This is intentionally generic: it does not classify the question or
    assign special handling to temporal/counting/category labels.
    """
    return {
        tok.lower()
        for tok in _COVERAGE_TOKEN_RE.findall(text or "")
        if tok.lower() not in _COVERAGE_STOPWORDS
    }


def _portfolio_pad(
    *,
    question: str,
    corpus: Corpus,
    seed_sids: list[str],
    selected_sids: list[str],
    target_size: int,
    seed_scores: dict[str, float],
) -> tuple[list[str], dict]:
    """Add a diverse evidence portfolio using generic lexical MMR signals.

    Candidates remain restricted to upstream seed sids.  At each step the
    candidate with the best combination of question overlap, new lexical
    coverage, distance from already selected evidence, and upstream score is
    added.  No question category or question-shape rule is used.
    """
    target_size = max(0, target_size)
    selected = list(dict.fromkeys(selected_sids))
    selected_set = set(selected)
    seed_order = list(dict.fromkeys(seed_sids))
    q_tokens = _coverage_tokens(question)
    signatures = {
        s: _coverage_tokens(corpus.display_entry(s, max_chars=4000) or "")
        for s in seed_order
    }
    groups = {s: s.split(":", 1)[0] for s in seed_order}
    covered = set().union(*(signatures.get(s, set()) for s in selected))
    covered_groups = {groups[s] for s in selected if s in groups}
    additions: list[str] = []
    scores: dict[str, float] = {}

    while len(selected) < target_size:
        remaining = [s for s in seed_order if s not in selected_set]
        if not remaining:
            break

        best = None
        best_key = None
        for sid in remaining:
            tokens = signatures.get(sid, set())
            if not tokens:
                continue
            overlap = len(tokens & q_tokens)
            novelty = len(tokens - covered)
            group_gain = 1.0 if groups.get(sid) not in covered_groups else 0.0
            # Penalize redundancy against the most similar selected item.
            max_similarity = 0.0
            for picked in selected:
                other = signatures.get(picked, set())
                union = tokens | other
                if union:
                    max_similarity = max(max_similarity, len(tokens & other) / len(union))
            diversity = 1.0 - max_similarity
            rerank = float(seed_scores.get(sid, 0.0))
            # Lexical question overlap is the relevance guard; novelty and
            # diversity prevent the portfolio from collapsing to near copies.
            score = 2.0 * overlap + 0.25 * novelty + diversity + 1.5 * group_gain + 0.5 * rerank
            key = (score, group_gain, novelty, overlap, rerank, -seed_order.index(sid))
            if best_key is None or key > best_key:
                best_key = key
                best = sid

        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        additions.append(best)
        scores[best] = round(float(best_key[0]), 4)
        covered.update(signatures.get(best, set()))
        covered_groups.add(groups.get(best, best))

    return selected, {
        "added": additions,
        "candidate_count": len(seed_order),
        "selected_count": len(selected),
        "covered_token_count": len(covered),
        "covered_group_count": len(covered_groups),
        "scores": scores,
    }


def _vector_search(
    artifact_dir,
    corpus: Corpus,
    query: str,
    *,
    exclude: set[str],
    topn: int,
    min_score: float,
) -> str:
    """VECTOR 指令的執行端:query embed 後查該題 summaries VDB,
    回傳 inline 候選清單(與 GREP 同格式,agent 需自行 READ/GREP 驗證)。"""
    from experiment.agent_filter.gap_vector import vector_gap_candidates

    cands = vector_gap_candidates(
        artifact_dir, query, exclude=exclude, topn=topn, min_score=min_score,
    )
    if not cands:
        return (f"vector {query!r}: 0 hits above threshold. "
                "Try rephrasing the query, or fall back to GREP with rare literal words.")
    lines = [f"vector {query!r}: {len(cands)} semantically similar turns "
             "(NOT verified — check with READ/GREP before including)"]
    for s, sc in cands:
        turns = corpus.resolve(s)
        entry = corpus.display_entry(s, max_chars=200) or "(text unavailable)"
        dt = f"[{turns[0].date}] " if turns and turns[0].date else ""
        lines.append(f"[sid={s}] (score={sc:.2f}) {dt}{entry}")
    return "\n".join(lines)


def _rebuild_context(
    context: str,
    corpus: Corpus,
    final_sids: list[str],
    *,
    include_pair: bool = True,
    include_prefix: bool = True,
) -> tuple[str, list[str]]:
    """重組 Evidence Summary block。include_pair=True 時,選中 sid 的 pair 夥伴
    (同一個 user↔assistant exchange 的另一側)一併帶入 — agent 有時會選到正確
    pair 的錯誤一側,成對呈現能保住關鍵證據。回傳 (context, context_sids)。"""
    idx = context.find(_EVIDENCE_HEADER)
    if include_prefix:
        head = context[:idx].rstrip("\n") if idx != -1 else context.rstrip("\n")
    else:
        head = ""
    lines = [head, _EVIDENCE_HEADER] if head else [_EVIDENCE_HEADER]

    entries: list[str] = []  # sid(pair base 或 split sid),保序去重
    seen: set[str] = set()
    for s in final_sids:
        key = s.rsplit(":", 1)[0] if include_pair and (s.endswith(":u") or s.endswith(":a")) else s
        if key not in seen:
            seen.add(key)
            entries.append(key)

    context_sids: list[str] = []
    for key in entries:
        turns = corpus.resolve(key)
        if not turns:
            continue
        context_sids.extend(t.sid for t in turns)
        sid_tags = "".join(f"[sid={t.sid}]" for t in turns)
        entry = corpus.display_entry(key, max_chars=4000 * len(turns))
        dt_str = f"[{turns[0].date}]" if turns[0].date else ""
        lines.append(f"  • {dt_str}{sid_tags}[score=--] {entry} ")
    return "\n".join(lines), context_sids


def _check_sufficiency(
    llm,
    *,
    question: str,
    question_date: str | None,
    corpus: Corpus,
    sids: list[str],
) -> tuple[bool, str]:
    """獨立審計 call:證據夠不夠完整回答?回傳 (sufficient, missing_desc)。
    解析失敗時視為 sufficient(不要因 verifier 抽風而空轉)。"""
    lines = []
    for s in sids:
        t = corpus.resolve(s)
        if not t:
            continue
        # 必須與最終 context 同樣完整(4000/側):verifier 看截斷版會把
        # 「埋在長 turn 深處的細節」誤判成缺失(實測 42% 誤觸發的主因)。
        entry = corpus.display_entry(s, max_chars=4000 * len(t))
        lines.append(f"[{t[0].date}] {entry}")
    reply_msgs = [
        {"role": "system", "content": SUFFICIENCY_SYSTEM},
        {"role": "user", "content": SUFFICIENCY_USER.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            evidence="\n".join(lines) or "(none)",
        )},
    ]
    resp = llm.chat(messages=reply_msgs, temperature=0.0, max_tokens=512)
    reply = (resp.choices[0].message.content or "").strip()
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\**\s*INSUFFICIENT\s*\**\s*[::]?\s*(.*)$", line, re.IGNORECASE)
        if m:
            return False, (m.group(1) or reply[:300]).strip()
        if re.match(r"^\**\s*SUFFICIENT\b", line, re.IGNORECASE):
            return True, ""
    return True, ""


def _adjudicate_candidates(
    llm,
    *,
    question: str,
    question_date: str | None,
    corpus: Corpus,
    pending: list[str],
) -> tuple[list[str], dict]:
    """Answer-blind 逐條裁決:獨立 call(無 agent 搜尋歷史,看不到 agent 已推出
    的答案),對每條被 FINAL 丟掉的 seed 判 KEEP/DROP。判準=與問題主題相關,
    非「含答案」。回傳 (KEEP 的 sids, 逐條 verdict dict)。未給 verdict 的
    候選視為 DROP(裁決是 add-only 的回收,不裁決=不補回)。"""
    from experiment.agent_filter.prompts import (
        ADJUDICATE_SYSTEM,
        ADJUDICATE_USER,
    )
    lines = []
    for s in pending:
        t = corpus.resolve(s)
        if not t:
            continue
        entry = corpus.display_entry(s, max_chars=700)
        dt = f"[{t[0].date}] " if t[0].date else ""
        lines.append(f"[sid={s}] {dt}{entry}")
    msgs = [
        {"role": "system", "content": ADJUDICATE_SYSTEM},
        {"role": "user", "content": ADJUDICATE_USER.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            n=len(lines),
            candidates="\n".join(lines) or "(none)",
        )},
    ]
    # reasoning model:verdict 前有隱藏思考,token 預算要夠——2048 實測會把
    # 14 條 verdict 輸出到一半截斷(child run: 129 條 unjudged 全是這個)。
    resp = llm.chat(messages=msgs, temperature=0.0, max_tokens=4096)
    reply = (resp.choices[0].message.content or "").strip()
    kept: list[str] = []
    # reply=裁決 call 原始回覆(<sid> KEEP|DROP <short reason> 逐行);保留全文供
    # 前端還原裁決軌跡,並逐條抽 verdict+reason(reasons: sid → "KEEP|DROP: 理由")。
    verdicts: dict = {
        "kept": [], "dropped": [], "unjudged": [],
        "reply": reply, "reply_chars": len(reply), "reasons": {},
    }
    judged: set[str] = set()
    for line in reply.splitlines():
        m = re.search(r"\b(KEEP|DROP)\b", line, re.IGNORECASE)
        if not m:
            continue
        for s in pending:
            if s in judged or s not in line:
                continue
            judged.add(s)
            decision = m.group(1).upper()
            # 抽該行 KEEP/DROP 之後的短理由(去掉 sid 與 verdict token)
            reason = line[m.end():].strip(" \t:-—．。")
            verdicts["reasons"][s] = f"{decision}: {reason}" if reason else decision
            if decision == "KEEP":
                kept.append(s)
                verdicts["kept"].append(s)
            else:
                verdicts["dropped"].append(s)
            break
    verdicts["unjudged"] = [s for s in pending if s not in judged]
    return kept, verdicts


def refine_context(
    *,
    question: str,
    context: str,
    csv_path: str | Path,
    llm,
    question_date: str | None = None,
    category: str | None = None,
    params: dict | None = None,
    artifact_dir: str | Path | None = None,
    corpus: Corpus | None = None,
) -> tuple[str, dict]:
    """跑 grep agent,回傳 (refined_context, trace)。任何失敗都退回原 context。
    corpus 可外部預建(如 LoCoMo chunk 級 corpus);未提供時從 csv_path 載入。"""
    p = params or {}
    mode = p.get("grep_agent_mode", "filter_fetch")
    max_calls = int(p.get("grep_agent_max_calls", 8))
    max_sids = int(p.get("grep_agent_max_sids", 16))
    grep_max_lines = int(p.get("grep_agent_grep_max_lines", 30))
    include_filter_graph = bool(p.get("grep_agent_filter_include_graph_context", False))
    include_answer_graph = bool(p.get("grep_agent_answer_include_graph_context", True))

    trace: dict = {"enabled": True, "mode": mode, "commands": [], "fallback": None}
    try:
        if corpus is None:
            corpus = load_corpus(csv_path)
        seed = seed_sids_from_context(context)
        trace["seed_sids"] = seed
        trace["seed_scores"] = seed_scores_from_context(context)  # sid → rerank 分數
        graph_context = graph_context_from_context(
            context,
            max_chars=int(p.get("grep_agent_graph_context_max_chars", 12000)),
        )
        trace["graph_context_available"] = bool(graph_context)
        trace["filter_graph_context"] = include_filter_graph
        trace["answer_graph_context"] = include_answer_graph
        if not seed and mode == "filter":
            trace["fallback"] = "no_seed"
            return context, trace

        if category is None:
            category = Path(csv_path).parent.name
        # _abs 棄答題(答案不在語料):force_verified_final 對它們必須保留全量
        # 保護色不窄化——fvf-73 實測窄化(即使 verified 多)誘使模型放棄棄答改口。
        is_abstention = bool(csv_path) and Path(csv_path).stem.endswith("_abs")
        trace["is_abstention"] = is_abstention
        # Skill 庫(question-shape 驅動)優先;沒命中才退回 category hint
        hint = ""
        if p.get("grep_agent_use_skills", False):
            from experiment.agent_filter.skills import select_skills
            matched = select_skills(question)
            trace["skills"] = [n for n, _ in matched]
            hint = "\n\n".join(s for _, s in matched)
        if not hint:
            hint = CATEGORY_HINTS.get(category, "")

        # VECTOR 工具:該題 summaries VDB 在場才開(agent 自主決定何時語意搜尋;
        # 與已證偽的 gap_vector「verifier 推候選」不同——這裡是 agent 主動拉取)。
        vector_ok = (
            bool(p.get("grep_agent_vector_search", True))
            and artifact_dir is not None
            and (Path(artifact_dir) / "summaries_chroma").exists()
        )
        trace["vector_tool"] = vector_ok
        # Provenance: VECTOR results are discovery-only. A sid is verified only
        # after it appears in a raw GREP or READ result.
        verified_sids: set[str] = set()
        vector_candidate_sids: set[str] = set()
        trace["evidence_provenance"] = {}
        from experiment.agent_filter.prompts import (
            VECTOR_TOOL_BLOCK, _active_hypothesis_block,
        )
        emit_hyp = bool(int(p.get("grep_agent_emit_hypothesis", 0)))
        system = SYSTEM_PROMPT.format(
            max_calls=max_calls,
            vector_tool=VECTOR_TOOL_BLOCK if vector_ok else "",
            hypothesis_line=_active_hypothesis_block() if emit_hyp else "",
        )
        if mode == "filter":
            system += "\nIMPORTANT: you may only KEEP or DROP candidates; do not add new sids in FINAL."
        user = USER_TEMPLATE.format(
            question=question,
            date_line=f"QUESTION DATE: {question_date}\n" if question_date else "",
            hint_line=f"{hint}\n" if hint else "",
            graph_context=(
                "GRAPH FACTS (Entities/Relationships; use as supporting evidence):\n"
                + graph_context
                if include_filter_graph and graph_context else ""
            ),
            candidates=_candidates_block(corpus, seed),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _extract_final(arg: str, full_reply: str = "") -> list[str]:
            sids = _SID_RE.findall(arg) + [
                s for s in re.split(r"[,\s]+", _SID_RE.sub("", arg)) if s and ":" in s
            ]
            if not sids and full_reply:
                # FINAL 參數空:sid 可能寫在其他行(FINAL: 後換行列點等)
                sids = _SID_RE.findall(full_reply) + [
                    s.strip("*•-,.")
                    for s in re.split(r"[,\s]+", _SID_RE.sub("", full_reply))
                    if ":" in s and re.search(r":\d+", s)
                ]
            return sids

        def _parse_response(resp):
            """Parse a response without feeding hidden reasoning back to the model."""
            candidates = _response_command_candidates(resp)
            raw_reply = candidates[0][1] if candidates else ""
            for source, candidate in candidates:
                cmd = _parse_command(candidate)
                if cmd is not None:
                    return raw_reply, candidate, cmd, source
            return raw_reply, raw_reply, None, None

        def _run_loop(budget: int) -> list[str] | None:
            """跑一輪 GREP/READ→FINAL 的 tool loop(主搜尋與 verify 補搜共用)。"""
            parse_failures = 0
            repeat_count = 0
            prev_cmd: tuple[str, str] | None = None
            for _ in range(budget):
                _t0 = time.perf_counter()
                resp = llm.chat(messages=messages, temperature=0.0, max_tokens=1024)
                diag = _resp_diag(resp)
                raw_reply, reply, cmd, source = _parse_response(resp)
                if source:
                    diag["command_source"] = source
                # Only retain the compact command in conversation history.
                # Raw reasoning is diagnostic data, not useful context, and
                # replaying it would inflate the next request's input tokens.
                messages.append({
                    "role": "assistant",
                    "content": f"{cmd[0]} {cmd[1]}" if cmd else "",
                })
                if cmd is None:
                    parse_failures += 1
                    trace["commands"].append({"cmd": "PARSE_FAIL", "arg": raw_reply[:200],
                                              "ms": round((time.perf_counter() - _t0) * 1000),
                                              **diag})
                    if parse_failures >= 2:
                        return None
                    messages.append({"role": "user", "content":
                        "Could not parse a command. Reply with exactly one command as the "
                        "last line: GREP <regex> | READ <sid> [k] | FINAL <sid> ..."})
                    continue

                kind, arg = cmd
                if kind == "FINAL":
                    if emit_hyp:
                        # agent 自報的答案假說(生產化「假說回收」,取代 hyp-v1 的
                        # 4o-mini 事後抽取)。從 reply 找 HYPOTHESIS: 行;找不到就
                        # 退回整段 reasoning(reasoning_content 或 reply)供下游。
                        # 只抓 HYPOTHESIS 到「行尾」;若同行/相鄰接了 FINAL 或 sid
                        # token(agent 把兩者寫在一起),在 FINAL 前截斷,避免把 FINAL
                        # 行的 sids 吞進 hypothesis(hyp-v1 06db6396、120b filter 實見)。
                        hm = re.search(r"HYPOTHESIS\s*[::]\s*([^\n]+)", reply, re.IGNORECASE)
                        hyp = hm.group(1).strip() if hm else ""
                        hyp = re.split(r"\bFINAL\b", hyp, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                        if hyp and hyp.upper() != "NONE":
                            trace["hypothesis"] = hyp[:200]
                    trace["commands"].append({"cmd": "FINAL", "arg": arg[:500],
                                              "reply": reply[:1200],
                                              "ms": round((time.perf_counter() - _t0) * 1000),
                                              **diag})
                    return _extract_final(arg, reply)

                # 複讀機斷路器:同一指令連續重複就催收,三次直接跳強制 FINAL
                if cmd == prev_cmd:
                    repeat_count += 1
                    if repeat_count >= 2:
                        trace["commands"].append({"cmd": "REPEAT_BREAK", "arg": f"{kind} {arg}"[:200],
                                                  "ms": round((time.perf_counter() - _t0) * 1000)})
                        return None
                    messages.append({"role": "user", "content":
                        "You already ran that exact command. Try DIFFERENT keywords, "
                        "or reply FINAL <sid> ... with your current best selection."})
                    continue
                prev_cmd = cmd
                repeat_count = 0
                if kind == "GREP":
                    result = corpus.grep(arg, max_lines=grep_max_lines)
                    verified_sids.update(corpus.normalize_sids(_SID_RE.findall(result)))
                elif kind == "VECTOR":
                    if vector_ok:
                        result = _vector_search(
                            artifact_dir, corpus, arg,
                            exclude=set(corpus.normalize_sids(seed)),
                            topn=int(p.get("grep_agent_vector_topn", 8)),
                            min_score=float(p.get("grep_agent_vector_min_score", 0.30)),
                        )
                        _vhits = corpus.normalize_sids(_SID_RE.findall(result))
                        vector_candidate_sids.update(_vhits)
                        # VECTOR 命中直接視為 verified(與 GREP/READ 同級)。
                        # provenance gate 已移除:撈回來即可信,不再要求二次驗證。
                        verified_sids.update(_vhits)
                    else:
                        result = "VECTOR is not available for this question; use GREP or READ."
                else:  # READ
                    sid, k = arg.rsplit(" ", 1)
                    result = corpus.read_window(sid, k=int(k))
                    verified_sids.update(corpus.normalize_sids(_SID_RE.findall(result)))
                trace["commands"].append({"cmd": kind, "arg": arg[:300], "result_chars": len(result),
                                          "reply": reply[:1200], "result": result[:1500],
                                          "ms": round((time.perf_counter() - _t0) * 1000),
                                          **diag})
                # 收尾提醒常駐於最近 context — 模型在多輪搜尋後常忘記怎麼結束。
                # 明示 partial FINAL 可接受:49/73 fallback 是彙整題「湊不齊全部
                # 實例不敢交」打滿輪次;裁決層本來就會補漏,不必追求收集完整。
                result += ("\n\n(When you have identified the evidence, reply with one line: "
                           "FINAL <sid> <sid> ... — copy sids exactly. A PARTIAL set is "
                           "acceptable: FINAL the turns you have confirmed so far — a separate "
                           "audit step recovers anything you miss. Do not keep searching for "
                           "completeness.)")
                messages.append({"role": "user", "content": result})
            return None

        final_raw = _run_loop(max_calls)

        _SALVAGE_MSGS = [
            "STOP searching. Reply NOW with only one line listing the selected evidence "
            "sids, copied EXACTLY from this list (or ones you found via GREP):\n{seeds}\n"
            "FINAL <sid> <sid> ...",
            "Output ONLY the single line below, filled in with sids from this list — "
            "no other text, no tool calls:\n{seeds}\nFINAL <sid> <sid> ...",
        ]

        def _ask_final(attempt: int) -> list[str]:
            """強制收尾:附上候選 sid 清單讓模型照抄,提取 sid。"""
            messages.append({"role": "user", "content":
                _SALVAGE_MSGS[min(attempt, 1)].format(seeds=" ".join(seed))})
            _t0 = time.perf_counter()
            resp = llm.chat(messages=messages, temperature=0.0, max_tokens=512)
            diag = _resp_diag(resp)
            raw_reply, reply, cmd, source = _parse_response(resp)
            if source:
                diag["command_source"] = source
            messages.append({
                "role": "assistant",
                "content": f"{cmd[0]} {cmd[1]}" if cmd else "",
            })
            arg = cmd[1] if cmd and cmd[0] == "FINAL" else ""
            out = _extract_final(arg, reply)
            trace["commands"].append({"cmd": "FINAL(forced)", "arg": (arg or reply)[:500],
                                      "reply": reply[:1200],
                                      "ms": round((time.perf_counter() - _t0) * 1000),
                                      **diag})
            return out

        if not (final_raw and corpus.normalize_sids(final_raw)):
            # 沒有 FINAL / FINAL 空 / sid 無法解析 → 催收一次。二度催收無用
            # (實測 129/138 次模型仍回工具呼叫);用 verified 命中收窄 context
            # 也驗證為負(salvage 組 54→43%:窄而「可信」的 context 反而誘使
            # 答題模型亂編,全量 16 條的雜訊對難題反而是保護色)。
            if final_raw:
                trace["final_raw_unresolved"] = final_raw[:20]
            final_raw = _ask_final(0)

        # 不確定訊號:agent 拒絕自主 FINAL = 它找不到能確認的答案證據
        # (_abs 棄答題 fallback 率 70%)。hint 為條件式棄答提示,只掛在
        # 全量回退路徑——「收窄 context + hint」已驗證為負(一般題 46.7→33.3)。
        abstain = False
        if not (final_raw and corpus.normalize_sids(final_raw)):
            # 強制 verified→FINAL:agent 不肯自主收尾時,把它 GREP/READ 實際
            # 確認過的 verified sids 當 FINAL 走完整 finalize pipeline(裁決+floor
            # +rebuild),而非退回全量 raw context 標記 fallback。設計依據:
            #   - max-calls 拉高測試證偽「逼 agent 交砍過的窄 context」(-6),因為
            #     bare 1-2 sid FINAL 讓答題模型脫離全量雜訊保護色而亂編。
            #   - 但 verified→FINAL 不同:彙整題 verified 常 >16 條(agent GREP 過
            #     數十 turn),經 cap 後 ≈ 全量大小、只是 verified-first 重排;搜空題
            #     verified=0 → 回退全 seed = 保住保護色。兩端都不會裸窄化。
            #   - 走 finalize_from_raw 保留 adjudicate 補回主題相關 seed + floor pad
            #     回 12,provenance 完整,no_final 標記消失。
            # Gate:只有 agent 確認的 verified 證據「夠多」才走 finalize 窄化;
            # verified 不足(含 _abs 搜空題、彙整題只湊到零星幾條)→ 保留全量
            # 保護色(fvf-73 測試:窄化傷害全集中在 verified 少的 _abs,verified
            # 大的題全部安全或改善)。門檻預設=12(即「至少湊到一份量的確認
            # 證據」才視為可信到能取代全量)。注:evidence_floor 盲補已於
            # 2026-07-20 停用,此處不再借用其值,fallback 直接寫 12。
            verified_norm = corpus.normalize_sids(list(verified_sids))
            fvf_min = int(p.get("grep_agent_force_verified_min", 12))
            # 窄化條件:啟用 flag + 非棄答題 + verified 夠多。任一不滿足 → 保全量。
            if (int(p.get("grep_agent_force_verified_final", 0))
                    and not is_abstention
                    and len(verified_norm) >= fvf_min):
                trace["forced_verified_final"] = verified_norm
                return finalize_from_raw(
                    final_raw=verified_norm,
                    context=context, corpus=corpus, seed=seed,
                    verified_sids=verified_sids,
                    vector_candidate_sids=vector_candidate_sids,
                    question=question, question_date=question_date, category=category,
                    llm=llm, p=p, artifact_dir=artifact_dir, trace=trace,
                )
            trace["fallback"] = "no_final"
            trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
            if bool(p.get("grep_agent_abstention_hint", 0)):
                trace["abstention_hint"] = True
                return context + ABSTENTION_HINT, trace
            return context, trace

        final = corpus.normalize_sids(final_raw)
        if mode == "filter":
            seed_set = set(corpus.normalize_sids(seed))
            final = [s for s in final if s in seed_set]
        elif mode == "fetch_only":
            # 只補不砍:保住 baseline context 的 serendipity,吃下 agent 的補撈 recall。
            # LoCoMo 實測:agent 全中率 +19.8pp 但砍掉非 gold 有用內容抵銷收益 → 此模式解耦。
            seed_norm_ = corpus.normalize_sids(seed)
            final = seed_norm_ + [s for s in final if s not in set(seed_norm_)]

        # Provenance gate 已移除(2026-07-22):VECTOR 命中現與 GREP/READ 同視為
        # verified(見 VECTOR branch),GREP/READ/VECTOR 撈回來的證據一律信任;
        # 唯一未驗證的只剩幻覺 sid,會在 _rebuild_context 因 corpus 解析不到而自然丟棄。
        trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
        trace["vector_candidate_sids"] = corpus.normalize_sids(list(vector_candidate_sids))
        seed_set = set(corpus.normalize_sids(seed))
        trace["evidence_provenance"] = {
            s: (
                "seed+verified" if s in seed_set and s in verified_sids
                else "seed" if s in seed_set
                else "verified" if s in verified_sids
                else "unverified"
            )
            for s in final
        }
        trace["final_before_cap"] = list(final)  # 截斷前(診斷 top-k truncation 用)
        final = final[:max_sids]
        if not final:
            trace["fallback"] = "empty_final"
            return context, trace

        # ── Sufficiency 迴圈:verifier 判證據不足 → 帶缺口 hint 補搜,只加不刪 ──
        verify_rounds = int(p.get("grep_agent_verify_rounds", 0))
        verify_budget = int(p.get("grep_agent_verify_max_calls", 4))
        verify_cats = p.get("grep_agent_verify_categories")
        if verify_cats is not None and category not in verify_cats:
            verify_rounds = 0  # 選擇性啟動:非彙整型類別跳過(verify 只會稀釋)
        trace["sufficiency"] = []
        for vr in range(verify_rounds):
            try:
                ok, missing = _check_sufficiency(
                    _verify_llm(llm), question=question, question_date=question_date,
                    corpus=corpus, sids=final,
                )
            except Exception as exc:  # verifier 掛掉不影響主流程
                trace["sufficiency"].append({"round": vr, "error": str(exc)[:200]})
                break
            trace["sufficiency"].append({"round": vr, "sufficient": ok, "missing": missing[:300]})
            if ok:
                break
            gap_msg = GAP_HINT_TEMPLATE.format(missing=missing)
            # 向量補搜:grep 的修復臂常因 paraphrase gap 空手(~87%),把缺口
            # 描述 embed 後查 summaries VDB,撈語意近鄰給 agent 確認。
            gap_topn = int(p.get("grep_agent_gap_vector_topn", 0))
            if artifact_dir is not None and gap_topn > 0:
                from experiment.agent_filter.gap_vector import vector_gap_candidates
                cands = vector_gap_candidates(
                    artifact_dir,
                    f"{question}\n{missing}",
                    exclude=set(final),
                    topn=gap_topn,
                    min_score=float(p.get("grep_agent_gap_vector_min_score", 0.30)),
                )
                trace["sufficiency"][-1]["vector_cands"] = [s for s, _ in cands]
                if cands:
                    lines = []
                    for s, sc in cands:
                        entry = corpus.display_entry(s, max_chars=200) or "(text unavailable)"
                        lines.append(f"[sid={s}] (score={sc:.2f}) {entry}")
                    gap_msg += (
                        "\n\nA semantic search for the missing information surfaced these "
                        "candidate turns (NOT yet verified — check with READ/GREP before "
                        "including):\n" + "\n".join(lines)
                    )
            messages.append({"role": "user", "content": gap_msg})
            extra_raw = _run_loop(verify_budget)
            if not extra_raw:
                break
            extra = corpus.normalize_sids(extra_raw)
            if mode == "filter":
                extra = [s for s in extra if s in set(corpus.normalize_sids(seed))]
            # 單調遞增:verify 輪只准加,不准動已選的
            added_now = [s for s in extra if s not in set(final)]
            trace["sufficiency"][-1]["added"] = added_now
            if not added_now:
                break
            final = (final + added_now)[:max_sids]

        seed_norm = corpus.normalize_sids(seed)

        # ── Answer-blind 逐條裁決:agent 的 FINAL 是「答案引用」(minimal-
        # citation 天性:先解題→只留含 answer span 的最小 turn 集),不含
        # answer span 的支持證據被系統性丟棄(preference/multi-hop 病灶)。
        # 對策:獨立裁決 call(看不到 agent 對話,故不知道「答案」)對每條被
        # 丟掉的 seed 逐一 KEEP/DROP,判準=與問題主題相關。KEEP 補回 final
        # (只加不刪,agent 自選的 0.84 precision 不動)。裁決成功時取代
        # evidence_floor 的盲補——floor 按 rerank 原序回填,補不回偏好線索。
        adj_on = int(p.get("grep_agent_adjudicate", 0))
        adj_cats = p.get("grep_agent_adjudicate_categories")
        if adj_cats is not None and category not in adj_cats:
            adj_on = 0
        # KEEP-all 類別:KU/temporal 的 gold 含大量「不含答案但推理必需」的支撐
        # turn(時間錨點、dated mention),20B 裁決用「含答案」判準系統性 DROP
        # 掉它們(B 桶病因)。這些類改成 recall-recovery-only:被丟的 seed 全補
        # 回不經 LLM DROP。與 min-keep 不同——這是類別級「不砍」,非問題形狀觸發。
        keep_all_cats = p.get("grep_agent_adjudicate_keep_all_categories")
        adjudicated = False
        if adj_on and len(final) < max_sids:
            pending = [s for s in seed_norm if s not in set(final)]
            if pending and keep_all_cats and category in keep_all_cats:
                trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
                final = (final + [s for s in pending if s not in set(final)])[:max_sids]
                adjudicated = True
            elif pending:
                _t0 = time.perf_counter()
                try:
                    kept_adj, verdicts = _adjudicate_candidates(
                        llm, question=question, question_date=question_date,
                        corpus=corpus, pending=pending,
                    )
                    verdicts["ms"] = round((time.perf_counter() - _t0) * 1000)
                    trace["adjudication"] = verdicts
                    final = (final + [s for s in kept_adj
                                      if s not in set(final)])[:max_sids]
                    adjudicated = True
                except Exception as exc:  # 裁決掛掉不影響主流程,floor 續行
                    trace["adjudication"] = {"error": str(exc)[:200]}

        # ── Min-keep(問題驅動,非類別特化):彙整/最新值型問題(how many/
        # how often/total/most recent/current...)需要目標事實的所有 dated
        # mentions 共存(計數要全、取最新要能比)。agent 砍到太薄時依 rerank
        # 原序從 seed 補滿——由問題形狀觸發,對任何資料集通用。
        min_keep = int(p.get("grep_agent_min_keep_aggregation", 0))
        if min_keep and len(final) < min_keep and _AGG_QUESTION_RE.search(question):
            pad = [s for s in seed_norm if s not in set(final)]
            trace["min_keep_padded"] = pad[: min_keep - len(final)]
            final = final + pad[: min_keep - len(final)]

        # ── evidence_floor 盲補已於 2026-07-20 停用 ────────────────────────
        # 按 rerank 原序硬塞、繞過 agent 決定,對 accuracy 零貢獻、只讓 kept
        # 定性失真(見 experiment_config.grep_agent_evidence_floor 說明)。整段
        # 註解保留以備考;grep_agent_evidence_floor 預設已為 0。
        # evidence_floor = int(p.get("grep_agent_evidence_floor", 0))
        # if adjudicated:
        #     evidence_floor = 0  # 裁決已做過 informed 的逐條決定,盲補只會稀釋
        # if evidence_floor > 0 and len(final) < evidence_floor:
        #     final, coverage = _portfolio_pad(
        #         question=question,
        #         corpus=corpus,
        #         seed_sids=seed_norm,
        #         selected_sids=final,
        #         target_size=min(evidence_floor, max_sids),
        #         seed_scores=seed_scores_from_context(context),
        #     )
        #     trace["evidence_coverage"] = coverage
        #     if coverage["added"]:
        #         trace["evidence_floor_padded"] = coverage["added"]

        trace["final_sids"] = final
        trace["kept"] = [s for s in final if s in set(seed_norm)]
        trace["added"] = [s for s in final if s not in set(seed_norm)]
        trace["dropped"] = [s for s in seed_norm if s not in set(final)]

        # Safety net: if the selector dropped every upstream summary, keep the
        # original 16-summary context instead of answering from agent-fetched
        # evidence alone.  This is distinct from a no_final/exception
        # fallback: the agent completed, but produced zero retained seeds.
        if not trace["kept"]:
            trace["fallback"] = "zero_keep"
            trace["context_sids"] = seed_norm
            if abstain:
                return context + ABSTENTION_HINT, trace
            return context, trace

        if mode == "fetch_only":
            # 純附加:原 context 一字不動(baseline 證據可能是無 sid 的純文字,
            # rebuild 會誤刪),只把補撈到的單位掛在後面。保證資訊 ≥ baseline。
            if not trace["added"]:
                trace["fallback"] = "no_addition"
                return context, trace
            lines = ["", "### Additional Evidence (agent-retrieved)"]
            ctx_sids = list(seed_norm)
            for s in trace["added"]:
                t = corpus.resolve(s)[0]
                entry = corpus.display_entry(s)
                dt = f"[{t.date}]" if t.date else ""
                lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
                ctx_sids.append(s)
            trace["context_sids"] = ctx_sids
            return context.rstrip("\n") + "\n".join(lines), trace

        refined, context_sids = _rebuild_context(
            context, corpus, final,
            include_pair=bool(p.get("grep_agent_include_pair", True)),
            include_prefix=include_answer_graph,
        )
        trace["context_sids"] = context_sids
        if abstain:
            refined += ABSTENTION_HINT
        return refined, trace

    except Exception:
        trace["fallback"] = "exception"
        trace["error"] = traceback.format_exc()[-2000:]
        return context, trace


def finalize_from_raw(
    *,
    final_raw,
    context: str,
    corpus: Corpus,
    seed: list[str],
    verified_sids: set,
    vector_candidate_sids: set,
    question: str,
    question_date: str | None,
    category: str | None,
    llm,
    p: dict,
    artifact_dir,
    trace: dict,
) -> tuple[str, dict]:
    """v1 後段 pipeline(provenance gate → filter_fetch → adjudicate → floor →
    rebuild)抽成獨立函式,供 planner-worker harness 復用同一套 v1 主線邏輯。

    行為與 refine_context 的 811-1011 段一致,唯一差異:sufficiency 迴圈需要
    v1 的 _run_loop(planner-worker 沒有),故此路徑跳過 sufficiency(v1 預設
    verify_rounds=0,對主線無影響)。final_raw 為 None / 空 → 回退原 context。"""
    mode = p.get("grep_agent_mode", "filter_fetch")
    max_sids = int(p.get("grep_agent_max_sids", 16))
    include_answer_graph = bool(p.get("grep_agent_answer_include_graph_context", True))

    if not (final_raw and corpus.normalize_sids(final_raw)):
        trace["fallback"] = "no_final"
        trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
        return context, trace

    final = corpus.normalize_sids(final_raw)
    if mode == "filter":
        seed_set = set(corpus.normalize_sids(seed))
        final = [s for s in final if s in seed_set]
    elif mode == "fetch_only":
        seed_norm_ = corpus.normalize_sids(seed)
        final = seed_norm_ + [s for s in final if s not in set(seed_norm_)]

    # Provenance gate 已移除(2026-07-22):VECTOR 命中視為 verified,fetch 一律信任。
    trace["verified_sids"] = corpus.normalize_sids(list(verified_sids))
    trace["vector_candidate_sids"] = corpus.normalize_sids(list(vector_candidate_sids))
    seed_set = set(corpus.normalize_sids(seed))
    trace["evidence_provenance"] = {
        s: ("seed+verified" if s in seed_set and s in verified_sids
            else "seed" if s in seed_set
            else "verified" if s in verified_sids
            else "unverified")
        for s in final
    }
    trace["final_before_cap"] = list(final)
    final = final[:max_sids]
    if not final:
        trace["fallback"] = "empty_final"
        return context, trace

    seed_norm = corpus.normalize_sids(seed)

    # Answer-blind 逐條裁決(v1 主線,adjudicate 補回被丟的主題相關 seed)
    adj_on = int(p.get("grep_agent_adjudicate", 0))
    adj_cats = p.get("grep_agent_adjudicate_categories")
    if adj_cats is not None and category not in adj_cats:
        adj_on = 0
    keep_all_cats = p.get("grep_agent_adjudicate_keep_all_categories")
    adjudicated = False
    if adj_on and len(final) < max_sids:
        pending = [s for s in seed_norm if s not in set(final)]
        if pending and keep_all_cats and category in keep_all_cats:
            trace["adjudication"] = {"keep_all": True, "kept": pending, "dropped": []}
            final = (final + [s for s in pending if s not in set(final)])[:max_sids]
            adjudicated = True
        elif pending:
            _t0 = time.perf_counter()
            try:
                kept_adj, verdicts = _adjudicate_candidates(
                    llm, question=question, question_date=question_date,
                    corpus=corpus, pending=pending,
                )
                verdicts["ms"] = round((time.perf_counter() - _t0) * 1000)
                trace["adjudication"] = verdicts
                final = (final + [s for s in kept_adj if s not in set(final)])[:max_sids]
                adjudicated = True
            except Exception as exc:
                trace["adjudication"] = {"error": str(exc)[:200]}

    min_keep = int(p.get("grep_agent_min_keep_aggregation", 0))
    if min_keep and len(final) < min_keep and _AGG_QUESTION_RE.search(question):
        pad = [s for s in seed_norm if s not in set(final)]
        trace["min_keep_padded"] = pad[: min_keep - len(final)]
        final = final + pad[: min_keep - len(final)]

    # ── evidence_floor 盲補已於 2026-07-20 停用(見上一處註解與 config)──────
    # evidence_floor = int(p.get("grep_agent_evidence_floor", 0))
    # if adjudicated:
    #     evidence_floor = 0
    # if evidence_floor > 0 and len(final) < evidence_floor:
    #     final, coverage = _portfolio_pad(
    #         question=question, corpus=corpus, seed_sids=seed_norm,
    #         selected_sids=final, target_size=min(evidence_floor, max_sids),
    #         seed_scores=seed_scores_from_context(context),
    #     )
    #     trace["evidence_coverage"] = coverage
    #     if coverage["added"]:
    #         trace["evidence_floor_padded"] = coverage["added"]

    trace["final_sids"] = final
    trace["kept"] = [s for s in final if s in set(seed_norm)]
    trace["added"] = [s for s in final if s not in set(seed_norm)]
    trace["dropped"] = [s for s in seed_norm if s not in set(final)]

    if not trace["kept"]:
        trace["fallback"] = "zero_keep"
        trace["context_sids"] = seed_norm
        return context, trace

    if mode == "fetch_only":
        if not trace["added"]:
            trace["fallback"] = "no_addition"
            return context, trace
        lines = ["", "### Additional Evidence (agent-retrieved)"]
        ctx_sids = list(seed_norm)
        for s in trace["added"]:
            t = corpus.resolve(s)[0]
            entry = corpus.display_entry(s)
            dt = f"[{t.date}]" if t.date else ""
            lines.append(f"  • {dt}[sid={s}][score=--] {entry} ")
            ctx_sids.append(s)
        trace["context_sids"] = ctx_sids
        return context.rstrip("\n") + "\n".join(lines), trace

    refined, context_sids = _rebuild_context(
        context, corpus, final,
        include_pair=bool(p.get("grep_agent_include_pair", True)),
        include_prefix=include_answer_graph,
    )
    trace["context_sids"] = context_sids
    return refined, trace


_agent_llm_cache = None
_verify_llm_cache = None


def _agent_llm(default_llm):
    """GREP_AGENT_LLM_API/GREP_AGENT_MODEL_NAME 可指到不同 endpoint
    (比照 JUDGE_* 慣例);未設定時共用傳入的答題 LLM。"""
    global _agent_llm_cache
    base = os.getenv("GREP_AGENT_LLM_API")
    name = os.getenv("GREP_AGENT_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _agent_llm_cache is None:
        from grace_mem.llm import LLMClient
        _agent_llm_cache = LLMClient(base_url=base or None, model_name=name or None)
    return _agent_llm_cache


def _verify_llm(default_llm):
    """GREP_AGENT_VERIFY_LLM_API/GREP_AGENT_VERIFY_MODEL_NAME 可把 sufficiency
    verifier 單獨指到不同 endpoint(如 120B),agent loop 與答題不受影響——
    v3-v6 蓋棺主因之一是 oss-20b verifier 誤判率 43%,此鉤子供換強 verifier 重測。"""
    global _verify_llm_cache
    base = os.getenv("GREP_AGENT_VERIFY_LLM_API")
    name = os.getenv("GREP_AGENT_VERIFY_MODEL_NAME")
    if not (base or name):
        return default_llm
    if _verify_llm_cache is None:
        from grace_mem.llm import LLMClient
        _verify_llm_cache = LLMClient(base_url=base or None, model_name=name or None, timeout=300.0)
    return _verify_llm_cache


def maybe_refine_context(
    *,
    question: str,
    context: str,
    csv_path: str | Path | None,
    llm,
    question_date: str | None = None,
    category: str | None = None,
    log_dir=None,
    artifact_dir: str | Path | None = None,
) -> str:
    """qa_eval 流程的統一掛載點(processor 與 rerun 兩條路徑共用)。
    GREP_AGENT_PARAMS.use_grep_agent 關閉時為 no-op;任何失敗都退回原 context。"""
    from experiment.experiment_config import GREP_AGENT_PARAMS

    if not GREP_AGENT_PARAMS.get("use_grep_agent"):
        return context
    if not csv_path or not Path(csv_path).exists():
        print(f"[QA] Grep agent skipped: source csv not found ({csv_path})")
        return context

    print("[QA] Grep agent refining evidence...")
    refined, trace = refine_context(
        question=question,
        context=context,
        csv_path=csv_path,
        llm=_agent_llm(llm),
        question_date=question_date,
        category=category,
        params=GREP_AGENT_PARAMS,
        artifact_dir=artifact_dir,
    )
    if trace.get("fallback"):
        print(f"[QA] Grep agent fallback: {trace['fallback']} (context unchanged)")
    else:
        print(
            f"[QA] Grep agent: kept={len(trace.get('kept', []))} "
            f"added={len(trace.get('added', []))} dropped={len(trace.get('dropped', []))} "
            f"({len(trace.get('commands', []))} tool calls)"
        )
    if log_dir is not None:
        try:
            from grace_mem.utils.error_analysis import append_analysis_record
            append_analysis_record(log_dir, "grep_agent", {"question": question, **trace})
        except Exception as exc:  # 記錄失敗不影響答題
            print(f"[QA] Grep agent trace logging failed: {exc}")
    return refined
