"""Dated Fact Ledger:答題前把 evidence 編譯成日期事實表(編譯≠判斷)。

範式:與 grep agent 的「選擇」正交的第二機制——「表示變換」。不推 recall
(respects the Precision Wall),把 temporal 算術/最新值判斷從答題時的 LLM
推理搬到編譯後的表上讀。

實測(temporal+KU 子集,兩輪複製):
  temporal:grep→ledger 疊加 81.5/82.3 vs v2 79.2(4/4 全勝)
  KU:     ledger 單獨 74.4/74.4 vs v2 73.1(疊加會把舊值 mentions 砍掉,不可疊)
"""
from __future__ import annotations

COMPILE_SYSTEM = """You are a fact compiler. From the conversation evidence below, extract every
dated fact as a table row: | date | who/what | fact |
- One row per dated mention. Use the [YYYY/MM/DD] stamps shown on each excerpt.
- Include EVERY dated mention of the same entity/fact (do not merge updates —
  each update is its own row).
- Only extract what is literally in the evidence. Output ONLY the table."""

LEDGER_HEADER = "\n\n### Dated Fact Table (compiled from the evidence above)\n"


def compile_table(llm, evidence: str, *, max_chars: int = 24000, max_tokens: int = 1500) -> str:
    """把 evidence block 編譯成 dated fact table(一次 LLM call)。失敗回空字串。"""
    try:
        resp = llm.chat(messages=[
            {"role": "system", "content": COMPILE_SYSTEM},
            {"role": "user", "content": evidence[:max_chars]},
        ], temperature=0.0, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def append_ledger(context: str, table: str) -> str:
    if not table:
        return context
    return context + LEDGER_HEADER + table
