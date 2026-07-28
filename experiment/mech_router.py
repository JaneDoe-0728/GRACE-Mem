"""Unified mechanism router — LoCoMo 與 LongMem 共用的問題形狀路由(單一來源)。

用戶定版(2026-07-08):檢索單位可隨資料形態不同,但 agent 模式與機制棧
兩邊必須固定同一套。此檔取代兩邊各自的 route() 副本:

  shape            機制                      出處/驗證
  when             when-anchor mention 表    LoCoMo 過閘(統一棧一層)
  duration         duration/temporal ledger  LoCoMo durledger 過閘;LongMem temporal_stack 過閘
  latest           dated-fact ledger(不砍)   LongMem ledger_alone 過閘
  enum / count     enum 表(+sweep+desum)     LongMem 過閘(+3.8pp);LoCoMo 中性納入
  (答案層觸發) abstain → corpus sweep 重答   兩 benchmark 統一基底皆中性,一致性納入

優先序:when > duration > latest > enum/count。
"""
from __future__ import annotations

import re

_WHEN = re.compile(r"^\s*when (did|was|were|do|does|has|have)\b", re.I)
_DATED_WHAT = re.compile(
    r"^(what|which|who|how)\b.*\b(in|on|during)\s+"
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|\d{4}|the (spring|summer|fall|winter))", re.I)

_DURATION = re.compile(
    r"\bhow long\b|\bhow many (days|weeks|months|years)\b|\bafter how many\b|"
    r"\bhow much time\b|\bdays? (before|after|between)\b|\bduration\b|\bsince when\b", re.I)

_LATEST = re.compile(
    r"\b(most recent(ly)?|latest|current(ly)?|now|these days|still|"
    r"nowadays|as of|right now|today)\b|\b(did i .*(change|update|switch))\b", re.I)

_STOP_S = {
    "was", "is", "has", "does", "his", "its", "this", "as", "says", "apos", "s",
    "wants", "needs", "feels", "gets", "goes", "loves", "likes", "enjoys",
    "plans", "yes", "besides", "perhaps", "always", "sometimes", "across", "us",
}
_IRREGULAR = re.compile(r"\b(people|children|men|women)\b", re.I)
_LIST_TAIL = re.compile(r"\ballergic to\b|\ballergies\b", re.I)
_PLURAL = re.compile(r"\b([a-z]{3,}s)\b", re.I)
_WH = re.compile(r"^(what|which|who)\b", re.I)
_HOWMANY = re.compile(r"\bhow many\b|\bhow much\b|\b(in )?total\b", re.I)

ABSTAIN = re.compile(
    r"not (mention|specif|provid|enough|indicat)|no (record|information|indication)|"
    r"don.t have|does not (specify|list|provide|mention)|isn.t (specified|provided)|"
    r"cannot be determined|doesn.t (name|specify|mention|indicate)", re.I)


def route(question: str) -> str | None:
    q = (question or "").strip()
    if _WHEN.match(q) or _DATED_WHAT.match(q):
        return "when"
    if _DURATION.search(q):
        return "duration"
    if _LATEST.search(q):
        return "latest"
    if _HOWMANY.search(q):
        return "count"
    if _WH.match(q):
        if _IRREGULAR.search(q) or _LIST_TAIL.search(q):
            return "enum"
        for m in _PLURAL.finditer(q):
            w = m.group(1).lower()
            if w not in _STOP_S and not w.endswith("ss"):
                return "enum"
    return None


def is_abstention(answer: str, head_chars: int = 250) -> bool:
    return bool(ABSTAIN.search(str(answer)[:head_chars]))
