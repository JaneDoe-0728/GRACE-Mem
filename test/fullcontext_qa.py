# answer_all_sessions_as_one.py
import json
import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Optional
import pandas as pd

# 讓相對匯入可用
sys.path.append(str(Path(__file__).resolve().parent.parent))
from KG.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. Prefer facts in the provided context. "
    "If the context is insufficient, you may use general knowledge, but keep the answer precise."
)

REQUIRED_COLS = {
    "matched_answer_session_id", "turn_index", "role", "content", "haystack_date", "question"
}

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位: {sorted(missing)}")
    df = df.sort_values(["matched_answer_session_id", "turn_index"], kind="mergesort").copy()
    df["session_id"] = df["matched_answer_session_id"].astype(str)
    df["turn_index"] = pd.to_numeric(df["turn_index"], errors="coerce").fillna(0).astype(int)
    df["role"] = df["role"].astype(str).str.lower().str.strip()
    df["content"] = df["content"].astype(str)
    df["haystack_date"] = df["haystack_date"].astype(str).str.strip()
    df["question"] = df["question"].astype(str).str.strip()
    return df

def first_non_empty_question(df: pd.DataFrame) -> str:
    for q in df["question"].tolist():
        if q:
            return q
    raise ValueError("question 欄位全為空")

def build_session_dialogue(g: pd.DataFrame) -> Tuple[str, str]:
    t = g["haystack_date"].iloc[0]
    lines: List[str] = [f"{r['role']}: {r['content'].strip()}" for _, r in g.iterrows()]
    return t, "\n".join(lines)

def stitch_global_context(df: pd.DataFrame, *, max_sessions: Optional[int] = None) -> str:
    parts: List[str] = []
    for i, (sid, g) in enumerate(df.groupby("session_id", sort=False), start=1):
        if max_sessions is not None and i > max_sessions:
            break
        t, d = build_session_dialogue(g)
        parts.append(f"[Session {sid} | {t}]\n{d}")
    return "\n\n".join(parts)

def maybe_truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... [truncated] ...\n\n" + text[-(max_chars - half):]

def build_messages(question: str, global_context: str) -> list:
    stitched = f"---All Sessions Dialogue---\n{global_context}\n------------------"
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{stitched}"},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
    ]

def ask_llm(llm: LLMClient, messages: list, temperature: float, max_tokens: int) -> str:
    resp = llm.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=None, help="只取前 N 個 session 串 context（除錯用）")
    ap.add_argument("--max-chars", type=int, default=None, help="context 上限字元（過長會中段截斷）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    CSV_PATH = "data/answer_sessions_gpt4_d84a3211.csv"   
    
    df = load_dataframe(CSV_PATH)
    question = first_non_empty_question(df)

    global_ctx = stitch_global_context(df, max_sessions=args.max_sessions)
    global_ctx = maybe_truncate(global_ctx, args.max_chars)

    llm = LLMClient()
    messages = build_messages(question, global_ctx)
    print("=== MESSAGES TO LLM ===\n", json.dumps(messages, ensure_ascii=False, indent=2))
    answer = ask_llm(llm, messages, args.temperature, args.max_tokens)

    print(json.dumps({
        "question": question,
        "use_kg": False,
        "max_sessions": args.max_sessions,
        "max_chars": args.max_chars,
        "answer": answer
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
