"""一次算一個 run 的全部指標:accuracy / F1 / BLEU-1 + agent 定性(fallback% / kept / added / dropped 每題)。
支援 LongMem 與 LoCoMo。

用法:
    python experiment/longmem/agent_filter/score_all.py longmem adjn3-lm20b-r1
    python experiment/longmem/agent_filter/score_all.py locomo  adjn3-lc120b-r1

輸出一行:run acc=.. F1=.. BLEU=.. | fb%=.. kept=.. added=.. dropped=.. (n=..)
注意 F1/BLEU 是 token 重疊,對答案長度敏感,與 judge accuracy(語意)並列讀。
"""
import sys, glob, csv, json
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
csv.field_size_limit(10**8)
from experiment.locomo.stages.judge import compute_f1_and_bleu1


def _agent_stats(trace_glob):
    """從 _grep(_agent)_traces.jsonl 算 fallback% / kept / added / dropped 每題。
    以 (sample) 去重(--force 重跑會 append),取每 sample 最後一筆。"""
    latest = {}
    for f in glob.glob(trace_glob):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (f, d.get("sample") or d.get("question", "")[:60])
            latest[key] = d
    rows = list(latest.values())
    n = len(rows)
    if not n:
        return None
    fb = sum(1 for r in rows if r.get("fallback"))
    def _n(v):  # LongMem 存 list、LoCoMo 存 int,都轉數量
        return len(v) if isinstance(v, (list, tuple, set)) else (int(v) if isinstance(v, (int, float)) else 0)
    def avg(field):
        vals = [_n(r[field]) for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return dict(n=n, fbr=100 * fb / n, kept=avg("kept"), added=avg("added"), dropped=avg("dropped"))


def longmem(run):
    fs = [x for x in glob.glob(f"experiment/longmem/output/{run}/*/*.csv") if "_grep" not in x]
    f1s, bls, accs = [], [], []
    for f in fs:
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        f1, bl = compute_f1_and_bleu1(str(r[0].get("answer", "")), str(r[0].get("Generated_Answer", "")))
        f1s.append(f1); bls.append(bl)
        try: accs.append(int(float(str(r[0].get("correctness_new", "")).strip())))
        except: pass
    ag = _agent_stats(f"experiment/longmem/output/{run}/*/_grep_agent_traces.jsonl")
    return f1s, bls, accs, ag


def locomo(run):
    fs = [x for x in glob.glob(f"experiment/locomo/output/standard/{run}/sample_*/*_eval_*.csv") if "judge" not in x]
    jf = {}
    for f in glob.glob(f"experiment/locomo/output/standard/{run}/sample_*/*_judge_4omini.csv"):
        for r in csv.DictReader(open(f)):
            try: jf[str(r.get("question", "")).strip()] = int(float(str(r.get("correctness_4omini", "")).strip()))
            except: pass
    f1s, bls, accs = [], [], []
    for f in fs:
        for r in csv.DictReader(open(f)):
            g = str(r.get("gold_answer", ""))
            if not g.strip():
                continue
            f1, bl = compute_f1_and_bleu1(g, str(r.get("model_answer", "")))
            f1s.append(f1); bls.append(bl)
            q = str(r.get("question", "")).strip()
            if q in jf: accs.append(jf[q])
    ag = _agent_stats(f"experiment/locomo/output/standard/{run}/sample_*/_grep_traces.jsonl")
    return f1s, bls, accs, ag


def main():
    bench, run = sys.argv[1], sys.argv[2]
    f1s, bls, accs, ag = (longmem if bench == "longmem" else locomo)(run)
    n = len(f1s)
    acc = f"{100*sum(accs)/len(accs):.1f}%" if accs else "n/a"
    line = f"{run}: acc={acc} F1={100*sum(f1s)/n:.2f}% BLEU={100*sum(bls)/n:.2f}% (n={n})"
    if ag:
        line += f" | fb%={ag['fbr']:.1f} kept={ag['kept']:.1f} added={ag['added']:.2f} dropped={ag['dropped']:.1f}"
    print(line)


if __name__ == "__main__":
    main()
