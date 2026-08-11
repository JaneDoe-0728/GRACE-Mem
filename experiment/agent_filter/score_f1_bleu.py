"""算既有 run 的 F1 / BLEU-1（官方 compute_f1_and_bleu1），支援 LongMem 與 LoCoMo。

用法:
    python experiment/agent_filter/score_f1_bleu.py longmem adjn3-lm20b-r1
    python experiment/agent_filter/score_f1_bleu.py locomo adjn3-lc120b-r1

LongMem run: experiment/longmem/output/<run>/*/*.csv  欄 answer / Generated_Answer
LoCoMo  run: experiment/locomo/output/standard/<run>/sample_*/*_eval_*.csv  欄 gold_answer / model_answer

注意:F1/BLEU 是 token 重疊指標,對答案長度敏感（答得囉嗦 → precision 稀釋 → F1 低）。
與 4o-mini judge accuracy（語意正確率）並列讀,不可單看。
"""
import sys, glob, csv
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
csv.field_size_limit(10**8)
from experiment.locomo.stages.judge import compute_f1_and_bleu1


def score_longmem(run: str):
    fs = [x for x in glob.glob(f"experiment/longmem/output/{run}/*/*.csv") if "_grep" not in x]
    f1s, bls = [], []
    for f in fs:
        r = list(csv.DictReader(open(f)))
        if not r:
            continue
        f1, bl = compute_f1_and_bleu1(str(r[0].get("answer", "")), str(r[0].get("Generated_Answer", "")))
        f1s.append(f1); bls.append(bl)
    return f1s, bls


def score_locomo(run: str):
    fs = [x for x in glob.glob(f"experiment/locomo/output/standard/{run}/sample_*/*_eval_*.csv") if "judge" not in x]
    f1s, bls = [], []
    for f in fs:
        for r in csv.DictReader(open(f)):
            g = str(r.get("gold_answer", ""))
            if not g.strip():
                continue
            f1, bl = compute_f1_and_bleu1(g, str(r.get("model_answer", "")))
            f1s.append(f1); bls.append(bl)
    return f1s, bls


def main():
    bench, run = sys.argv[1], sys.argv[2]
    f1s, bls = (score_longmem if bench == "longmem" else score_locomo)(run)
    n = len(f1s)
    print(f"{run} ({bench}): n={n}  F1={100*sum(f1s)/n:.2f}%  BLEU-1={100*sum(bls)/n:.2f}%")


if __name__ == "__main__":
    main()
