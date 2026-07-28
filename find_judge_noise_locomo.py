"""
從 LoCoMo 的 _time_rewrite_flips.json 找出「答案語意相同但 judge 判不同」的翻題
（純 judge 雜訊，例如 Evan 那題：baseline 與 TR 都說 week of May 29，卻一對一錯）。

判定「同一答案」：關鍵日期集合相同(且非空) 或 內容詞 Jaccard >= 0.85。
輸出：終端表格 + JSON（judge_noise_regressions / judge_noise_gains）。

用法：
    python find_judge_noise_locomo.py \
        experiment/locomo/output/standard/locomo-n8-20b-92-tr/_time_rewrite_flips.json
"""
import json
import re
import sys
from pathlib import Path

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_STOP = {"the", "a", "an", "on", "in", "of", "to", "and", "during", "week", "i", "you",
         "your", "so", "far", "has", "have", "been", "for", "was", "were"}


def norm(s):
    return " ".join(str(s).split()).strip().lower()


def facts(s):
    """關鍵事實：ISO / 月日年 / 月年 / 年份。"""
    t = norm(s)
    out = set()
    out |= set(re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", t))
    out |= set(re.findall(rf"(?:{MONTHS})\s+\d{{1,2}}(?:,?\s*\d{{4}})?", t))
    out |= set(re.findall(rf"(?:{MONTHS})\s+\d{{4}}", t))
    out |= set(re.findall(r"\b(?:19|20)\d{2}\b", t))
    return out


def ctoks(s):
    return set(re.findall(r"\w+", norm(s))) - _STOP


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else
               "experiment/locomo/output/standard/locomo-n8-20b-92-tr/_time_rewrite_flips.json")
    d = json.loads(src.read_text(encoding="utf-8"))

    noise = {"regressions": [], "gains": []}
    for grp in ("regressions", "gains"):
        for e in d[grp]:
            fa, fb = facts(e["baseline_answer"]), facts(e["tr_answer"])
            ta, tb = ctoks(e["baseline_answer"]), ctoks(e["tr_answer"])
            tj = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
            if (fa == fb and fa) or tj >= 0.85:
                e["_tokjac"] = round(tj, 3)
                e["_facts_same"] = bool(fa == fb and fa)
                noise[grp].append(e)

    for grp, zh in [("regressions", "退步"), ("gains", "進步")]:
        tot = len(d[grp])
        nb = len(noise[grp])
        print(f"### {zh}題(共 {tot})：答案語意相同卻 judge 判不同 = {nb} 題 ({100*nb/tot:.0f}%)")

    out = dict(
        _meta=dict(
            desc="LoCoMo baseline vs TR：答案語意相同(關鍵日期集合相同或內容詞Jaccard>=0.85)但judge判決不同 = 純judge雜訊",
            counts=dict(
                regressions_total=len(d["regressions"]),
                gains_total=len(d["gains"]),
                judge_noise_regressions=len(noise["regressions"]),
                judge_noise_gains=len(noise["gains"]),
            ),
        ),
        judge_noise_regressions=noise["regressions"],
        judge_noise_gains=noise["gains"],
    )
    dst = src.parent / "_judge_noise_flips.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已存 → {dst}")

    print("\n--- 退步題(base對→TR錯) 純 judge 雜訊範例 ---")
    for e in noise["regressions"][:10]:
        print(f"[{e['category']}] {e['question'][:60]}  (facts_same={e['_facts_same']})")
        print(f"   正解: {e['gold_answer'][:55]}")
        print(f"   base(對): {norm(e['baseline_answer'])[:82]}")
        print(f"   TR(錯) : {norm(e['tr_answer'])[:82]}")


if __name__ == "__main__":
    main()
