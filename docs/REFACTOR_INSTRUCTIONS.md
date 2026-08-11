對，現在最值得整理的就是 longmem/ 和 locomo/ 裡面這些直接散在根目錄的 .py 檔案。你現在這個結構不是不能用，但從 open-source repo 的可讀性來看，確實會讓人第一眼不知道：

哪些是 entrypoint
哪些是核心 benchmark flow
哪些只是 helper
哪些是 offline analysis
哪些是 rerun / maintenance script
哪些是 snapshot / artifact lifecycle
哪些只是一次性實驗工具

你目前的畫面大概是：

experiment/
├── common/
├── locomo/
│   ├── analysis/
│   ├── helpers/
│   ├── prompts/
│   ├── stages/
│   ├── utils/
│   ├── aggregate.py
│   ├── cli.py
│   ├── decision.py
│   ├── grep_replay.py
│   ├── models.py
│   ├── pipeline.py
│   ├── run_filter_sweep.py
│   ├── run_one_exp.sh
│   ├── snapshot.py
│   ├── stage_adapter.py
│   ├── summary.py
│   ├── vote_merge.py
│   └── workers.py
│
└── longmem/
    ├── analysis/
    ├── helpers/
    ├── prompts/
    ├── stages/
    ├── utils/
    ├── aggregate.py
    ├── decision.py
    ├── models.py
    ├── processor.py
    ├── rebuild_split_summaries.py
    ├── rerun_split_experiments.py
    ├── rerun.py
    ├── run_batch.py
    ├── snapshot.py
    ├── stage_adapter.py
    └── watchdog.py

我會建議你整理，但不要為了資料夾漂亮就硬搬。重點是讓檔案的責任一眼能看懂。

我會把它整理成 5 類
1. pipeline/

放「正式 benchmark 執行流程」。

例如 LoCoMo：

locomo/
├── pipeline/
│   ├── runner.py
│   ├── worker.py
│   ├── decision.py
│   └── stage_adapter.py

對應你現在：

pipeline.py
workers.py
decision.py
stage_adapter.py

這四個其實都很明顯是：

orchestration / 執行流程

所以放一起非常合理。

LongMem 也可以：

longmem/
├── pipeline/
│   ├── processor.py
│   ├── batch.py
│   ├── watchdog.py
│   ├── decision.py
│   └── stage_adapter.py

對應：

processor.py
run_batch.py
watchdog.py
decision.py
stage_adapter.py

這樣外部使用者一看：

pipeline/

就知道：

benchmark 怎麼跑。

2. artifacts/ 或 runtime/

你現在：

snapshot.py

單獨放在根目錄有點孤單。

如果 snapshot 主要負責：

save graph
restore graph
save VDB
restore artifacts

那它其實屬於：

artifact lifecycle

可以整理成：

locomo/
├── artifacts/
│   ├── snapshot.py
│   ├── restore.py
│   └── paths.py

但如果目前只有 snapshot.py 一個檔案，我反而不會硬建三層。

可以先：

locomo/
├── artifacts/
│   └── snapshot.py

等未來真的有：

artifact_loader.py
artifact_export.py
restore.py

再拆。

3. analysis/

這個你已經有了，我會把所有：

offline
post-run
diagnostic
ablation
sweep
comparison

都盡量歸這裡。

例如 LoCoMo：

run_filter_sweep.py
grep_replay.py
vote_merge.py
summary.py

但要先看實際功能。

可能變：

locomo/
├── analysis/
│   ├── recall_hunter.py
│   ├── filter_sweep.py
│   ├── grep_replay.py
│   ├── vote_merge.py
│   └── summarize_results.py

如果：

grep_replay.py

其實是正式 Agent Filter pipeline 的一部分，那就不要丟 analysis。

判斷原則很簡單：

它會不會參與正式 benchmark execution？

如果：

pipeline.py → grep_replay.py

那它不是 analysis。

如果：

benchmark 已經跑完
↓
讀 artifact
↓
grep_replay.py 分析結果

那就是 analysis。

4. tools/ 或 maintenance/

LongMem 這兩支很典型：

rebuild_split_summaries.py
rerun_split_experiments.py

我不會讓它們一直放在 longmem/ root。

尤其：

rebuild_split_summaries.py

聽名字比較像：

maintenance / artifact migration utility

適合：

longmem/
├── tools/
│   ├── rebuild_split_summaries.py
│   └── rerun_split_experiments.py

或：

longmem/
├── maintenance/

我個人比較推薦 tools/，因為 open-source 使用者比較容易理解。

5. cli.py / entrypoints 留根目錄也可以

有些檔案可以刻意留在最上層。

例如：

locomo/
├── cli.py
└── pipeline/

如果 cli.py 是真正 public CLI entrypoint，那留 root 不一定不好。

甚至你可以讓：

# locomo/cli.py
from .pipeline.runner import main

if __name__ == "__main__":
    main()

這樣 root 很乾淨。

我會推薦的 LoCoMo 結構

我會整理成：

experiment/
└── locomo/
    ├── __init__.py
    │
    ├── cli.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── runner.py
    │   ├── worker.py
    │   ├── decision.py
    │   └── stage_adapter.py
    │
    ├── stages/
    │   ├── ingest.py
    │   ├── qa_eval.py
    │   └── judge.py
    │
    ├── artifacts/
    │   └── snapshot.py
    │
    ├── analysis/
    │   ├── recall_hunter.py
    │   ├── filter_sweep.py
    │   ├── vote_merge.py
    │   └── result_summary.py
    │
    ├── prompts/
    ├── helpers/
    ├── utils/
    └── models.py

注意我保留：

models.py

在 root。

因為：

RunConfig
RunRuntime
SamplePlan

這種 benchmark-level model 很多 module 都會使用。

grep_replay.py 我會先不要決定

它名字比較 ambiguous。

可能是：

Agent Filter 真正的 replay pipeline

也可能只是：

offline analysis

所以我會先 trace caller。

如果：

pipeline
↓
worker
↓
grep_replay

它應該放：

pipeline/replay.py

或：

retrieval_modes/grep_replay.py

如果是：

existing run
↓
grep_replay
↓
new statistics

才放：

analysis/grep_replay.py
LongMem 我會整理成這樣
experiment/
└── longmem/
    ├── __init__.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── processor.py
    │   ├── batch.py
    │   ├── watchdog.py
    │   ├── decision.py
    │   └── stage_adapter.py
    │
    ├── stages/
    │   ├── ingest.py
    │   ├── qa_eval.py
    │   ├── judge.py
    │   └── upload.py
    │
    ├── artifacts/
    │   └── snapshot.py
    │
    ├── analysis/
    │   ├── recall_hunter.py
    │   └── ...
    │
    ├── tools/
    │   ├── rebuild_split_summaries.py
    │   └── rerun_split_experiments.py
    │
    ├── rerun/
    │   └── runner.py
    │
    ├── prompts/
    ├── helpers/
    ├── utils/
    └── models.py
不過 rerun.py 要不要獨立資料夾？

要看它有多大。

如果只有：

rerun.py

一支，而且 200 行：

longmem/rerun.py

其實完全沒問題。

不要為了一個檔案硬建：

rerun/
    runner.py

這反而會造成：

folder explosion。

所以你可以先整理成：

longmem/
├── pipeline/
├── stages/
├── analysis/
├── tools/
├── helpers/
├── utils/
├── prompts/
├── models.py
├── rerun.py
└── snapshot.py

這已經乾淨非常多。

你現在另外一個問題是 helpers/ + utils/

你同時有：

helpers/
utils/

這很容易開始混亂。

因為很多工程師最後自己都不知道：

這支 function 到底應該放 helpers 還是 utils？

例如：

load_json()

放 utils？

resolve_dataset()

放 helpers？

我會建議未來慢慢讓名稱更 domain-specific：

helpers/
    datasets.py
    checkpoints.py
    progress.py

這其實還 OK。

而：

utils/
    io.py
    log.py

這也還算合理。

所以目前不用急著合併。

但如果你看到：

helpers/io.py
utils/io.py

就要開始處理。

common/ 又應該放什麼？

你現在已經有：

experiment/common/

這是好事。

但不要把：

LongMem + LoCoMo 看起來類似

就全部丟進 common。

common/ 只放真正的：

benchmark infrastructure。

例如：

experiment/common/
├── metadata.py
├── reproducibility.py
├── subprocess.py
├── artifacts.py
└── models.py

比較合理。

例如：

LongMem write_run_metadata()
LoCoMo write_run_metadata()

如果真的一樣：

common/run_metadata.py

但：

LongMem GoldEvidence
LoCoMo GoldEvidence

即使名字一樣，也可能不要合。

因為 dataset semantics 不一樣。

我認為你目前最理想的層次是這樣
experiment/
│
├── common/
│   └── benchmark infrastructure
│
├── locomo/
│   ├── pipeline/
│   ├── stages/
│   ├── analysis/
│   ├── tools/
│   ├── artifacts/
│   ├── helpers/
│   └── prompts/
│
└── longmem/
    ├── pipeline/
    ├── stages/
    ├── analysis/
    ├── tools/
    ├── artifacts/
    ├── helpers/
    └── prompts/

整體概念：

                     experiment
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
      common           locomo          longmem
                         │               │
             ┌───────────┼──────┐        │
             ↓           ↓      ↓        ↓
          pipeline     stages analysis  pipeline
                                   ...