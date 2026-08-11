# GRACE-Mem Refactor 變更報告

## 1. 報告範圍

本報告記錄 branch `fix/readme-accuracy-locomo-chunking` 上，從 refactor 開始前的
commit `ea5d063` 到 `4a27d01` 為止的變更。

此範圍包含：

- package/import 邊界整理；
- runtime resource lifecycle；
- LongMem/LoCoMo orchestration 重構；
- 測試收集與 architecture gates；
- 根 README 與 experiment guide 重寫。

此範圍**不包含**後續規劃中的開源發布清理，例如刪除分析紀錄、NocoDB 或開發文件；
那些項目目前只完成盤點，尚未修改 working tree。

## 2. 整體統計

| 項目 | 數量 |
|---|---:|
| 階段 commits | 16 |
| 異動路徑 | 102 |
| 新增檔案 | 22 |
| 修改檔案 | 77 |
| 刪除檔案 | 3 |
| 新增行數 | 3,336 |
| 刪除行數 | 2,229 |

目前驗證基準：

- 182 個 `KG`/`experiment` Python modules；
- 374 條 package-local import edges；
- 73 條 `experiment -> KG` edges；
- 0 條 `KG -> experiment` reverse dependency；
- 0 個 static circular dependency；
- `298 passed, 6 skipped, 3 xfailed`。

## 3. 階段 commits

| Commit | 階段 |
|---|---|
| `ed3a432` | 建立 package architecture boundaries |
| `9aec452` | 將 pipeline runtime 改成明確 lifecycle owner |
| `2b871b2` | 內部模組不再繞行 helper facade |
| `ffcdf3c` | helper compatibility facade 改成 lazy import |
| `8c0b4a1` | 集中 adaptive retrieval trace 建構 |
| `f52f853` | 更新 dependency baseline 文件 |
| `53031d6` | 集中 `VDBManager` cleanup |
| `60cc32d` | 關閉 LLM transport lifecycle |
| `718378f` | `MultiDatasetProcessor` 擁有並釋放 LongMem resources |
| `556bef0` | LongMem dataset configuration 型別化 |
| `9c0e9e0` | 限制 dataset logger monkeypatch scope |
| `2926412` | `LongMemRerun` 擁有 rerun/watchdog resources |
| `458e7ba` | 隔離 standalone script import bootstraps |
| `3e1719e` | 明確化 pytest collection policy |
| `f282ee6` | 重建根 README 資訊架構 |
| `4a27d01` | 重建 experiment operations guide |

## 4. 改前與改後

### 4.1 Import 與 package 邊界

**改前**

- `KG` 與 `experiment` 之間存在不清楚的責任邊界。
- 部分 LoCoMo 模組透過 `helpers.__init__` 取得實際 owner 的功能。
- helper facade eager import optional layers，package import 可能觸發額外初始化。
- 多個 standalone scripts 在被 package import 時直接修改全域 `sys.path`。
- token tracking 與 LLM/entity operation code 形成反向依賴。

**改後**

- `KG` 保持 benchmark-neutral，禁止 import `experiment`。
- 內部模組直接 import 實際 owner；helper facade 只保留 lazy compatibility API。
- 新增 AST import graph tool 與 CI-style architecture tests。
- direct-file CLI 仍可執行，但 package import 不再污染 `sys.path`。
- token tracking 抽離至獨立模組。

**改善**

- static import graph 已無 circular dependency。
- package import side effects 降低，測試與外部整合更可預測。
- owner 與 compatibility layer 的責任可被明確辨識。

### 4.2 Runtime resource lifecycle

**改前**

- `build_pipeline()` 回傳一般 dict，graph/LLM 沒有統一 close contract。
- `VDBManager` cleanup 分散在 caller，部分程式直接操作 reset/private state。
- LongMem processor、rerun、watchdog 建立共用 graph/LLM 後沒有一致釋放流程。
- startup 中途失敗時，已開啟的 transport 可能遺留。

**改後**

- `PipelineRuntime` 同時提供 attribute 與 mapping compatibility，並支援 context manager。
- `LLMClient`、`VDBManager`、`MultiDatasetProcessor`、`LongMemRerun` 都有 idempotent
  `close()`。
- constructor failure 會 rollback 已開啟的 graph/LLM。
- dataset-local VDB clients 在每題 teardown 時關閉，共用 transport 在 owner scope 結束時關閉。

**改善**

- 降低長時間 benchmark run 的連線、thread、Chroma client 與記憶體洩漏風險。
- cleanup failure 不會阻止後續資源繼續釋放。
- lifecycle 行為可用 unit tests 驗證。

### 4.3 LongMem 與 LoCoMo orchestration

**改前**

- LongMem dataset configuration 混用 dict/namespace，欄位 contract 不明確。
- dataset logger 透過 module monkeypatch 設定，但 teardown 後不一定還原。
- LoCoMo aggregation/summary 邏輯散落於 aggregate 與 upload layers。
- adaptive retrieval trace 在多條路徑重複建構。

**改後**

- LongMem 使用 immutable typed `DatasetConfig`。
- logger binding 會記錄原值並依反向順序還原。
- LoCoMo summary calculation 集中至 `experiment/locomo/summary.py`。
- adaptive pass trace 由 Retriever 內部單一 helper 建構。
- reproducibility runtime 下沉到 `KG/runtime`，experiment 只保留 adapter。

**改善**

- dataset 設定錯誤更早被發現。
- dataset 之間不會互相污染 logger/runtime state。
- aggregation 與 adaptive behavior 減少重複邏輯及結果漂移。

### 4.4 Test collection 與品質 gates

**改前**

- live API/model scripts 使用 `test_*.py` 命名，pytest collection 意圖不清楚。
- 一個指向不存在 `experiment.analysis` package 的測試被靜默 ignore。
- architecture、resource lifecycle 與 import side effects 缺少固定 gates。

**改後**

- 9 個 manual probes 透過 `MANUAL_SCRIPT_NAMES` 明確排除。
- 移除無 production implementation 的 orphan benchmark-analysis contract。
- pytest 開啟 strict config/marker。
- 新增 architecture、runtime lifecycle、typed config、token tracking、collection policy
  與 documentation contracts。

**改善**

- 預設 suite 保持 deterministic/offline，同時不再用 ignore 隱藏缺少的 production code。
- regression 可直接阻止 cycle、reverse dependency、`sys.path` pollution 與 resource leak 回歸。

### 4.5 README 與操作文件

**改前**

- 根 `readme.md` 共 618 行，setup、benchmark、internal API reference 與 FalkorDB
  操作混在同一層。
- `experiment/readme.md` 共 226 行，資料 contract 不完整，資訊以 Part A/Part B
  長篇方式排列。
- `pyproject.toml` 指向 `README.md`，但 tracked filename 是小寫 `readme.md`。

**改後**

- 根檔名標準化為 `README.md`，275 行，聚焦定位、架構、Quick Start、核心 API、
  benchmark 入口與文件索引。
- experiment 文件標準化為 `experiment/README.md`，268 行，依 execution model、data
  layout、LongMem、LoCoMo、shared config、artifact compatibility 與 recovery 排列。
- documentation tests 會實際驗證 local links、anchors、CLI flags 與 default stage model。

**改善**

- 新使用者能先完成最短可執行流程，再進入 benchmark 細節。
- 實作細節與操作文件分層，降低 README 過期範圍。
- Linux case-sensitive checkout 與 packaging metadata 一致。

## 5. 新增檔案

### Core/runtime

- `KG/llm/token_tracking.py`：獨立 token tracker 與 context ownership。
- `KG/runtime/__init__.py`：runtime package public surface。
- `KG/runtime/reproducibility.py`：benchmark-neutral reproducibility settings/runtime。
- `experiment/__init__.py`：將 experiment 明確定義為 package。
- `experiment/locomo/summary.py`：集中 aggregation summary calculation。
- `experiment/noco/client_loader.py`：不修改 `sys.path` 的 bundled Noco client loader。

### Documentation/tooling

- `README.md`：新的 root project overview。
- `experiment/README.md`：新的 experiment operations guide。
- `docs/architecture/import-graph.md`：dependency baseline 與 removed cycles。
- `tools/__init__.py`：tool package marker。
- `tools/import_graph.py`：AST-based internal import graph/cycle checker。

### Tests

- `test/README.md`
- `test/conftest.py`
- `test/test_architecture.py`
- `test/test_collection_policy.py`
- `test/test_llm_client_lifecycle.py`
- `test/test_longmem_config.py`
- `test/test_longmem_processor_lifecycle.py`
- `test/test_longmem_rerun_lifecycle.py`
- `test/test_pipeline_runtime.py`
- `test/test_reproducibility_runtime.py`
- `test/test_token_tracking.py`

## 6. 刪除檔案

- `readme.md`：由新的 case-correct `README.md` 取代，並非移除使用者文件。
- `experiment/readme.md`：由新的 `experiment/README.md` 取代。
- `test/test_benchmark_analysis_imports.py`：測試指向從未存在於 tracked repository 的
  `experiment.analysis` package，而且原先被 pytest ignore；移除失效 contract，沒有刪除
  可執行 production behavior。

## 7. 修改檔案

以下列出基準 commit 至 `4a27d01` 的所有 modified paths。

### KG core（7）

- `KG/llm/__init__.py`
- `KG/llm/client.py`
- `KG/pipeline/factory.py`
- `KG/pipeline/ingestor.py`
- `KG/pipeline/retriever.py`
- `KG/services/entity_manager.py`
- `KG/storage/chroma_manager.py`

### Documentation/configuration（4）

- `docs/CHANGES-2026-08-03.md`
- `pyproject.toml`
- `setup_env.sh`
- `experiment/reproducibility.py`

### LoCoMo（27）

- `experiment/locomo/aggregate.py`
- `experiment/locomo/cli.py`
- `experiment/locomo/decision.py`
- `experiment/locomo/grep_replay.py`
- `experiment/locomo/helpers/__init__.py`
- `experiment/locomo/helpers/analyze.py`
- `experiment/locomo/helpers/dataset.py`
- `experiment/locomo/helpers/llm.py`
- `experiment/locomo/helpers/normalize_golden_answers.py`
- `experiment/locomo/helpers/run_hooks.py`
- `experiment/locomo/helpers/sample_hooks.py`
- `experiment/locomo/helpers/session_export.py`
- `experiment/locomo/judge_eval_4omini.py`
- `experiment/locomo/pipeline.py`
- `experiment/locomo/prompts/__init__.py`
- `experiment/locomo/recall_hunter_locomo.py`
- `experiment/locomo/rejudge_3vote_4omini.py`
- `experiment/locomo/rejudge_4omini.py`
- `experiment/locomo/snapshot.py`
- `experiment/locomo/stages/ingest.py`
- `experiment/locomo/stages/judge.py`
- `experiment/locomo/stages/qa_eval.py`
- `experiment/locomo/stages/upload.py`
- `experiment/locomo/turn_filter_analysis.py`
- `experiment/locomo/utils/graph.py`
- `experiment/locomo/vote_merge.py`
- `experiment/locomo/workers.py`

### LongMem/Noco（30）

- `experiment/agent_filter/grep_reachability.py`
- `experiment/agent_filter/oracle_eval.py`
- `experiment/agent_filter/replay_run.py`
- `experiment/agent_filter/resample_replay.py`
- `experiment/agent_filter/score_all.py`
- `experiment/agent_filter/score_f1_bleu.py`
- `experiment/agent_filter/smoke_test.py`
- `experiment/agent_filter/tribunal.py`
- `experiment/longmem/analysis/collect.py`
- `experiment/longmem/analysis/summarize.py`
- `experiment/longmem/gold_recall_metrics.py`
- `experiment/longmem/gold_summary_eval.py`
- `experiment/longmem/gold_upstream_score_dist.py`
- `experiment/longmem/models.py`
- `experiment/longmem/processor.py`
- `experiment/longmem/rebuild_split_summaries.py`
- `experiment/longmem/rejudge_multi_dataset.py`
- `experiment/longmem/rejudge_output_dirs.py`
- `experiment/longmem/replay_fact_multi_dataset.py`
- `experiment/longmem/replay_fact_user_only.py`
- `experiment/longmem/rerun.py`
- `experiment/longmem/rerun_split_experiments.py`
- `experiment/longmem/run_batch.py`
- `experiment/longmem/stages/qa_eval.py`
- `experiment/longmem/summary_score_dist.py`
- `experiment/longmem/utils/__init__.py`
- `experiment/longmem/utils/io.py`
- `experiment/longmem/watchdog.py`
- `experiment/noco/noco_progress.py`
- `experiment/noco/upload_progress_to_noco.py`

完整 machine-readable 清單可用
`git diff --name-status ea5d063..4a27d01` 重新產生。

### Tests（8）

- `test/test_adaptive_trace.py`
- `test/test_chroma_manager_paths.py`
- `test/test_chunk_and_split_config.py`
- `test/test_gemini.py`
- `test/test_gemini_new.py`
- `test/test_gpt.py`
- `test/test_gpt_new.py`
- `test/test_readme_claims.py`

## 8. 行為相容性與刻意變更

- `PipelineRuntime` 保留 mapping access，因此舊的 `runtime["retriever"]` caller 仍可使用。
- direct-file CLI 仍支援 `python path/to/script.py`，只有 package import 不再修改 `sys.path`。
- `VDBManager.close()` 預設不刪 artifacts；刪除資料仍需明確呼叫 reset/delete 行為。
- LongMem split-summary 與 retrieval mode 仍由同一 config flag 驅動。
- README filename 大小寫改名需要外部連結改用 `README.md` 與
  `experiment/README.md`。
- manual live-service scripts 仍保留，只是不進入預設 pytest collection。

## 9. 最終改善摘要

這次 refactor 沒有重寫 KG 演算法；主要目標是讓既有實驗行為具備可維護、可測試、
可釋放資源與可對外說明的結構。最主要成果是：

1. dependency direction 可被工具與測試驗證；
2. graph、LLM、VDB lifecycle 有明確 owner；
3. benchmark dataset 之間的 mutable state 不再任意外洩；
4. LongMem/LoCoMo 共用設定與核心 runtime contract 更清楚；
5. 預設測試 suite 對 manual/integration/expected-failure 有明確分類；
6. README 從 implementation dump 改成可執行的開源專案入口。
