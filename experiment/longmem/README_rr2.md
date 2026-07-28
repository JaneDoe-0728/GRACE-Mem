# LongMemEval rr2 — 新 reranker prompt + category-aware judge

branch `longmem-summary`。在 split-embed / rerank16 檢索的基礎上,換上**新的 reranker prompt**(entity/relationship 各自的 memory-QA 指令)並重跑 7 個檢索設定(rr2),另外把 judge 換成**逐類別 rubric**(對齊 hindsight),並提供 gpt-oss-20b vs gpt-4o-mini 的重判工具。

## 改了什麼

### 1. reranker prompt 重寫 — `KG/utils/reranker.py`
- 為 `entity` / `relationship` 各寫一套 **memory-QA 專屬指令**(先推斷 query 的檢索需求:temporal / preference / multi-hop / knowledge-update,再套對應規則)。
- Qwen3-Reranker chat 格式(system + `<Instruct>/<Query>/<Document>`),以 yes/no logit 差評分。
- 新增 **API 後端**(`RERANKER_API`,OpenAI-compatible + logprobs,可平行、免本地 GPU)與 **local 後端**(CUDA/CPU,含 OOM 自動減半 batch)。
- 重新接上 **`KG_RERANKER_BATCH_SIZE`** env(先前失效,寫死 8);可調小以避 CUDA OOM。

### 2. reranker doc_type 接線 — `KG/pipeline/retrieval_steps/filtering.py`
四個 reranker 呼叫端補上 `doc_type="entity"/"relationship"`。先前呼叫端沒傳 doc_type,relationship 都誤用 entity 指令 → 新寫的 relationship 指令根本沒生效。

### 3. category-aware judge — `experiment/longmem/prompts/judge.py`
逐字對齊 hindsight `judge_answer()`:每個 LongMemEval category(single-session-user/assistant、multi-session、temporal-reasoning、knowledge-update、single-session-preference)用**不同 rubric**;judge LLM 回 JSON `{reasoning, correct}`。

### 4. 實驗工具
- **`rerun_split_experiments.py`** — 依「當時的設定」重跑 7 個 split-embed 檢索實驗(rr2):逐實驗改 `experiment_config.py` 的 5 個檢索旋鈕(surgical edit,保留其餘)、用 watchdog 重跑 retrieval+QA(重用 ingest artifacts)、再用 category-aware judge 判分。
- **`rejudge_output_dirs.py`** — category-aware 重判既有 output CSV,寫到獨立欄(不覆蓋)。支援指定 judge model / base-url(可指向 OpenAI gpt-4o-mini),429/5xx 指數退避,可續跑。
- **`SPLIT_EMBED_RERANK_SUMMARY.md`** — split-embed / rerank16 檢索實驗的完整分析與結論。

## 結果(499 題,含 abstention,gpt-oss-20b judge)

新 reranker prompt 下 7 個實驗:rerank16-rr2 = **78.6%**(全域最佳);direct-vector 系列(extraslot / rerank16)全數受益,單純放大 prov 的 sweep-topk 系列反而變差 → reranker 的價值在「排序更準」,對需要挑乾淨 context 的設定加分。

judge 比較:gpt-4o-mini 判得比 gpt-oss-20b 嚴約 1–6pp(部分完成,受 OpenAI 額度限制)。

> 註:此分支被 `locomo-fine-chunk-rerank16` 分支延伸,把 rerank16 flow 搬到 LoCoMo(細粒度切塊 + 餵 raw context)。
