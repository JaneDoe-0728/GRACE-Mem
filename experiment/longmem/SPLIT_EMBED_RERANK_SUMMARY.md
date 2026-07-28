# Split-embed 檢索改進實驗總結

## 0. 背景與目標
- **對象**:LongMemEval,split-embed 模式(檢索時把每個 turn-pair summary 展開成 `:u`(user raw)/`:a`(assistant 壓縮)兩個候選,各自打分競爭 top-K)。
- **題庫**:真實 500 題 = 470 base + 30 abstention(`_abs`)。baseline run(`split-embed`)完成 499,缺 1 題(`temporal_reasoning/gpt4_7ddcf75f`,watchdog 重啟時工作清單漏掉,非資料/執行錯誤)。
- **目標**:提高 gold summary 返回率 → 進而提高答題正確率。

## 1. 指標定義
- **gold sid**:來源 CSV 中 `has_answer=True` 的 turn,對到 split sid `{session}:{msgid}:{u|a}`(user turn msgid=turn+1、assistant msgid=turn)。`has_answer` 是 LongMemEval 原始資料標的(`answer_*` 證據 session);knowledge_update 是整個 session 全標(~24/題),其餘類別只標證據 turn(2–5/題)。
- **返回率(recall)** = 撈到的 gold sid / 全部 gold sid(micro)。
- **全中率** = gold 全部被撈到的題 / 有 gold 的題。
- **全中正確率** = 全中題目中答對的比例(context 稀釋的敏感指標)。
- **precision** = 返回的 summary 中是 gold 的比例(context 乾淨度)。
- **正確率** = judge 判對 / 全部題(含 abstention,499 口徑)。

## 2. 診斷:漏掉的 gold 長什麼樣
- baseline 返回率 ~52%(side-level)。gold 對 question 的 cosine **中位名次 = 16**(剛好卡在 top-16),但分佈兩極:p25≈5(一批很前)、p75≈88、p90≈241(一批埋很深)。
- **top-16 涵蓋 ~50% gold、top-40 涵蓋 ~65%,35% 的 gold 排在 40 名外**(cosine 搆不到 → embedding 品質極限,調檢索救不回)。
- 高分漏網(cosine≥0.5 卻沒撈到)絕大多數是**沒進候選池**(entity/relationship 擴散沒連到),不是被 topk 截掉 → 動機:加「summary 直接向量檢索」。
- **user 側 gold 返回率(62%)遠高於 assistant 側(38%)**:assistant `:a` 存 LLMlingua 壓縮亂碼,embedding/reranker 都對它不準(assistant 是後續所有方法唯一持續退步的類別)。

## 3. 試過的方法與整體結果(499 題,含 abstention)

| 方法 | 正確率 | 返回率 | 全中正確率 | avg summary |
|---|---|---|---|---|
| base(prov top-16) | 69.3% | 51% | 85% | 15.9 |
| topk24(prov 放大) | 70.9% | 57% | 81% | 24 |
| topk32(prov 放大) | 71.9% | 62% | 78% | 32 |
| extra0.5(prov16 + direct≥.5) | 70.1% | 54% | 82% | 17.7 |
| extra0.4 | 70.7% | 61% | 79% | 31.9 |
| extra0.35 | 69.7% | 65% | 77% | 43.1 |
| **rerank16(wide≥.35 → 重排 → 留16)** | **73.5%** | 50% | 82% | **16.0** |
| rerank24(重排留24) | 71.1% | 58% | — | 23.9 |

> **Judge 口徑**:上表正確率為 **gpt-oss-20b judge**。改用 **gpt-4o-mini judge 時 rerank16 = 78.6%**(判定較寬鬆,整體上抬,但 rerank16 仍為最佳)。返回率/全中正確率/avg summary 與 judge 無關,不變。

## 4. 核心發現
1. **正確率跟著 precision(context 乾淨度)走,不是 recall**。加料法(topk/extra 低門檻)都是 recall↑ 但 precision 砍半、context 翻倍 → 稀釋 → 正確率上不去。
2. **rerank16 是全域最佳**:73.5%(排除 KU 為 75.5%),context 只要 16 個(tk32 要 32)。6 類贏 5 類(user/multi/preference/temporal/knowledge),唯一輸 assistant(壓縮文本)。
3. **rerank 的價值不在「多撈 gold」**:翻轉分析顯示 base✗→rerank✓ 的 46 題裡,**68% 是在 gold 沒變多(甚至更少)下修好的**。價值在「讓整個 context(gold + 非 gold)都貼題」。
4. **top-16 是甜蜜點**:rerank24 反而掉到 71.1%(4/6 類變差)。手動查 flip:rerank24 evidence 是 rerank16 的超集(多 8 個),gold 相同或更多卻答錯 → 證明多的 context 直接造成稀釋(計數/彙整題把答案數錯)。
5. **reranker 排 gold ≈ cosine**(中位名次 7 vs 8,救進/踢出 top16 = 254/250,淨 +4;assistant 甚至把 gold 往下排 52%)。→ reranker 不是更會找 gold,而是更會排「非 gold 的相關性」。
6. **ablation(進行中)**:widecos16(wide→cosine top-16,無 reranker)在 KU = 57.7%,比 rerank16 的 62.8% 低 5.1pp(recall/precision 幾乎一樣)→ 至少在 KU,reranker 值得成本。其餘類別待驗證。

## 5. 程式改動(都在 flag 後、預設關,可安全提交)
`KG/pipeline/retriever.py`(config)+ `KG/pipeline/retrieval_steps/evidence.py`(`_build_evidence_split`):
- `summary_direct_vector_topn`:summary 直接向量檢索的 breadth。
- `summary_direct_vector_min_score`:extra-slot 模式,direct 命中 ≥ 門檻的加在 prov top-K 之上。
- `summary_rerank_topk`:retrieve-then-rerank,wide pool → cross-encoder 重排 → 留 top-N(pool cap 40、batch 2、OOM 時退回 cosine)。
- `summary_rerank_cosine_only`:ablation flag,跳過 reranker 只取 cosine top-N。

**最佳設定**:`use_split_embeddings=True, summary_direct_vector_topn=50, summary_direct_vector_min_score=0.35, summary_rerank_topk=16`(cosine_only=False)。

## 6. 重要基礎設施注意事項
- GPU 24GB 被 **LM Studio(gpt-oss-20b 答題 LLM)固定佔 15.4GB**,retrieval 只剩 ~8GB。reranker 對大 pool 會 OOM → 已加 pool cap 40 + batch 2 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + OOM fallback。
- 跑實驗一律用 **watchdog 包裝**(自動重啟續跑),`rerun.py` 直跑沒有崩潰保護會半途掛掉。

## 7. 輸出位置
- 最終 top-16 結果:`experiment/longmem/output/rerank16/`(六類合併,499 題,73.5%)
- 對照:`split-embed/`(base)、`sweep-topk{16,24,32}/`、`extraslot-t{50,40,35}/`、`rerank24/`、`widecos16/`(ablation,進行中)
- 分析腳本:`gold_recall_metrics.py`、`summary_score_dist.py`、`gold_upstream_score_dist.py`

## 8. 待辦 / 開放項
- widecos16 ablation 跑完 → 確認 reranker 是否全域值得(還是只 KU 需要)。
- assistant 的 `:a` 壓縮問題:改存 raw / 輕壓縮再 embed(ingest 端,與 rerank 正交),預期補回唯一退步的類別。
- 缺的 1 題(`temporal/gpt4_7ddcf75f`)可單獨補跑。
- 可考慮 assistant 用較大 k、其餘 16 的類別自適應。
