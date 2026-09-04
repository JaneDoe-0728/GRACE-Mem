# Ubiquitous Language（統一語言）

GRACE-Mem 的權威領域詞彙。一個概念只有一個 canonical term，其餘一律列為
aliases to avoid。規格書、PRD、commit message、CLI flag 與程式識別字都只使用
這些詞。

> 專有名詞、型別名、檔案路徑一律保留英文原文，不翻譯。
> 英文版為主文件：[ubiquitous-language.md](ubiquitous-language.md)。兩份內容需同步更新。

**收錄邊界。** 本文件只收 *domain* 詞彙。架構性用語 —— `domain`、`adapters`、
`runtime`、`bootstrap`、作為檔名的 `pipeline` —— 講的是程式碼放在哪裡，
刻意不放進來；那些字說明不了 GRACE-Mem 是什麼。一個字要進入下方表格，必須帶有 GRACE-Mem 專屬的語意：
**Step** 合格，因為它的存在就是為了與 **Stage** 對比；**Adapter** 不合格。

涵蓋範圍：`grace_mem/`（記憶系統）、`experiment/`（benchmark harness）、
`scripts/`（開發工具）。重新產生候選詞清單：

```bash
python3 .claude/skills/uncle-dev-ubiquitous-language/scripts/scan_terms.py --top 40
```

---

## Knowledge graph core（知識圖譜核心）

GRACE-Mem 實際儲存的物件，全部定義於 `grace_mem/utils/common.py`。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Entity** | 知識圖譜中的一個節點：具備型別與可嵌入描述的具名事物。 | `ent`、node、item、concept |
| **Relationship** | 兩個 entity 之間的一條有向邊，端點以 entity **名稱**而非 id 定址。 | `rel`、relation、edge、link、fact、triple |
| **EntityType** | Entity 所帶的分類：Activity、Concept、Date、Event、Location、Organization、Person、Product、Service、Time、Timespan、Topic。 | kind、class、label |
| **ExtractionResult** | 從單一 turn 抽取出的全部內容 —— 一次 LLM 呼叫產出的 entities 與 relationships，尚未經過 resolution。 | extraction、extracted、payload |
| **Provenance** | 記錄某個 entity 或 relationship 來自哪些 turn。 | source、origin、trace |
| **Summary** | 一個或多個 turn 的壓縮重述，與圖譜並存且可獨立被檢索。 | digest、abstract、compressed context |

**這裡不存在的詞彙。** *fact* 與 *memory* 兩者合計出現不到 100 次，且從未被宣告為
型別。不要引入它們：外界口中的「fact」在此就是 **Relationship**，口中的「memory」
指的是整個圖譜加上它的 summaries。

## Conversation source（對話來源）

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Turn** | 單一 speaker 的一次發言，是 ingestion 消費的最小單位。 | message、utterance、line、exchange |
| **Session** | 共享同一時間框架的一串有序 turns。 | conversation、dialogue、thread |
| **Speaker** | 一個 turn 所歸屬的參與者。 | user、role、actor、author |
| **sid** | Turn 的穩定識別碼，格式為 `"session:pair:role"`（例如 `answer_abc:6:u`）。retrieval、gold 標註與 agent filter 都以此定址 turn。 | turn_id、uid、key |

## Ingestion（寫入）

`ingest` 是一個 **stage**。`compress`、`extract`、`sync` 是它的 **step**
（位於 `grace_mem/pipeline/ingest_steps/`），本身不是獨立 stage。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Ingestion** | 將 **Turns** 轉換為已儲存圖譜狀態的能力。指涉做這件事的程式碼，以及 `ingestion/` 套件。 | ingest（那是 stage）、indexing、loading |
| **Ingest** | 為整個 run 呼叫 **Ingestion** 的 benchmark **Stage**。是一個 CLI 值；見 **Stage**。 | ingestion（那是能力）、index、load、import、build |
| **Compress** | 在 extraction 之前縮短 turn 文字的 step。 | summarize（保留給 **Summary**）、shrink、prune |
| **Extract** | 向 LLM 索取某個 turn 中的 entities 與 relationships，產出 **ExtractionResult** 的 step。 | parse、mine、derive |
| **Sync** | 將抽取到的名稱解析至既有節點，並寫入 graph、vector store 與 cache 的 step。 | persist、save、upsert、merge、commit |
| **Extractor** | 對單一物件種類執行 **Extract** 的元件（`EntityExtractor`、`RelationshipExtractor`）。 | miner、parser |
| **Manager** | 對單一物件種類執行解析、合併與持久化的元件（`EntityManager`、`RelationshipManager`）。擁有寫入權責，本身不持有狀態。 | service、handler、repository、DAO |

## Retrieval（檢索）

位於 `grace_mem/pipeline/retrieval_steps/`。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Retrieval** | 將問題轉換為 evidence block 的 stage。 | search（保留給 vector/BM25 的那個 step）、lookup、recall |
| **Query** | 經改寫與嵌入之後、用於搜尋的問題形式。 | prompt、request、input |
| **Evidence** | 交給作答 LLM 的組裝結果：entities、relationships、provenance 與 summaries 的集合區塊。 | context（見〈已標記的歧義〉）、passages、snippets、retrieved docs |
| **Filtering** | 對檢索到的 entities 與 relationships 進行收斂與重排的 step。 | pruning、selection、cleanup |
| **Narrowing** | 以關鍵字與 entity 重疊度，將 evidence block 縮減至與問題相關片段的 step。 | filtering（是不同的 step）、compression、agent filter |
| **Spreading activation** | 從種子 entities 沿圖譜邊擴展的 step。 | traversal、walk、expansion |
| **SummaryScore** | 單一 summary 候選的評分細目。 | rank、weight、relevance |
| **Retriever** | 端到端執行 retrieval stage 的元件。 | searcher、engine、reader |

## Temporal（時間）

位於 `grace_mem/utils/temporal/`。此區用詞本身已相當精確，視為既定不動。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Temporal** | 偵測、分類並解析 turn 與問題中時間表達的能力。指涉 `temporal/` 套件。 | time handling、datetime、chrono |
| **TimeContext** | 相對時間表達所依據的參考框架，錨定在該 turn 自身的時間戳上。 | context（見〈已標記的歧義〉）、frame、now、clock |
| **TemporalMatch** | 一個被偵測到的時間表達 —— 其文字、位置範圍與分類 —— 尚未經 resolution。 | hit、mention、candidate |
| **TimeCategory** | 時間表達的分類（`RELATIVE_DAY`、`SEASON_POINT`、`MONTH_WEEK_RANGE` 等）。 | type、kind、pattern |
| **TimeGranularity** | 已解析區間的粗細度：DAY、WEEK、WEEKEND、MONTH、SEASON、YEAR、TIME、RANGE。 | precision、resolution（見下）、scale |
| **ResolvedTimeRange** | 完全解析為起訖區間的單一時間表達，附帶 provenance。即使表達的是時間點也一律以區間表示。 | timestamp、date、interval |
| **ResolutionStatus** | 表達被解析的完整程度：RESOLVED、PARTIALLY_RESOLVED、AMBIGUOUS、UNRESOLVED、INVALID。是分級的，絕不塌縮成布林值。 | state、success、valid |
| **TemporalConstraint** | 一個 `ResolvedTimeRange` 加上連結問題與它的運算子（「in July」與「before July」的差別）。 | filter、predicate、range |
| **Anchor** | 模糊時段詞（如「morning」）所對應的設定時鐘時間。 | default、base、pivot |

## Experiment harness（實驗框架）

兩個 benchmark —— **LoCoMo**（`experiment/locomo/`）與 **LongMem**
（`experiment/longmem/`）—— 以不同的調度方式執行相同的三個 stage。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Run** | 在單一 `RunConfig` 之下對某個 benchmark 的一次端到端執行。 | job、session（保留給對話）、trial、experiment |
| **Stage** | 恰好三個頂層階段之一：`ingest`、`qa_eval`、`judge`。由 CLI flag 選擇，也是 run 能夠斷點續跑的單位。 | step（保留給 pipeline 內部）、phase、task |
| **Step** | grace_mem pipeline 內部的一個單元（`ingest_steps/`、`retrieval_steps/`）。永遠不可由 CLI 定址。 | stage、substage、module |
| **Probe** | 對一次已完成 run 的 log 所做的單一編號診斷檢查（`step2_ingest` … `step9_evidence`）。只讀 artifact，不執行任何 pipeline。 | step（見〈已標記的歧義〉#8）、check、test、assertion |
| **Sample** | 一個 LoCoMo 對話實例，以整數 `sample_index` 定址。LoCoMo 的平行化單位。 | record、item、case、instance、example |
| **Dataset** | 一個 LongMem 資料夾及其設定。LongMem 的平行化單位，對應 LoCoMo 的 **Sample**。 | data、corpus、collection |
| **Runner** | 規劃工作單位並為每個單位派生一個 **Worker** 的調度器。 | processor、driver、executor、controller |
| **Worker** | 端到端執行單一 **Sample** 或 **Dataset** 的子行程。只透過檔案與外界溝通。 | job、task、child、thread |
| **Artifact** | Stage 為了後續 stage 或分析而寫出的任何檔案：CSV、stats JSON、snapshot、error bundle。 | output、result、dump、export |
| **Snapshot** | 已儲存的 graph 與 vector store 狀態，run 可由此續跑 ingestion。 | checkpoint、backup、cache（見 **Cache**） |
| **RunConfig** | 單一 run 由 CLI 解析而來、不可變且可雜湊的設定。 | args、options、settings、params |

## Evaluation（評測）

位於 `experiment/common/evaluation/`。

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **QAEval** | 為 run 中每個問題檢索 evidence 並生成答案的 stage。它**不**評分。 | eval、evaluation、inference、answering |
| **Judge** | 以 LLM 判定生成答案是否符合 gold answer 的 stage。 | eval、grade、score、assess |
| **JudgeEngine** | Judge stage 所套用、依 benchmark 而異的 prompt、重試與投票策略。 | judge（那是 stage）、grader、scorer |
| **Verdict** | Judge 對單一問題所下的二元判定。 | score（保留給數值）、result、correctness、grade |
| **Score** | 由 verdicts 或文字重疊度計算出的數值指標（accuracy、F1、BLEU-1）。 | metric、rating、verdict |
| **Gold** | 用來比對生成答案的參考答案。 | truth、ground truth、expected、label |
| **Category** | LoCoMo/LongMem 問題所屬的類別，用於拆解 accuracy。 | type、bucket、label、tag |
| **Flip** | 在兩次 run 之間 verdict 改變的問題；這是單純 accuracy 差值所掩蓋的訊號。 | regression、diff、change |

## Storage（儲存）

| Term | 定義 | Aliases to avoid |
| --- | --- | --- |
| **Graph** | 保存單一 run 之 entities 與 relationships 的 FalkorDB 知識圖譜。 | KG、db、store、neo4j |
| **VDB** | 保存 entity 與 relationship 嵌入向量的 vector store。識別字中寫作 `VDB`，行文中寫作「vector store」。 | vector db、chroma、index、embedding store |
| **Cache** | Extraction 結果的磁碟 pickle，使重跑得以略過 LLM 呼叫。 | snapshot、store、memo |
| **Artifacts directory** | 每個 run、每個 sample 專屬的目錄，所有 store 都由此推導路徑（`KG_ARTIFACTS_DIR`）。 | output dir、workdir、run dir |

---

## Relationships（概念關係）

- 一個 **Session** 包含有序的 **Turns**；每個 **Turn** 由其 **sid** 識別。
- **Ingest** 消費 **Turns**，在 **Graph**、**VDB**、**Cache** 中產出 **Entities**、**Relationships** 與 **Summaries**。
- **Ingest** 由 **Compress**、**Extract**、**Sync** 三個 step 依序組成。
- **Extract** 產出 **ExtractionResult**；**Sync** 將其解析並寫入各儲存層。
- **Relationship** 以 **Entity** 名稱指涉其端點；一條指向從未被抽取之 entity 的邊，會在 **Sync** 階段被丟棄。
- **Retrieval** 透過 **Filtering**、**Spreading activation**、**Narrowing** 等 step，將 **Query** 轉換為 **Evidence** block。
- **QAEval** 呼叫 **Retrieval** 後生成答案；**Judge** 將該答案與 **Gold** 比對並產出 **Verdict**。
- **Score** 由 **Verdicts** 聚合而來，可再依 **Category** 拆解。
- 一次 **Run** 以 `ingest → qa_eval → judge` 的順序執行各 **Stage**。
- **Runner** 為每個 **Sample**（LoCoMo）或每個 **Dataset**（LongMem）派生一個 **Worker**。
- **Worker** 只透過 **Artifacts** 與其 **Runner** 溝通。
- **TemporalMatch** 依 **TimeContext** 解析為 **ResolvedTimeRange**，並帶有 **ResolutionStatus**。

---

## 範例對話

> **開發者：** 當一個 **Worker** 跑完 `ingest` **Stage**，它實際上寫出了什麼？
>
> **領域專家：** 它那個 **Sample** 裡的每個 **Turn** 都經過了 **Compress**、**Extract**、**Sync**。所以 **Graph** 裡有 **Entities** 和 **Relationships**，**VDB** 裡有它們的嵌入向量，**Cache** 裡則存著原始的 **ExtractionResults**，讓重跑時可以略過 LLM。
>
> **開發者：** 那 **Summaries** 呢 —— 它們算 **Entities** 嗎？
>
> **領域專家：** 不算。**Summary** 是 turns 的壓縮重述，可以被獨立檢索。它在 **Retrieval** 階段是分開評分的，永遠不會變成節點。
>
> **開發者：** 所以我們在 `qa_eval` 產出 **Score**，對吧？
>
> **領域專家：** 不對 —— `qa_eval` 只檢索 **Evidence** 並生成答案，僅此而已。`judge` **Stage** 才把答案跟 **Gold** 比對並產出 **Verdict**。**Score** 是之後聚合 **Verdicts** 才算出來的。三件不同的事，三個不同的 stage。
>
> **開發者：** 所以講「eval 說有 62%」是錯的說法。
>
> **領域專家：** 對。是 **Judge** 產出了 **Verdicts**；在那些 verdicts 之上算出的 accuracy **Score** 是 62%。

---

## 已標記的歧義

**1. 「Context」同時指兩件無關的事。** `ContextFilter`
（[filtering.py:38](../grace_mem/pipeline/retrieval_steps/filtering.py#L38)）處理的是
檢索到的 entities 與 relationships；`TimeContext`
（[types.py:97](../grace_mem/utils/temporal/types.py#L97)）則是時間參考框架。兩者毫無關聯。
*建議：* **TimeContext** 維持原樣 —— 對「參考框架」而言它就是精確的詞。其餘所有
地方的 "context" 一律退役，改用 **Evidence**；`ContextFilter` 更名為 `EvidenceFilter`。

**2. 「Category」同時指兩件無關的事。** `TimeCategory` 分類的是時間表達；
`CategoryScore`（[score.py:68](../experiment/common/evaluation/score.py#L68)）拆解的是
依問題類別的 accuracy。
*建議：* 兩者都保留前綴，永遠不以裸露的 "Category" 稱呼。評測那一側應改為
**QuestionCategory** 讓前綴顯性化；`CATEGORIES` / `LONGMEM_CATEGORIES` 改為
`QUESTION_CATEGORIES`。

**3. 「Runner」與「Processor」是同一個概念的兩個名字。**
`experiment/locomo/pipeline/runner.py` 與 `MultiDatasetProcessor`
（[processor.py:93](../experiment/longmem/pipeline/processor.py#L93)）都在規劃工作單位、
驅動逐單位執行。命名分歧的原因單純是兩個 benchmark 分開開發。
*建議：* **Runner** 為 canonical。`MultiDatasetProcessor` 更名為 `DatasetRunner`。
`EntityOpsProcessor` 保留 —— 那裡的 "Processor" 意思是「裁決一批資料」，是不同且
正當的角色。

**4. 「qa_eval」根本沒有在 evaluate。** 該 stage 做的是檢索與生成；判定在 `judge`，
計分在那之後。這個名字讓每一次談到「eval 結果」時，都無法分辨指的是生成答案、
verdicts、還是 accuracy 數字。
*建議：* 概念名為 **QAEval**，並由上述定義把它釘死在「檢索 + 生成」。
**刻意不改這個 stage 的識別字** —— `qa_eval` 是 CLI 參數值、artifact 目錄名，也是
所有歷史結果 CSV 的欄位前綴。修正行文用語即可，識別字保留。

**5. `ent` / `rel` 縮寫的使用量與全稱不相上下。** `rel` 676 次 vs `relationship`
565 次；`ent` 389 次 vs `entity` 1400 次。
*建議：* 全稱為 canonical。`ENT_FILE` → `ENTITY_CACHE_FILE`，
`REL_FILE` → `RELATIONSHIP_CACHE_FILE`。此改動安全：磁碟上的檔名本來就已是
`entities_cache.pkl` / `relationships_cache.pkl`
（[cache.py:30-31](../grace_mem/storage/cache.py#L30-L31)），因此不影響 artifact。

**6.「Turn」在兩個不同粒度上各被宣告一次。** `SpeakerTurn`
（[evidence_speaker_enricher.py:21](../grace_mem/utils/evidence_speaker_enricher.py#L21)）
是 speaker + text；`Turn`（[corpus.py:23](../experiment/agent_filter/corpus.py#L23)）
則以 sid 定址且帶有位置。
*建議：* 這是一個概念的兩種投影，不是歧義。兩者都保留，但 `SpeakerTurn` 應被理解為
「僅填入 speaker 與 text 的 **Turn**」。不要再引入第三種寫法。

**7. 其餘縮寫配對。** 每一組都用 AST 掃描分離「識別字」與「字串常值」後逐一查核。
結果各不相同，其中兩項原本的裁決根本是錯的。

| 配對 | 原裁決 | 實際狀態 |
| --- | --- | --- |
| `ctx` → **Evidence** / **TimeContext** | 依 #1 | **已完成。** `ctx_dataset`、`ctx_stage`、`ctx_base` 保留：那是 token-tracking 語境與 prompt 語境，是這個字的第三、第四種意思，都不是 Evidence |
| `art` → **Artifact** | 寫全稱 | **已完成。** 44 個識別字，不涉及任何 artifact schema，也沒有對外文件引用 |
| `vdb` | **VDB** 本來就是識別字中的 canonical 拼法 | **本來就符合。** 無事可做；原本的寫法暗示相反，是誤導 |
| `stat` → `stats` | — | **撤銷。** 25 處全部是 `Path.stat()`。掃描器的 `stat`/`statuse` 配對是它自己詞幹處理的產物，不是真的同義詞 |
| `data` vs `dataset` | `data` 僅用於路徑常數 | **本來就符合。** `DATA_ROOT`、`SCRIPT_DATA_DIR`、`LOCOMO_DATA`、`DATA_JSON` 都是路徑常數；`graph_data`、`export_data` 指的是一份 payload，不是 **Dataset** 這個領域詞 |
| `meta` → **metadata** | 寫全稱 | **未做。** 42 個名稱中 21 個凍結：`"metas"` 是 BM25 pickle 內的 key，`"meta"` 是 `cases/<id>.json` 的 key，`entity_meta`/`rel_meta` 是參數。可改的 21 個與它們交錯 —— 與 #5 完全相同的「只改一半」問題 |
| `vec` → **vector** | 數值用全稱 | **未做。** `summary_vec_threshold`、`entity_vec_threshold`、`relationship_vec_threshold` 同時是 config key 與 `DatasetConfig` 欄位，動不了；`query_vec` 是跨七個模組的參數。改掉其餘的會讓兩半互相矛盾 |
| `qa` 在 `qa_eval` 之外 | 移除 | **未做。** `"qa_json"` 是 worker、judge 與路徑解析器共用的 dataset-kind 查詢 key；真正自由的只有 5 個名稱 |
| `eval` | 依 #4 寫全稱 | **未做**，而且大致無意義 —— 123 處中多數就是 `qa_eval` 本身，而 #4 已將它凍結 |

`meta`、`vec`、`qa` 與 #5 呈現同一個模式：必須先決定一份凍結名單 —— 哪些 config key
可以改、既有儲存的 key 需要什麼相容性 —— 才有辦法一致地掃除。那是設計問題，不是更名。

**8.「step」有第三個沒人宣告的意思。**
`experiment/longmem/helpers/analysis_cases.py` 定義了 `step2_ingest`、
`step3_has_answer` … `step9_evidence` —— 這些是走訪一次已完成 run 的 log 與
artifact 的編號探針。它們既不是 CLI 的 **Stage** 也不是 pipeline 的 **Step**：
不執行任何 pipeline，只讀 run 留下來的東西。
*建議：* 這個概念叫 **Probe**，已於上定義。將 `stepN_<thing>` 更名為
`probe_<thing>`；那些編號表達的閱讀順序在呼叫端本來就已經決定，反而讓這些
function 不重新編號就無法調整順序。

*實作時發現的更正。* 本條先前寫「不出現在任何 artifact」，那是錯的：`analyze_one`
回傳的 dict 帶著 `"step2_ingest"` … `"step9_evidence"` 這些 key，`collect_cases`
把它寫進 `cases/<id>.json`，summary 工具再讀回來。**函式已更名，dict key 沒有** ——
理由與 `qa_eval` 保留原名相同。

---

## 已排除的疑似項目

Scanner 有回報、經查證後刻意不列入詞彙表的項目：

- `scripts/gen_dep_graph.py` 中的 **Graph** —— 那是開發工具用的原始碼相依圖，與知識
  圖譜無關。`scripts/` 不屬於領域範圍。
- `scripts/download_datasets.py` 中的 **Dataset**（`DatasetFile`）—— 那是帶 checksum 的
  釘選下載檔，不是 LongMem 的執行單位。理由同上。
- 跨 `pipeline`、`services`、`utils` 宣告的 **Entity** / **Relationship** —— 同一概念
  的分層角色（`*Extractor`、`*Manager`、模型本身），角色後綴詞彙已足以區分。不是歧義。
- `locomo/stages/` 與 `longmem/stages/` 中重複的 **IngestStage** / **QAEvalStage** /
  **JudgeStage** —— 是同一個 stage 概念針對兩個 benchmark 的平行實作。這是設計上的
  命名重合，不是用詞漂移。（它們是否該共用程式碼是另一個問題，本文件不回答。）
- **Judge** 之於 `JudgeEngine` 與 `JudgeStage` —— engine 是策略，stage 是套用它的
  框架階段。兩者皆已於上定義。
- `EntityManager`、`RelationshipManager`、`VDBManager` 的 **Manager** 後綴 —— 一致地
  表示「擁有並持久化某一種東西」。雖然籠統，但用法統一，不值得為此更名。
- `filter`/`filtering`、`token`/`tokenize`、`final`/`finalize`、`day`/`daypart` ——
  這些是動詞／名詞配對，不是同義詞。

---

## 更名待辦

以下全部以 commit 的形式落在同一個 branch `refactor/terminology` 上，依風險排序。
這個 branch 排在 package 搬移**之前**，這樣就不會有東西被更名到一個它即將離開的
目錄裡。

| 順序 | Commit | 解決 | 風險 |
| --- | --- | --- | --- |
| 1 | `refactor(analysis): rename numbered steps to probes` —— `longmem/helpers/analysis_cases.py` 的 `stepN_*` → `probe_*` | #8 | 無 —— 單一模組，不暴露於 artifact |
| 2 | `refactor(evaluation): qualify question category scores` —— `CategoryScore` → `QuestionCategoryScore`、`CATEGORIES` → `QUESTION_CATEGORIES` | #2 | 低 —— 需先確認 CSV 欄位名 |
| 3 | `refactor(longmem): rename MultiDatasetProcessor to DatasetRunner` —— 同時 `processor.py` → `runner.py` | #3 | 中 —— 需檢查 LongMem CLI |
| 4 | `refactor(storage): spell out the cache file constants` —— `ENT_FILE` → `ENTITY_CACHE_FILE`、`REL_FILE` → `RELATIONSHIP_CACHE_FILE` | #5，部分 | 無 —— 兩個常數與其區域變數；磁碟檔名本來就是全稱 |
| 5 | `refactor(retrieval): rename ContextFilter to EvidenceFilter` —— 以及 `ctx_*` → `evidence_*` | #1 | 低 —— class 只有 6 個引用點，`ctx_*` 只在 2 個檔案 |

**更大範圍的 `ent_`/`rel_` 掃除不在這個 branch，而且本文件先前為它記的風險
「無 —— 僅內部識別字」是錯的。** 對 `grace_mem`、`experiment`、`scripts` 做的 AST
掃描把 97 個縮寫名稱分類，其中 38 個不可更名：

- `rel_id`、`rel_desc`、`rel_keywords`、`rel_strength` 是 **FalkorDB 的圖屬性
  名稱**，直接從 Cypher record 讀出；`KG_REL` 這個關係型別標籤也是。改了就讀不出
  既有圖譜。
- `ent_topk`、`rel_topk`、`ent_threshold`、`rel_threshold` 一個名字戴四頂帽子 ——
  `experiment_config.py` 的 config key、`DatasetConfig` 欄位、關鍵字引數、參數名 ——
  而且會進到 `run_metadata.json`。

只有 59 個是安全的，而且與凍結的那些交錯：`ent_id2meta` 可改，`rel_id2meta` 不行
（後者是關鍵字引數）。只改一半反而讓詞彙比原本**更不一致**。要做得完整，得先決定
那些凍結名稱 —— 哪些 config key 可以改、圖屬性需要什麼相容性。那是設計問題，
不是更名。

**同樣不在這個 branch 的：**

- **#4，`qa_eval`。** 完全不做 —— CLI 參數值、artifact 目錄名，以及所有歷史結果
  CSV 的欄位前綴。
- **#3b，三個 `*context*` function** —— `assemble_context_from_query`、
  `build_kg_context`、`_render_context_text`。它們的名字被鏡射成 structured log
  的 event 字串（`"build_kg_context_start"`），而
  `experiment/locomo/analysis/flips.py` 與
  `experiment/longmem/analysis/fact_replay.py` 是字面比對。只改 function 名會讓
  event 名變成孤兒；兩邊都改則讀不了既有 log。這需要獨立決策，不該混進一次
  更名掃除。

每個 commit 需驗證：僅更名（無行為變更）· import 已完整更新 · CLI flag 與
artifact 路徑未變 · ruff、mypy、pytest 全綠 · 每個被動到的識別字都與本文件一致。
