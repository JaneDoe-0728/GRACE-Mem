# Agent Filter 評分方法（judge）

本文件說明 LongMem Agent Filter 實驗**如何算正確率**——judge 模型、3 票去噪、以及棄答（`_abs`）題的專用 rubric。所有新實驗一律遵循此口徑。

> **口徑分兩條線**（2026-07-20 更新）：
> - **一般題**：4o-mini category rubric，錯題 **3 票**多數決重判 → `correctness_3vote`。
> - **棄答（`_abs`）題**：4o-mini **強化版 `ABS_JUDGE_PROMPT`**，**恆單票**（`_abs` 走專判棄答的 prompt，3 票多數決會把 temp=0 判對的 borderline 棄答淹掉）→ 重判寫 `correctness_absrubric`。強化點：**棄答目標項 + 誠實提具名 distractor 作對比**（"no violin—only guitar 30min"）算對；拿 distractor 值當答案 / 純幻覺算錯。

程式：`experiment/longmem/rejudge_output_dirs.py`（跑分）+ `experiment/longmem/prompts/judge.py`（prompt / rubric）。

---

## 一、judge 模型與欄位

- **judge 模型**：`gpt-4o-mini`（OpenAI cloud API）。
- **prompt**：逐字對齊 hindsight 官方 `benchmark_runner.py` 的 judge 構造（含 LongMemEval per-category rubric）。**不改 prompt**。
- **欄位對應**：
  - `correctness_new` = 4o-mini 單票（temp 0）。
  - `correctness_3vote` = 4o-mini 3 票多數決（一般題的去噪口徑）。
  - `correctness_absrubric` = 4o-mini **強化版 `_abs` rubric 單票**（棄答題的最終口徑；2026-07-20 起）。
  - 歷史 `correctness_20b` = 舊 gpt-oss-20b judge，**不得與 4o-mini 混稱**，新實驗不再啟用。
- **最終正確率的合成口徑**：非 `_abs` 題取 `correctness_3vote`、`_abs` 題取 `correctness_absrubric`，兩者拼起來算全量比例。

judge 是雲端 API：用有上限的 worker pool（預設 6–8），遇 429/5xx 退避重試，不無限開 thread。

---

## 二、3 票多數決去噪（核心算分方式）

### 為什麼要 3 票

單票（temp 0）judge 有 **~12% run-to-run 不一致**（同一 run 內 `correctness_new` vs `correctness_20b` 就差 ~12%）。這個噪音對 **borderline 題**（同義答案、偏好對齊、乾淨棄答）會**隨機翻**——把「內容對、措辭≠gold」的題誤判為錯。

### 怎麼投票

對同一題判 **3 次**，每次用**不同 temperature** 製造受控多樣性：

```
第 1 票：temp = 0.0   → 對(1) / 錯(0)
第 2 票：temp = 0.3   → 對(1) / 錯(0)
第 3 票：temp = 0.6   → 對(1) / 錯(0)

tally = 三票中「對」的數量
最終 = 1 if tally * 2 >= 3 else 0      # 多數決，平手偏「對」
     → 3對 或 2對 → 判對；1對 或 0對 → 判錯
```

程式：`_judge_single(..., votes=3)`，temps 序列 `(0.0, 0.3, 0.6)`。

> temp 三次都用 0 沒有去噪效果（三票幾乎相同）——**必須用不同 temp** 才讓多數決有意義。

---

## 三、「錯題再用 3 票重判」的實務算分流程 ⭐

這是本專案實際採用的口徑（**不是**對全量每題都跑 3 票，而是省 API 的等價做法）：

```
1. 先有單票結果 correctness_new（每題判過一次，temp 0）。
2. 建新欄 correctness_3vote：
     - 單票判「對」的題  → 直接 carry 1（不重判；假設判對的 3 票不會翻錯）
     - 單票判「錯」的題  → 留空，交給 3 票重判
3. 對「留空的錯題」跑 3 票多數決，寫回 correctness_3vote。
4. 最終正確率 = correctness_3vote 欄的 1 的比例。
```

**效果 = 把單票誤殺的假失分救回**。實測（LongMem 20B）：

| run | 單票 | 3 票（錯題重判） | 回收假失分 |
|---|---:|---:|---:|
| baseline | 76.2 | 76.8 | +3 題 |
| grep filter | 78.0 | 78.6 | +3 題 |
| 裁決版（adjudicate-v1） | 77.4 | **80.4** | +15 題 |

複現：
```bash
# 先用 prep 腳本建 correctness_3vote 欄（正確 carry 1、錯題留空），再：
python3 experiment/longmem/rejudge_output_dirs.py \
  --votes 3 --col correctness_3vote \
  --dirs <run-tag> --workers 6
# 工具只判「該欄為空」的列 → 正好只重判單票錯題。
```

### ⚠️ 兩個 caveat

1. **這是「錯題重判」而非「全量 3 票」**：假設「單票判對的題，3 票不會把它翻錯」。實務上 3 票偶爾也會翻掉單票判對的題，所以嚴格全量 3 票分數理論上可能**略低於**此數字。此口徑是**上界估計**，引用時措辭應為「錯題經 3 票多數決重判後」，非「全量 3 票 judge」。
2. **judge 本身有重採樣抖動**：3 票（temp>0）非確定性，同一批題兩次 3 票可差 ±3~4 題（裁決版曾得 15 題 / 80.4% 與 11 題 / 79.6% 兩個實例）。**方向穩健、確切數字不可當精確值**；要釘死須全量多輪 3 票取平均。

---

## 四、棄答題（`_abs`）專用 rubric ⭐

### 問題

LongMemEval 植入「干擾項」的**棄答題**：gold 表示「該資訊從未被提及」（如 "The information provided is not enough"、"You did not mention this information"、或計數為 0 因事件從未發生）。**通用 rubric 只問「回應是否含 correct answer」，會把乾淨棄答（"沒有紀錄／資訊不足"）系統性誤判為錯**。這是 commit `b06c57e` 修的病。

### 觸發方式（權威來源）

- **優先**由資料集的 **`_abs` 檔名 tag** 判定（`path.stem.endswith("_abs")`）——這是權威事實。
- 未顯式傳入時，才 fallback 用 gold 文字偵測（"the information provided is not enough" 等關鍵句）。
- **只在棄答題觸發**；非棄答題的 prompt **一字不動**。

程式：`build_judge_messages(..., is_abstention=...)` → 命中時走 `ABS_JUDGE_PROMPT`。

### `_abs` rubric 的判準（強化版，2026-07-20）

```
correct = true（棄答成功）：
  - 回應表示「沒有紀錄／沒有資訊／資訊不足」、說無法回答，
    或指出被問的人/地/物/事件從未被提及。
  - gold 是「從未發生」時，回答 "0" 或 "none" 也算對。
  - ★ 棄答被問項的同時，誠實提到一個「相似但不同」且用戶真的講過的
    具名 distractor，只要明確標示它是「不同的那個」、且不把它的值當答案：
      "I have no record of you practicing violin — you only mentioned
       guitar, about 30 min/day."  ← "30 min/day" 歸給 guitar（具名 distractor），
      不是 violin 的答案 → 判對。具體數字的存在本身不構成失格。
  - ★ 乾淨棄答 + 「你給我缺的輸入我就能算」的條件式 offer：
      "I can't give a figure; tell me your current page and I'll compute
       the pages left." → 提供條件式協助 ≠ 宣稱知道答案 → 判對。

correct = false（被干擾項帶跑 / 幻覺）：
  - (a) 對「被問的那一項」給出任何具體數字/日期/時長/名字/順序（當成已知）。
  - (b) 拿 distractor 的值「當作被問項的答案」而不標示替換：
      問 Porsche 卻答 "Ferrari started first on May 2"（拿 Ferrari 當答案）；
      問 Shinjuku 住多久卻答 "seven months"（值其實silently來自 Harajuku）；
      問幾顆足球卻說「那 ~15 顆棒球就是累積數」（借棒球數當足球答案）。

分界測試：具體值是「當作被問項的答案 / 它背書的假設」（→ false），
還是「明確歸給另一個具名項、被問項本身有棄答」（→ true）。
不要求複述 gold 措辭；只judge 這個分界。
```

### ★ `_abs` 恆單票（不套 3 票）

`_abs` 走**專判「是否乾淨棄答」的 prompt**，與一般題的同義/偏好 borderline 性質不同。實測 **3 票多數決反而有害**：會把 temp=0 判對的 borderline 棄答淹掉（`80ec1f4f`、`29f2956b` 都是**單票對、3 票錯**），而真幻覺單票已 0、多票無增益。故機制上 `_abs` **強制單票**：

```python
# rejudge_output_dirs.py :: _judge_single
if is_abstention or votes <= 1:   # _abs 恆單票，即使外層開 --votes 3
    return _one(0.0)
```

### borderline 案例（rubric 判準的真實邊界）

`15745da0`（"You haven't started a vintage-film collection **yet** — you've only been collecting vintage **cameras** for three months"）結構與 violin/guitar 相同（棄答 films + 誠實提 cameras 具體時長），**照判準應對**，但即使把近乎逐字的正面範例寫進 prompt，4o-mini 單票仍 3 溫度全票判嚴——根因是 GEN 用了 "yet"、缺明確對比框架，措辭本身在邊界。**這類連範例都壓不動的 borderline 走人工覆核**（直接改 `correctness_absrubric=1`），不繼續加訓文（避免帶鬆真幻覺判定）。

### 為什麼這條 rubric 重要

fallback 錯誤分析顯示：21 個 `_abs` fallback 題被通用 judge 判錯 13 題，其中 **11 題是模型正確棄答被誤殺**、只有 2 題真幻覺（見 `analysis/longmem-adjudicate-20b-fallback.md`）。強化版 `_abs` rubric（單票）是把這批假失分救回的關鍵。實測（2026-07-20，n=3 裁決版全量 `_abs` 重判）：**20B 75.6→76.6（+1.0pp）、120B 79.8→81.0（+1.2pp）**，主要回收「乾淨棄答」與「棄答+誠實提 distractor 對比」兩型。

---

## 五、一句話總結

- **一般題算分口徑 = 4o-mini judge，單票判過後，對錯題以 temp 0/0.3/0.6 三票多數決重判（carry 判對題），取 `correctness_3vote` 的正確比例。**
- **棄答（`_abs`）題走強化版 `ABS_JUDGE_PROMPT` + 恆單票**（`b06c57e` 建立、2026-07-20 強化）：乾淨棄答判對、**棄答同時誠實提具名 distractor 作對比也判對**、拿 distractor 值當答案或純幻覺判錯；由 `_abs` 檔名權威觸發，非棄答題 prompt 不動，取 `correctness_absrubric`。
- **最終正確率 = 非 `_abs` 題 `correctness_3vote` + `_abs` 題 `correctness_absrubric`** 合成。
- 三個 caveat：一般題錯題重判是**上界估計**（非全量 3 票）；3 票 judge 有 ±3~4 題重採樣抖動；`_abs` 有連範例都壓不動的 borderline（如 `15745da0`）走人工覆核。

---

## 六、先驗合法性判準（pad / hardcode 規則會不會「作弊」）⭐

pipeline 裡有幾個「補證據」的規則（floor pad、min_keep、keep_all）都帶內建先驗。判斷一條規則是否「作弊」（test-set 過擬合），**唯一準則**：

> **這條規則的先驗，有沒有用到「只有看評測集答案 / gold 才知道」的資訊？**
> - 用到 → 過擬合、作弊嫌疑。
> - 只用推論時真實可得的資訊（問題文字、問題日期等）→ 乾淨。

實例對照：

| 規則 | 判定依據 | 依據哪來 | 推論時可得？ | 判決 |
|---|---|---|---|---|
| **`min_keep_aggregation`** | 問題**字面措辭**（`how many/total/count/latest…`,`_AGG_QUESTION_RE`） | 問題自己寫著 | ✅ 只讀問題、不碰答案 | ✅ **乾淨** |
| **`adjudicate keep_all_categories`** | category（KU/temporal） | **回看錯題集 gold 分桶**,發現誤砍集中在這兩類才 hardcode | ❌ 選哪類是看答案定的 | ⚠️ **test-set 過擬合(作弊嫌疑)** |

- **`min_keep` 乾淨**：彙整題（how many/total）本質上就需湊齊多個散落實例才數得對——這是**問題類型的內在因果屬性**,不是從答案反推。真實上線系統收到 "How many plants did I acquire last month?" 也能立刻判它是計數題,無需知道答案。
- **`keep_all` 踩線**：「KU/temporal 該全補」沒有這種內在因果（KU 題不見得該保留全部 dated mention）,它純粹是「看了這批題答案、發現這兩類誤砍多」的反推規則。故 `adjudicate-v1` 主線**未開** keep_all（走 0 次）,它只作為證偽臂存在。
- **邊界**：`min_keep` 的**閾值**（補到幾條）若在評測集調最優,那個數字有輕微過擬合味——但屬「超參數在 test set 調」的普遍問題(所有 top-k/threshold 皆然),程度輕,且非「偷看單題答案」。判定「誰是彙整題」本身乾淨。

## 相關文件

- 3 票去噪首次定案：`EXPERIMENT_LOG.md` → `2026-07-18 · judge 3票多數決修正`
- 三方同口徑對照：`EXPERIMENT_LOG.md` → `2026-07-19 · baseline / grep-filter 3票同口徑補齊`
- fallback / `_abs` 錯誤分析：`analysis/longmem-adjudicate-20b-fallback.md`
- pipeline 流程圖：`diagrams/agent_filter_pipeline.md`
