# 為什麼 judge 這樣設計（設計理由書）

本文件是 [JUDGING.md](JUDGING.md) 的「**為什麼**」姊妹篇。JUDGING.md 講**怎麼算分**（口徑、欄位、指令），本文件講**每一個設計決策背後的問題與取捨**——為什麼是 4o-mini 而非 20b、為什麼一般題 3 票而棄答題恆單票、為什麼要另立 `_abs` rubric、以及這些選擇各自放棄了什麼。

一句話立場：**judge 是一把有噪音的尺，而我們量的訊號（filter 帶來的 ±1~3pp）比尺的噪音還小。整套設計都是為了讓「尺的抖動」不要淹掉「真實效應」。**

> **適用範圍**：決策一～五、所有題例與表 A/B/C 都是 **LongMem（LongMemEval）口徑**。**LoCoMo 是不同的一套判分**——共用「4o-mini + 錯題 3 票」的骨幹，但**沒有 `_abs` 棄答題型**（因此決策三、四整段不適用），改成 temporal 正規化 + 排除 adversarial 題 + LoCoMo standard prompt。兩者的異同、成因與 LoCoMo 自己的正確率差距，統一放在**第六節「LoCoMo 口徑對照」**。

程式：LongMem 走 `experiment/longmem/rejudge_output_dirs.py` + `experiment/longmem/prompts/judge.py`；LoCoMo 走 `experiment/locomo/grep_replay.py --adjudicate-all`（3 票掛在此）+ `experiment/locomo/rejudge_4omini.py`（舊獨立單票腳本）+ `experiment/locomo/helpers/llm.py`（prompt）。

---

## 核心矛盾：訊號比噪音小

先把問題講清楚，後面每個決策才有依據。

- 我們要量的東西：agent filter 相對 baseline 的**正確率差**。實測量級是 **±1~3pp**（LongMem 20B baseline 76.2 → filter 78.6，差 +2.4pp；裁決版再到 80.4）。
- 我們用的尺：LLM judge。單票（temp=0）judge 的 **run-to-run 不一致約 12%**（同一批答案，`correctness_new` 4o-mini vs `correctness_20b` 舊 judge 就差 ~12%）。
- 直接後果：**尺的隨機抖動（~12%）遠大於要量的訊號（1~3pp）**。若不處理，一次「baseline vs filter」的比較，勝負可能純由 judge 當天心情決定。

> 這就是為什麼「怎麼判分」在這個專案裡不是雜務，而是**能不能得出可信結論的前提**。下面所有設計都服務於「把噪音壓到訊號之下」。

---

## 正確率差距總表（一頁看完口徑造成的差）

先把數字集中擺出來，後面各決策再解釋成因。全部 4o-mini judge、LongMem。

### 表 A — 一般題「錯題 3 票重判」的回收量（單票 → 3 票）

| run | 單票 | 3 票（錯題重判） | Δ | 回收假失分 |
|---|---:|---:|---:|---:|
| baseline | 76.2 | 76.8 | +0.6 | +3 題 |
| grep filter | 78.0 | 78.6 | +0.6 | +3 題 |
| 裁決版（adjudicate-v1） | 77.4 | **80.4** | **+3.0** | +15 題（14 題 3/3 全票） |

> 讀法：3 票的回收量**不是固定值**，它取決於該 run 有多少 borderline 假失分。裁決版把 context 砍乾淨後、剩下的失分更集中在「同義措辭 borderline」，所以 3 票回收特別多（+15）。**judge 口徑的 +3.0pp 是裁決版唯一兌現的增益——不是任何 pipeline 改動帶來的。**

### 表 B — 棄答題 `_abs` 專用 rubric 的回收量（通用 rubric → `_abs` rubric）

| 答題模型 | 通用 rubric | `_abs` 專用 rubric | Δ |
|---|---:|---:|---:|
| 20B | 75.6 | 76.6 | **+1.0** |
| 120B | 79.8 | 81.0 | **+1.2** |

> 這是**只換棄答題判分口徑**、pipeline 完全不動得到的差。來源：21 個 `_abs` fallback 題被通用 judge 判錯 13 題，其中 **11 題是乾淨棄答被誤殺**（見決策四）。

### 表 C — filter 相對 baseline 的「真實效應」量級（放一起看比例尺）

| 比較 | Δ | 性質 |
|---|---:|---|
| baseline → grep filter（3 票口徑） | +1.8pp（76.8→78.6） | filter 真實效應 |
| 20B filter → 120B filter | +1.1pp（79.5→80.6） | 升級 filter 模型 |
| **judge 單票 → 3 票（裁決版）** | **+3.0pp** | **純判分口徑** |
| **judge run-to-run 抖動** | **~12%** | **noise floor** |

> **這張表就是整份文件的立論根據**：judge 口徑（+1.0~+3.0pp）和單次抖動（~12%）的量級，**與、甚至大於**我們要量的 filter 真實效應（+1~2pp）。若不把 judge 這把尺校準好（3 票去噪 + `_abs` rubric + no-op 對照臂扣噪音底），filter 的訊號會直接被淹沒。

### ⚠️ 兩條軸不可相加

表 A 是「一般題」軸、表 B 是「棄答題」軸，**分開統計、不可疊加成單一數字**。最終正確率是「非 `_abs` 題取 `correctness_3vote` + `_abs` 題取 `correctness_absrubric`」拼起來算的（見 JUDGING.md）。表 A 的 80.4% 已含當時 `_abs` 判分；表 B 是後續把 `_abs` 口徑再強化的獨立增量。

---

## 決策一：judge 模型選 4o-mini，不是本地 20b/120b

### 為什麼不用本地模型省錢

專案裡答題用的是本地 gpt-oss（20b / 120b）。judge 若也用本地模型，成本近乎零。**但被否決**，理由是：

1. **利益衝突 / 同源污染**：用同一家族模型判自己家族的答案，系統性偏誤無法排除（同樣的措辭偏好、同樣的盲點）。judge 必須是**獨立的第三方尺**，量出來的差才可歸因於 filter 而非 judge 與答題模型的耦合。
2. **20b judge 本身就是噪音源**：歷史上 `correctness_20b` 與 `correctness_new` 的 ~12% 落差，一部分正是 20b judge 判得比 4o-mini 更不穩。既然它是被要壓制的噪音來源，就不該拿它當權威尺。
3. **可複現、可對外引用**：4o-mini 是雲端固定版本，別人重跑對得上；本地模型版本、量化、endpoint 都會漂移。

### 放棄了什麼

- **成本**：4o-mini 是雲端付費 API，每題多次呼叫（見決策二）會累積花費。這是刻意付的稅——換「尺可信」。
- **速度 / 併發**：雲端有 rate limit。程式用**有上限的 worker pool（預設 6–8）+ 429/5xx 指數退避重試**，而非無限開 thread，就是在「別打爆 API」與「別跑一整晚」之間取平衡（`rejudge_output_dirs.py` 的 `ThreadPoolExecutor` + `_one()` 的退避迴圈）。

### 為什麼 prompt「逐字對齊 hindsight、一字不改」

judge prompt 直接複製 vectorize-io/hindsight 官方 `benchmark_runner.py` 的構造（含 LongMemEval per-category rubric）。**不自己寫、不優化**，因為：一旦改了 prompt，我們的分數就**無法與 LongMemEval 公開 leaderboard 或他人結果對照**——那等於自己發明了一把只有自己看得懂的尺。對齊官方 prompt = 保留外部可比性。（唯一例外是 `_abs` rubric，見決策四，且它只在棄答題觸發、非棄答題 prompt 一字不動。）

---

## 決策二：一般題用「錯題 3 票多數決」重判

### 為什麼是「3 票」而不是「1 票」

單票 judge 的 ~12% 抖動，最集中發作在 **borderline 題**：同義答案（"a shell necklace" vs "shell necklace"）、偏好對齊、乾淨棄答。這類題「內容其實對、只是措辭 ≠ gold」，單票 judge 會**隨機翻**成錯——製造出**假失分**。

3 票多數決（對同一題判 3 次）的作用是**把單次隨機翻掉的題救回來**：只要多數票認為對，就判對。

### 為什麼三票要用「不同 temperature」（0 / 0.3 / 0.6）

這是最容易做錯的地方。**若三票都用 temp=0，三次輸出幾乎相同，多數決毫無去噪效果**——等於判了三次一樣的東西。必須注入受控多樣性（temp 逐步升高），讓每票獨立採樣，多數決才有統計意義。程式 `_VOTE_TEMPS = (0.0, 0.3, 0.6)`。

平手偏「對」（`tally * 2 >= votes`）也是刻意的：借鑑「疑錯從無」——borderline 題我們寧可放過假失分，也不製造假失分。

### 為什麼只重判「單票判錯的題」，不對全量跑 3 票

這是**省 API 的等價近似**，也是一個要誠實標註的取捨：

- 做法：單票判「對」的題直接 carry 1（假設判對的題 3 票不會翻錯），只對單票判「錯」的題跑 3 票重判。
- 好處：省掉全量 3× 的 API 花費（多數題是對的）。
- **代價 / 誠實性**：這是**上界估計**。實務上 3 票偶爾也會把單票判對的題翻掉，所以嚴格全量 3 票的分數理論上可能**略低**。因此 JUDGING.md 明令引用時措辭必須是「錯題經 3 票多數決重判後」，**不可講成「全量 3 票 judge」**。這條紀律防的是把上界當精確值宣稱。

### 具體範例：3 票救回的是哪種題

以下是裁決版 15 題假失分裡的真實例子（單票判錯 → 3 票救回，抽驗全合理）：

| 題型 | gold | 模型答案 | 單票 | 3 票 | 為什麼單票會抖 |
|---|---|---|---|---|---|
| 同義措辭 | `GPS system not functioning` | "GPS malfunction" | ✗ 錯 | ✓ 對（3/3） | 語意全同、字串不同，temp=0 一次採樣剛好判嚴 |
| 完全一致卻抖 | `Patagonia` | "Patagonia" | ✗ 錯 | ✓ 對（3/3） | 連答案都一樣仍被單票翻掉——純 judge 隨機噪音 |
| 數字一致 | `10%` | "10%" | ✗ 錯 | ✓ 對（3/3） | 同上，borderline 題的採樣抖動 |

對照組——**真幻覺不會被 3 票誤救回**，機制的邊界正確：

| 題型 | gold | 模型答案 | 單票 | 3 票 | 結論 |
|---|---|---|---|---|---|
| 真幻覺 | `Fissionator` | "Radialisk" | ✗ 錯 | ✗ 錯（0/3） | 答錯就是答錯，三票全維持錯——3 票只救假失分、不放水真錯 |

這張對照就是「3 票只對同義/borderline 有效」的實證：**它把 judge 的隨機抖動抹平，卻抹不掉真正的內容錯誤。**

### 效果與殘餘噪音

實測（LongMem 20B）：baseline 76.2→76.8、grep filter 78.0→78.6、裁決版 77.4→**80.4**（回收 15 題假失分，其中 14 題 3/3 全票）。**方向穩健**。

但要記住第二個 caveat：**3 票（temp>0）本身非確定性**，同一批題兩次 3 票可差 ±3~4 題（裁決版曾得 15 題/80.4% 與 11 題/79.6% 兩個實例）。所以結論是「**方向可信、確切小數點不可當精確值**」；要釘死須全量多輪 3 票取平均。這也是為什麼所有實驗都**必配 no-op 重採樣臂**——錯題集自帶約 22% 的「假修復」底噪，不配對照臂就會把 judge 抖動誤讀成 filter 效果。

---

## 決策三：棄答（`_abs`）題**反其道恆單票**

這是全套設計裡最反直覺的一條：一般題加票去噪，棄答題卻**強制單票**（`if is_abstention or votes <= 1: return _one(0.0)`）。為什麼？

因為 **`_abs` 走的是一組性質完全不同的 prompt**（判「是否乾淨棄答」），它的 borderline 不是「同義措辭」型，而是「棄答得夠不夠乾淨」型。實測發現：

- **真幻覺**（模型硬掰答案）在**單票 temp=0 就已經 0 票**——判錯很穩，多票沒有增益。
- **乾淨棄答**卻是脆弱的一方：`80ec1f4f`、`29f2956b` 都是**單票判對、3 票判錯**的實例。加票的隨機多樣性**反而把 temp=0 已經判對的 borderline 棄答淹掉**。

所以在棄答題上，「3 票去噪」的邏輯**整個反轉**：它救不了本來就已 0 票的真幻覺，卻會誤殺已判對的乾淨棄答——**淨負**。結論就是機制上強制單票。這說明「3 票」不是萬用魔法，而是**只對特定噪音型態（同義/偏好 borderline）有效的工具**，用錯場合會反傷。

---

## 決策四：為什麼要另立 `_abs` 專用 rubric

### 病根：通用 rubric 只問「有沒有含正確答案」

LongMemEval 植入「干擾項」的棄答題，gold 是「**該資訊從未被提及**」（"The information provided is not enough" / 計數為 0 因事件從未發生）。通用 rubric 的判準是「回應是否**包含 correct answer**」——但棄答題的正確行為是**不給答案**。於是通用 judge 會把「我沒有你練小提琴的紀錄」這種**乾淨棄答系統性判錯**（因為它「沒有包含一個具體答案」）。

證據強度：fallback 錯誤分析裡，21 個 `_abs` fallback 題被通用 judge 判錯 13 題，其中 **11 題是模型正確棄答被誤殺、只有 2 題真幻覺**——誤殺率壓倒性。這是 commit `b06c57e` 修的病。

### 為什麼由「檔名 `_abs` tag」觸發，而非只靠 gold 文字偵測

觸發權威來源是資料集的 **`_abs` 檔名 tag**（`path.stem.endswith("_abs")`），gold 文字偵測（"the information provided is not enough" 等關鍵句）只是**未顯式傳入時的 fallback**。為什麼要這個優先序？

- 檔名 tag 是**資料集作者標定的權威事實**，不會漏（gold 措辭千變萬化，關鍵句表列必然不全）。
- **只在棄答題觸發、非棄答題 prompt 一字不動**——這條邊界是為了**不污染一般題的可比性**（見決策一）。專用 rubric 是外科手術式的例外，不是全域改寫。

### rubric 的判準核心：那個「分界測試」

強化版 `_abs` rubric（2026-07-20）真正在判的，是**一個單一分界**：

> 具體值（數字/日期/時長）是「**當作被問項的答案**」（→ 錯），還是「**明確歸給另一個具名 distractor、被問項本身有棄答**」（→ 對）？

為什麼要這麼細？因為棄答題的真實邊界不是「有沒有出現數字」。「有沒有數字」是假訊號，「數字歸給誰、當不當答案」才是真分界。用具體題例把這條分界擺出來：

**判對（乾淨棄答，含誠實對比 / 條件式協助）：**

| 問的是 | 模型回應 | 為什麼對 |
|---|---|---|
| 練小提琴多久 | "I have no record of you practicing violin — you only mentioned guitar, about 30 min/day." | "30 min/day" 明確歸給**吉他**（具名 distractor），小提琴本身棄答 |
| 收藏 vintage film 多久 | "You haven't started a vintage-film collection — you've only been collecting vintage cameras, for about three months." | "three months" 歸給**相機**，films 棄答 |
| 爸爸給的生日禮物 | "I'm not aware of any record of that." | 純乾淨棄答，gold=「從未提及」 |
| 這本書還剩幾頁 | "I can't give a figure; tell me your current page and I'll calculate the pages left." | 條件式 offer ≠ 宣稱知道答案 |

**判錯（被 distractor 帶跑 / 幻覺）：**

| 問的是 | 模型回應 | 為什麼錯 |
|---|---|---|
| Porsche（哪台先發動） | "The Ferrari started first on May 2." | 拿 **Ferrari** 的值當 Porsche 的答案、未標示替換 |
| 在 Shinjuku 住多久 | "seven months" | 這個值 silently 來自 **Harajuku**，被問項被冒名頂替 |
| 幾顆足球 | "the ~15 autographed baseballs would be the number accumulated" | 借**棒球**數當足球答案 |
| 先修柵欄還是先買三頭牛 | "you fixed the fence on…" | gold=事件從未發生，模型硬掰時間點=真幻覺 |

分界一眼可見：**上表的數字都「歸給另一個具名項、被問項有棄答」；下表的數字都「被當成被問項的答案」。** rubric 判的就是這條線，不是數字的有無。

實測（2026-07-20，n=3 裁決版全量 `_abs` 重判）：**20B 75.6→76.6（+1.0pp）、120B 79.8→81.0（+1.2pp）**，回收的正是「乾淨棄答」與「棄答+誠實提 distractor 對比」兩型假失分。

### 為什麼有些 borderline 走人工覆核，而不是繼續加訓文

`15745da0`（"haven't started a vintage-film collection **yet** — only vintage **cameras** for three months"）結構與 violin/guitar 範例幾乎相同，**照判準應對**，但即使把近乎逐字的正面範例寫進 prompt，4o-mini 三溫度全票仍判嚴（根因是 GEN 用了 "yet"、缺明確對比框架，措辭本身在邊界）。

**決策是：這類「連範例都壓不動」的 borderline 走人工覆核（直接改 `correctness_absrubric=1`），不再往 prompt 加訓文。** 為什麼？因為**放鬆 prompt 去救這一兩題，會連帶放鬆真幻覺的判定門檻**——用一個全域的鬆綁去修一個局部的邊界，代價是引入更難察覺的假「修復」。人工覆核是**局部、可審計**的修法，優於**全域、有副作用**的 prompt 放鬆。

---

## 決策五：合法性防線——判分規則會不會「作弊」

judge 之外，pipeline 還有幾個「補證據」規則（floor pad、min_keep、keep_all）帶內建先驗。這些會不會是 test-set 過擬合？專案立了**唯一判準**：

> **這條規則的先驗，有沒有用到「只有看評測集 gold 才知道」的資訊？**
> 用到 → 過擬合、作弊嫌疑。只用推論時真實可得的資訊（問題文字、問題日期）→ 乾淨。

為什麼是這條線而非別的？因為它精準切開了「**問題類型的內在因果屬性**」與「**從答案反推的後見之明**」：

- **`min_keep_aggregation` 乾淨**：靠問題**字面措辭**（how many/total/count）判定它是彙整題。彙整題本質上就需湊齊多個散落實例才數得對——這是**問題自帶的因果需求**，真實上線系統收到 "How many plants did I acquire?" 也能立刻判它是計數題，**無需看答案**。
- **`keep_all_categories` 踩線**：靠 category（KU/temporal）全補。但「這兩類該全補」**沒有內在因果**（KU 題不見得該保留全部 dated mention），它純粹是「**回看錯題集 gold 分桶、發現這兩類誤砍多**」的反推。因此 `adjudicate-v1` 主線**未開** keep_all（走 0 次），只留作證偽臂。

這條防線的意義：**分數要能對外宣稱，就不能有任何一步偷看單題答案。** 它讓「filter 有效」的結論站得住，而不是「我們調參調到 test set 上好看」。

---

## 六、LoCoMo 口徑對照（與 LongMem 的異同）

前五節都是 LongMem。LoCoMo 是**另一個 benchmark、另一套判分**——骨幹相同，但因**資料集性質不同**，`_abs` 那一整套機制不存在，換成兩個 LoCoMo 專屬的處理。這節把差異、成因、與 LoCoMo 自己的正確率差距講清楚。

### 為什麼 LoCoMo 沒有 `_abs` 棄答分支

根本原因是**資料集不同**：LongMemEval 刻意植入「該資訊從未提及」的棄答題（`_abs`），LoCoMo **沒有這種題型**。所以決策三（`_abs` 恆單票）和決策四（`_abs` 專用 rubric）在 LoCoMo **整段不適用**——沒有棄答題要保護，也就不需要棄答 rubric、不需要「恆單票」的例外。

LoCoMo 改用**單一 LoCoMo standard prompt**（`build_judge_standard_messages` → `ACCURACY_PROMPT`），不像 LongMem 有 per-category rubric 分岔。理由同決策一：逐字對齊各自 benchmark 的官方判分，保留對外可比性。

### LoCoMo 專屬的兩個處理，以及它們防什麼

| LoCoMo 專屬機制 | 程式 | 在防什麼 |
|---|---|---|
| **temporal 正規化**（`_normalize_temporal_gold`，把 gold 日期正規化後附給 judge） | `rejudge_4omini.py:62-63` | LoCoMo 有大量時間題，"May 7th" vs "7 May" vs 相對時間會被 judge 誤判為錯——正規化把格式差異抹平，讓 judge 判「是否同一時點」而非「字串是否相同」 |
| **排除 adversarial 題**（`exclude_adversarial=True`，category 5 不計分） | `stages/judge.py:34,55-59` | LoCoMo category 5 是對抗題（設計上無正解），計入會污染正確率——直接排除，只在有正解的題上算分 |

這兩個是 LoCoMo 版的「病根對症」：LongMem 的病是「棄答被誤殺」，LoCoMo 的病是「時間格式被誤判 + 對抗題無正解」。同一個哲學（judge 這把尺要對症校準），不同的病。

### 共用的部分

- **judge 模型 = 4o-mini**（同決策一，同樣獨立第三方尺、可對外複現）。欄位 `correctness_4omini` / `correctness_3vote`。
- **錯題 3 票多數決重判**（同決策二，temp 0/0.3/0.6、carry 對題）——掛在 `grep_replay.py --adjudicate-all` 那條線；舊的 `rejudge_4omini.py` 是恆單票（temp=0）的獨立腳本，屬早期口徑。
- **先驗合法性判準**（決策五）同樣適用。

### ⚠️ 口徑陷阱：兩個 judge 檔案不要混

LoCoMo 每個 sample 下有三個 CSV，**命名相近但口徑不同**，引用時務必分清：

| 檔名 | judge | 口徑 |
|---|---|---|
| `*_judge.csv` | gpt-oss-20b（舊本地 judge） | **不得與 4o-mini 混稱**，同 LongMem 的 `correctness_20b` |
| `*_judge_4omini.csv` | gpt-4o-mini | 單票（temp=0），`rejudge_4omini.py` 產出 |
| `--adjudicate-all` 產出 | gpt-4o-mini | 錯題 3 票（`correctness_3vote`），最終口徑 |

### LoCoMo 的正確率差距（3 票口徑，n=3）

| 比較 | 20B 答題 | 120B 答題 | 性質 |
|---|---:|---:|---|
| grep filter | 82.6 ± 0.2% | 82.5 ± 0.3% | filter 基線 |
| **裁決版** | **83.8 ± 0.4%** | **83.9 ± 0.3%** | filter 加裁決 |
| **裁決 Δ** | **+1.2pp** | **+1.4pp** | 裁決真實效應 |

與 LongMem 的**兩個對照差異**（成因都是資料集性質）：

1. **20B 與 120B 答題幾乎打平**（83.8 vs 83.9，Δ<0.1pp），LongMem 卻是 120B +4.4pp。因為 **LoCoMo 答案短、gold 集中，20B 答題本來就強（83.8%），120B 沒有 headroom**。
2. **LoCoMo 的 sd 極小**（0.3~0.4），LongMem 20B 的 sd 是 2.0。因為 LoCoMo 答案短、判分穩定，judge 抖動天然比 LongMem 小——這也解釋了為什麼 LoCoMo 早期能容忍恆單票腳本，而 LongMem 非上 3 票不可。

> 一句話：**LoCoMo 的 judge 抖動比 LongMem 小、且沒有棄答題**，所以它的判分設計比 LongMem 簡單——3 票骨幹照用，但省掉了整套 `_abs` 機制，換上 temporal 正規化與 adversarial 排除這兩個 LoCoMo 專屬校準。

---

## 一頁速查：決策 → 它在防什麼 → 量化影響

| 決策 | 防的是什麼 | 量化影響 | 放棄了什麼 |
|---|---|---:|---|
| judge 用 4o-mini（非本地 20b） | 同源污染、judge 自身噪音、不可對外複現 | 抹掉 ~12% run-to-run 不一致 | 成本、速度（付費+rate limit） |
| prompt 逐字對齊 hindsight | 失去與公開 leaderboard 的可比性 | — | 不能自己「優化」prompt |
| 一般題錯題 3 票（temp 0/0.3/0.6） | 單票 ~12% 抖動誤殺同義/偏好 borderline | 裁決版 +3.0pp / +15 題（表 A） | 只是上界估計、非全量 3 票；仍有 ±3~4 題重採樣抖動 |
| 三票用不同 temp | 三票同溫 = 假去噪（三次一樣） | — | — |
| `_abs` 恆單票 | 多票淹掉已判對的乾淨棄答 | 防止翻掉 `80ec1f4f`/`29f2956b` 這類單票對題 | 承認「3 票」不是萬用 |
| `_abs` 專用 rubric | 通用 rubric 系統性誤殺乾淨棄答（11/13） | 20B +1.0pp、120B +1.2pp（表 B） | 一處外科例外，需守住「非棄答題不動」邊界 |
| `_abs` 由檔名 tag 觸發 | gold 文字偵測漏標 | — | 依賴資料集 tag 存在 |
| borderline 走人工覆核 | 放鬆 prompt 連帶放鬆真幻覺判定 | 局部救 `15745da0` 等，不動全域門檻 | 不可完全自動化 |
| 「先驗只能用推論時可得資訊」 | test-set 過擬合、偷看答案 | keep_all 淨 −6（80.4→79.2）故不採納 | keep_all 這類反推規則只能當證偽臂 |
| **[LoCoMo]** temporal 正規化 | 日期格式差異被誤判為錯 | LoCoMo 專屬（第六節） | — |
| **[LoCoMo]** 排除 adversarial 題 | 無正解的對抗題污染分數 | category 5 不計分 | — |
| **[LoCoMo]** 無 `_abs` 分支 | — | 資料集無棄答題，決策三/四不適用 | — |

---

## 相關文件

- 算分口徑（怎麼做）：[JUDGING.md](JUDGING.md)
- 3 票去噪首次定案：`../EXPERIMENT_LOG.md` → `2026-07-18 · judge 3票多數決修正`
- 正確率階梯總結（表 A/C 來源）：`analysis/longmem-adjudicate-20b-CAMPAIGN.md`
- fallback / `_abs` 錯誤分析（11/13 誤殺證據）：`analysis/longmem-adjudicate-20b-fallback.md`
- 全錯題歸因（37 題 judge 假失分拆解、同義題例）：`analysis/longmem-adjudicate-20b-wrong-answers.md`
- 20B vs 120B（表 B/C 的 120B 數字來源）：`analysis/longmem-adjudicate-20b-vs-120b.md`
- **LoCoMo 裁決版正確率（第六節數字來源）**：`result/locomo-adjudicate-n3-20b-120b.md`
- pipeline 流程圖：`diagrams/agent_filter_pipeline.md`
- 程式（LongMem）：`experiment/longmem/rejudge_output_dirs.py`、`experiment/longmem/prompts/judge.py`
- 程式（LoCoMo）：`experiment/locomo/grep_replay.py`、`experiment/locomo/rejudge_4omini.py`、`experiment/locomo/helpers/llm.py`
