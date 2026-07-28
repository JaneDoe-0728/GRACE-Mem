# LongMem case study：hypothesis hint 為何能 boost —— trajectory 逐題剖析

日期：2026-07-21
分支：`codex/agent-filter-coverage-session`
資料：`experiment/longmem/output/fullfilt-20b-v2`（v2 精煉 hint，20b）vs `adjn3-lm20b-r1`（無 hint 對照，20b）
trace：`{cat}/_grep_agent_traces.jsonl` 的 `commands`（完整 agent trajectory）+ `hypothesis`（agent 自 emit 的答案）

> 目的：用 agent 的實際 trajectory，解釋「hypothesis hint 為何能把 LongMem 20b filter +4.9pp（75.5→80.4）」。
> 上位機制文：[`../hypothesis-hint-cross-model.md`](../hypothesis-hint-cross-model.md)。本文是它的 trajectory 底料。

---

## 0. 一句話

**boost 的來源不是「找到更多證據」，而是「agent 在搜尋過程中已經把聚合題算對了，把這個內心答案（hint）餵給答題模型，繞過答題模型自己會算錯的聚合步驟」。**

---

## 1. 先破一個誤解：hint boost 的是「字面/聚合正確率」，不是全面 accuracy

⚠️ **v1 vs v2 是兩件事，方向相反**（見上位文 §3.1）：

| hint 版本 | run | median hint 長度 | emit 率 | 效果 |
|---|---|---:|---:|---|
| **v1 冗長**（`HYPOTHESIS_LINE_BLOCK`） | `adjn3-lm20b-hyp` | **13 字元** | 80% | **−6.2pp** ❌（冗長句錨定） |
| **v2 精煉**（`HYPOTHESIS_LINE_BLOCK_V2`） | `fullfilt-20b-v2` | **2 字元** | 選擇性 emit | **+4.9pp** ✅ |

v1 的 hint 長這樣：`2 doctor's appointments in March`（完整句，把答題模型錨定到帶偏差的長句）。
v2 的 hint 長這樣：`8` / `Instant Pot` / `under my bed`（裸值）。

**「boost 這麼多」專指 v2 精煉裸值 hint。** 差別的真機制（上位文 §2）：v2 不是「產更精煉 hint」，而是**寧缺勿濫——把冗長/錯的毒 hint 換成 NONE**，只在 agent 真的推出乾淨裸值時才 emit。

### v2 hint 品質實測（`fullfilt-20b-v2`，100 題 emit）

agent 自 emit 的 hint 幾乎逐一命中 gold：

| 問題 | agent emit hint | gold |
|---|---|---|
| How many autographed baseballs…（3個月內） | `15` | 15 |
| Where do I keep my old sneakers? | `under my bed` | under my bed |
| What kitchen gadget did I invest in? | `Instant Pot` | Instant Pot |
| What vehicle model am I working on? | `Ford F‑150 pickup truck` | Ford F-150 pickup truck |

---

## 1.5 完整 prompt / input / output（三段管線的實際文字）

hint 機制是三段：**① agent 被要求 emit → ② harness 抽 HYPOTHESIS 行 → ③ 包成 NOTE 附到答題 context**。以下逐段給實際文字。

### ① emit prompt：注入 SYSTEM_PROMPT 的 `{hypothesis_line}` 槽

只在 `grep_agent_emit_hypothesis=1` 時注入（[`prompts.py`](../../../experiment/longmem/agent_filter/prompts.py) `_active_hypothesis_block()`，env `KG_HYP_PROMPT` 切版）。

**v1（`HYPOTHESIS_LINE_BLOCK`，預設）** —— 只要求「一句最佳答案」，無 few-shot、無禁冗長：

```
Before the FINAL line, add one line stating your best answer to the QUESTION
based on the evidence you found:
  HYPOTHESIS: <your answer as a short phrase, or NONE if you cannot determine it>
This is your own tentative conclusion; the FINAL sids remain the evidence.
```

**v2（`HYPOTHESIS_LINE_BLOCK_V2`，`KG_HYP_PROMPT=v2`）** —— 明令「只給裸值、禁整句」+ 5 個 few-shot：

```
Before the FINAL line, add one line with your best answer to the QUESTION.
Give ONLY the bare answer value — the exact word, name, number, date, or
duration that answers the question. Do NOT restate the question, do NOT
explain, do NOT write a full sentence. Match the form the question asks for.
Examples:
  Q: How many appointments in March?      HYPOTHESIS: 2
  Q: How much per mug?                     HYPOTHESIS: $12
  Q: How long using the Fitbit?            HYPOTHESIS: 9 months
  Q: Where do I keep my sneakers?          HYPOTHESIS: under my bed
  Q: What was the 7th job listed?          HYPOTHESIS: Transcriptionist
If you truly cannot determine it, write: HYPOTHESIS: NONE
This is your own tentative conclusion; the FINAL sids remain the evidence.
```

**唯一差別**：v2 加了「bare answer value / no full sentence」硬約束 + few-shot 錨定裸值形態。這一段 prompt 就是 −6.2pp 翻成 +4.9pp 的全部變因。

### ② agent 實際 emit 的 raw output（FINAL 那步的 reply）

抽取邏輯（[`harness.py:729-733`](../../../experiment/longmem/agent_filter/harness.py#L729)）：正則 `HYPOTHESIS\s*[::]\s*([^\n]+)` 只抓到行尾，再用 `\bFINAL\b` 截斷（防貪婪吞掉同行 sids）。

**v2 raw output（`fullfilt-20b-v2`，精煉裸值）**：

| id | 問題 | RAW FINAL reply | 抽出 hypothesis |
|---|---|---|---|
| `01493427` | 加了幾張明信片 | `HYPOTHESIS: 25\nFINAL answer_a7b44747_1:8:u answer_a7b44747_2:8:u` | `25` |
| `07741c44` | 舊球鞋原本放哪 | `HYPOTHESIS: under my bed\nFINAL answer_7e9ad7b4_1:4:u` | `under my bed` |
| `0977f2af` | 買了什麼廚房家電 | `HYPOTHESIS: Instant Pot  \nFINAL answer_3bf5b73b_1:4:u` | `Instant Pot` |

agent 在 FINAL 的**同一則訊息**先寫 `HYPOTHESIS:` 一行、再寫 `FINAL <sids>` 一行 —— hint 與證據選擇一次產出、同模型自洽、零外部依賴。

> 附：gpt-oss-20b 的 reply 常帶 harmony channel token（`<|channel|>commentary to=GREP <|constrain|>json<|message|>...`），20b 抽取端須 `_clean_hint` 剝除這些 token + U+202F narrow-space 正規化（見上位文 §4.1）。

### ③ 同一題 v1 vs v2 的 hint 對比（`07741c45`，「現在球鞋放哪」）

同一 agent、同一題，只換 emit prompt：

| 版本 | agent emit hypothesis | 附給答題模型的形態 |
|---|---|---|
| **v1** | `Under the bed (May 25) and in a shoe rack (May 29)` | 帶時間、帶舊值的整句 → 錨定答題模型輸出兩個地點 |
| **v2** | `in a shoe rack` | 裸值（最新值）→ 錨定答題模型只答最新 |

gold = `in a shoe rack in my closet`。v1 的整句 hint 把「舊值 under the bed」也帶進答題模型視野（knowledge_update 題最忌拿舊值）；v2 只給最新裸值。**這就是 v1 為何在 knowledge_update/multi_session 重災的縮影**。

更多 v1 冗長 hint 實例（都被答題模型當長句錨定）：

| id | gold | v1 冗長 hypothesis |
|---|---|---|
| `08e075c7` | 9 months | `As of 2023/06/18 you had been using the Fitbit Charge 3 for 6 months; by 2023/09/02 that duration was 9 months.` |
| `031748ae` | （帶新職位人數） | `I led 4 engineers when I first became Senior Software Engineer, and I now lead 5 engineers.` |
| `2133c1b5` | 3 months | `I have been living in my current apartment in Harajuku for 3 months as of the latest update` |

### ③ input：hint 怎麼包進答題 context

抽出的 hypothesis 被 [`replay_run.py:99-104`](../../../experiment/longmem/agent_filter/replay_run.py#L99) 包成 NOTE，**附在答題 context 尾端**（evidence 之後）：

```python
context = context + (
    "\n\nNOTE: A preliminary evidence-search analysis tentatively concluded "
    f"the answer may be: \"{hyp}\". Treat this only as a hint — verify it against "
    "the evidence above; if the evidence contradicts it, trust the evidence."
)
```

答題模型收到的**完整 input 尾端**（以 `b5ef892d` 露營題為例，v2）：

```
=== Entities ===
- 3‑day solo camping trip to Big Sur (Event): …
- 10-day trek (Activity): An outdoor backpacking trip lasting ten days … ← 干擾項
- 7‑day family road trip in February (Event): …
…（其餘 entity / 原始 turn）…

NOTE: A preliminary evidence-search analysis tentatively concluded the answer
may be: "8". Treat this only as a hint — verify it against the evidence above;
if the evidence contradicts it, trust the evidence.
```

**注意 NOTE 明示「與證據矛盾時以證據為準」** —— hint 是防禦性導航，不是強制覆寫。答題模型仍讀完整 evidence，只是有了 `8` 這個錨點就不再把 `10-day trek` 誤加進總和（§3 詳解）。

---

## 2. hint 救場的 10 題全是聚合/計算題

`fullfilt-20b-v2`（有 hint）vs `adjn3-lm20b-r1`（無 hint），逐題翻盤中 **hint 救場（無hint錯→v2hint對）= 10 題**，其中主體是「答案要從多段證據**算出來**」的題：

| id | 問題 | gold | 無 hint 答（錯） | hint | v2hint 答（對） |
|---|---|---|---|---|---|
| `b5ef892d` | 今年在美國露營幾天 | 8 | **18**（3+5+**10**，多加無關 trek） | `8` | 8（3+5） |
| `10d9b85a` | 四月參加工作坊/講座幾天 | 3 | **5** | `3` | 3 |
| `aae3761f` | 三個公路旅行目的地共開幾小時 | 15 | **17–18** | `15` | 15 |
| `0ddfec37` | 前三月加入幾顆簽名棒球 | 15 | 15 顆**足球**（抽錯品項） | `15` | 15 棒球 |
| `09ba9854` | 搭火車省多少錢 | $50 | 沒給數字 | `50` | ~$50 |
| `b86304ba` | 日落畫值我付的幾倍 | 三倍 | 「資訊不足」棄答 | `triple what I paid` | 三倍 |

**共同結構**：答題模型單獨面對發散的 entity summary 時，聚合步驟出錯——多加一個無關項、算錯倍數、或抽錯鄰近品項。hint 把 agent 內心已算對的答案直接餵進去，繞過這個易錯步驟。

---

## 3. 決定性 trajectory：`b5ef892d`（露營 8 天）

### 3.1 兩版看到**同一份** entity summary（含干擾項）

context 裡有三個帶天數的 event：

```
- 3‑day solo camping trip to Big Sur   ← 真（3）
- 2023-03-29 camping trip to Yellowstone（後文另述 5 天）  ← 真（5）
- 10-day trek（backpacking，用淨水器那段）   ← 干擾！這是「計畫用」的 trek，非今年美國露營
- 7‑day family road trip in February（Utah 公路旅行）   ← 干擾（road trip 非 camping）
```

gold = 3 + 5 = **8**。干擾項 `10-day trek`、`7-day road trip` 都不該算。

### 3.2 無 hint 版（`adjn3-lm20b-r1`）→ 答 18 ❌

> You spent a total of **18 days** … (3 days at Big Sur, 5 days at Yellowstone, and **10 days on your backpacking trek**).

答題模型看到 `10-day trek` 也帶「camping/backpacking」字樣 → **誤加進總和**（3+5+10=18）。這是聚合題的典型失敗：發散 summary 裡的干擾項被一起加總。

### 3.3 v2 hint 版（`fullfilt-20b-v2`）→ 答 8 ✅

agent 的 trajectory（`commands`）：

```
[step 0] GREP "camping trip"         ← 一次搜尋定位露營證據
[step 1] HYPOTHESIS: 8               ← agent 在 reasoning channel 內部已算對：只加 3+5，排除 trek
         FINAL answer_a8b4290f_1:2:u answer_a8b4290f_2:2:u
```

agent 只 FINAL 了 2 條 seed（Big Sur + Yellowstone 的那兩輪），**它在內心就判定 trek 不算、算出 8**。這個 `8` 被當 hint 餵給答題模型：

> You spent **8 days** on camping trips … (3 days at Big Sur + 5 days at Yellowstone).

**同一份含干擾的 context，答題模型這次沒被 `10-day trek` 帶偏——因為 hint `8` 錨定了正確聚合。**

### 3.4 機制拆解

| 環節 | 無 hint | v2 hint |
|---|---|---|
| context（entity summary） | 相同（含 10-day trek 干擾） | 相同 |
| 聚合判斷「哪些天數該加」 | **答題模型自己判** → 誤加 trek → 18 | **agent 搜尋時已判好** → hint=8 |
| 答題模型角色 | 讀 summary + 自己聚合（易錯） | 讀 summary + **對齊 hint**（穩） |

**關鍵洞察**：agent 在 grep 搜尋回合裡「讀原始 turn 上下文」比答題模型「讀壓縮後 entity summary」更能分辨哪個 event 該算。agent 把這個判斷結果濃縮成一個裸值 hint，等於**把聚合決策從易錯的答題端移到證據充分的搜尋端**。

---

## 4. 為什麼「boost 落在 LongMem 難題」（對照 LoCoMo +0.00pp）

上位文 §5 已量化：LongMem hint 有效 = **落在難題 + 抽得準**；LoCoMo 兩者皆不成立。本文 trajectory 給出 LongMem 側的微觀理由：

1. **LongMem 答題 context 是壓縮 entity summary**（發散、含跨 event 干擾項）→ 聚合題答題模型易被干擾項帶偏（§3.2 的 `10-day trek`）。
2. **agent 搜尋端看得到原始 turn**（`GREP` 回傳原文），對「哪段該算」判斷更準 → hint 品質高（§1 的逐一命中）。
3. 兩者疊加：**hint 把「充分證據下的聚合判斷」從弱答題端搬到強搜尋端**。這在「答案=多段相加且有干擾項」的 LongMem 聚合題上收益最大。

---

## 6. filter_fetch 版對照：同機制複現 + 一個新模式（2026-07-21）

本文正文用的是 **filter 版**（`fullfilt-20b-v2`：agent 只篩不搜）。後續補跑 **filter_fetch 版**（`ff-hint-20b`：agent 會 fetch 補搜，`--mode filter_fetch --emit-hypothesis`），ctrl 用純 filter_fetch 無 hint 的 `grep-lm20b-r1`，同口徑（`correctness_new`）。逐題救場模式**與本文高度一致，且多出一個 filter_fetch 特有模式**。

### 6.1 核心聚合機制完全複現

| 維度 | 本文（filter，`fullfilt-20b-v2`） | filter_fetch（`ff-hint-20b`，實測） |
|---|---|---|
| 救場題全是聚合/數值題 | 100%（10/10） | **100%（10/10）** |
| 主模式：grep 聚合算錯 → hint 錨定糾正 | 10/10 | **8/10 同模式** |
| accuracy Δ | +4.9pp（vs 純 filter ctrl） | **+3.0pp**（vs grep filter_fetch ctrl，77.9→80.9） |

**同一題在兩版都複現**——`10d9b85a`「四月工作坊幾天」：兩版 grep 都答 **5**、hint 都 `3` → 對。其餘複現例：
- `Starbucks Gold 幾顆星`：grep 幻覺 **300 stars** → hint `120` → 對。
- `charity events 幾個`：grep 數成 **3** → hint 拉回。

**證明本文 §3「hint 把聚合決策從易錯答題端搬到搜尋端」的機制在 filter_fetch 上同樣成立**，不是 filter 特有。

### 6.2 filter_fetch 新增模式：「棄答 → 補答」（2/10）

本文的 filter 版救場全是「grep 答錯值 → hint 糾正」。**filter_fetch 多出一類**：grep 答題模型**棄答/找不到**，但 hint 補上答案：

| id | 問題 | grep（棄答） | hint | hint 版（對） |
|---|---|---|---|---|
| — | 5 天旅行打包幾件襯衫 | 「沒有記錄」 | `7` | 7 shirts ✅ |
| — | 幾天前 launch 網站 | 「資訊不可用」 | `19` | 19 days ✅ |

**為何 filter_fetch 才有這模式**：filter_fetch 的 agent 會 **fetch 補搜**（filter 只篩不搜），常能搜到答題模型在壓縮 entity summary 裡找不到的證據，emit 成 hint。這是 filter_fetch 獨有的價值——**agent 補搜到的資訊透過 hint 傳給答題模型，救回答題模型本會棄答的題**。對應救場拆解：**8 題聚合糾正 + 2 題棄答補答 = 10**。

### 6.3 耐噪佐證：hint 抽錯仍不盲從

filter_fetch 救場題裡有 2 題 hint **抽錯**（`clothing items` hint=`2`/gold 3、`binoculars` hint=`3 weeks`/gold 2 weeks），但答題模型**沒盲從錯 hint**、仍答對。這印證本文 §1.5「NOTE 明示與證據矛盾時以證據為準」的防禦性設計有效——hint 是導航錨點，非強制覆寫。

### 6.4 判讀

filter_fetch 版**強化並擴展**了本文結論：
1. **核心聚合機制跨 mode 穩健**（filter +4.9 / filter_fetch +3.0，同為正、救場全聚合題）。
2. **filter_fetch 多一條增益路徑**：fetch 補搜 → hint 補答，救棄答題（filter 無此路徑，因 agent 不補搜）。
3. **hint 的防禦性設計在兩版都成立**（抽錯 hint 不盲從）。

> ⚠️ 口徑：本節 filter_fetch 為 `correctness_new` 單票、vs `grep-lm20b-r1` 純 filter_fetch ctrl；本文正文 filter 版為 3vote、vs `adjn3` base（已知 base 錯配，見 [`hint-20b-vs-120b.md`](hint-20b-vs-120b.md) 更正節）。兩版 Δ 不同口徑，但**救場模式（100% 聚合題）跨口徑一致**。4b/27b filter_fetch 對照跑中，齊後補跨模型 trajectory。

---

## 5. 結論

1. **v2 hint 的 +4.9pp 不是 recall（找到更多證據），是把 agent 內心已算對的聚合答案餵給答題模型**，繞過答題模型在發散 summary 上易錯的聚合步驟。
2. **救場題 100% 是聚合/計算題**（露營天數、開車時數、倍數、品項）——hint 錨定正確裸值，壓過干擾項導致的錯誤加總（18→8、5→3、17-18→15）。
3. **v1 冗長 hint 反而 −6.2pp**：長句錨定把偏差一起帶進去；v2「寧缺勿濫、只 emit 乾淨裸值」才是 boost 的必要條件。
4. **與 case-study §一 的 20B/120B 洗牌互補**：那份說「聚合題是答題模型的弱點」；本份說「hint 正是靠替聚合題預先算好答案來補這個弱點」。

## 相關文件
- hint 機制主文：[`../hypothesis-hint-cross-model.md`](../hypothesis-hint-cross-model.md)
- 20B vs 120B 逐題對照：[`case-study-20b-vs-120b-hitmiss.md`](case-study-20b-vs-120b-hitmiss.md)
- prompt 實作：[`../../../experiment/longmem/agent_filter/prompts.py`](../../../experiment/longmem/agent_filter/prompts.py)（`HYPOTHESIS_LINE_BLOCK_V2`）
