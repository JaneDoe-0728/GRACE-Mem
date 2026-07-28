# LongMem hypothesis hint：20B vs 120B 的差異剖析

日期：2026-07-21
分支：`codex/agent-filter-coverage-session`
資料：v2 精煉 hint `fullfilt-20b-v2` / `fullfilt-120b-v2`，無 hint 對照 `adjn3-lm20b-r1` / `adjn3-lm120b-r1`
判分：`correctness_3vote`（一般題）+ abs rubric，469 題對齊

> 專門對照「hypothesis hint 這條線上，20B 與 120B 答題模型的差異」。
> 上位機制文：[`../hypothesis-hint-cross-model.md`](../hypothesis-hint-cross-model.md)；trajectory 底料：[`case-study-hypothesis-hint-trajectory.md`](case-study-hypothesis-hint-trajectory.md)。

---

## ⭐ 更正（2026-07-21）：filter_fetch 同口徑對照 —— 120B 是中性 −0.2，非翻負 −1.9

> **本文原用 base 錯配的對照（filter+hint 臂 vs filter_fetch+裁決 base `adjn3-lm*-r1`），得出「120B −1.9pp 翻負」的結論。此結論作廢。**
> 錯配根因：hint 臂是 **filter 模式**（`fullfilt-*-v2`，見 §3 trace `mode=filter`），卻拿 **filter_fetch+裁決** 的 `adjn3` 當 ctrl——兩個 pipeline 不同，差異被誤記成「hint 脆弱」。

**⚠️ 重要：`ff-hint-*` 是三疊加（filter_fetch + 裁決 + hint），非「無裁決」。** 啟動時未加 `--no-adjudicate`，config 預設 `grep_agent_adjudicate=1` → 裁決開（20b 251/495、120b 262/495 題有撿回）。故 hint 的 Δ 取決於「跟哪種 ctrl 比」，**兩種都報**：

| 模型 | ff-hint（三疊加） | **vs grep（無裁決）** | **vs adjn3（裁決無hint）** | emit | 命中 gold |
|---|---:|---:|---:|---:|---:|
| **20b** | 80.9% | 77.9 → **+3.0pp**（疊加總效果） | 76.8 → **+4.1pp**（hint 淨效果） | 102 | 60% |
| **120b** | 79.6% | 79.8 → **−0.2pp** | 80.5 → **−0.9pp** | 98 | 60% |
| 4b | 跑中 | — | — | — | — |
| **27b** ¹ | 82.5% | 81.9 → **+0.5pp** | 待算 | 293 | **83%** |

- **vs grep（無裁決）**：hint **+ 裁決**的合併效果（ctrl 基底最低）。
- **vs adjn3（裁決無hint）**：**hint 的淨效果**（只差 hint 一個變因，最純；對應上位文 §4「裁決基底上加 hint」）。

¹ 27b = **自抽快版**（`ff-hint-27b-fast`：從 `qwen27b-grep-r1` reply 正文自抽 hint + 重答，非 agent 現場 emit；因 27b reasoning channel 空）；**382/480 部分配對**（`.52:8000` 27b endpoint 推理慢、中途斷，缺 98 題待補）。方向初步。

**修正後的真相（以 hint 淨效果 = vs adjn3 裁決基底為準）：**
1. **120B hint 淨效果 −0.9pp（微負），非「有害翻負 −1.9」**。原 −1.9 是 base 錯配假象（filter 臂 vs filter_fetch+裁決 base）。裁決基底上加 hint：20b +4.1 / 120b −0.9，對應上位文 §4「LongMem 裁決+hint 微正/微負」定律。
2. **20B hint 淨效果 +4.1pp 穩健**（裁決基底 76.8→80.9）。
3. **20B/120B 差異是「有益 vs 微負」**——120B 裁決基底已高（80.5%），hint 空間被壓到微負；20B 基底低（76.8%）仍有大空間。
4. **hint 品質 20b(60%)≈120b(60%)**，filter_fetch 下兩者相近。**「120b agent hint 更差 51%」是舊 filter 版數字，結論減弱**。
5. **⭐ 27b 揭示核心律：hint 品質高 ≠ 增益大**。27b hint 命中 gold **83%（全家族最高）**，但 vs grep 只 **+0.5pp**——因 **27b 答題端最強、base 最高（81.9%），hint 空間被壓到接近零**。與 120B 同機制：**強答題模型下 hint 增益趨零，與 hint 品質無關**。決定增益的是**答題端有無空間**（base 高低），非 hint 準不準。
6. **三疊加 vs 疊加拆分**：`ff-hint` = filter_fetch + 裁決 + hint。vs grep（無裁決）= 裁決+hint 合併（20b +3.0）；vs adjn3（裁決）= hint 淨（20b +4.1）。**hint 淨效果比合併還高**，因裁決在 20b 上本身微負（見上位文 grep-vs-adjudicate §二 LongMem 20B 裁決 −2.7），hint 補回並超越。

> 下方 §1–§4 為**舊 filter 版錯配對照的存檔**（保留作方法學教訓：base 必須同 mode）。數字（−1.9pp、120b 品質 51%）已被本節 filter_fetch 同口徑重跑取代。

---

## 0. 一句話（舊版，已被上方更正取代）

**hint 對 20B 有益（+2.8pp）、對 120B有害（−1.9pp）——分歧的根因不是「120B 消化不了 hint」，而是「120B 的 agent hint 品質更差」（命中 gold 只 51% vs 20B 65%），加上 120B base 本就高、邊際空間小，錯 hint 的傷害壓過收益。**

---

## 1. 主對照表（3vote 口徑）

| 模型 | base 無 hint | v2 hint | **Δ** | emit 題數 | hint 救場 | hint 弄壞 | 淨 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **20B** | 77.6% | **80.4%** | **+2.8pp** ✅ | 100 | 10 | 6 | **+4** |
| **120B** | 81.7% | **79.7%** | **−1.9pp** ❌ | 74 | 7 | 9 | **−2** |

> ⚠️ 與上位文 §3 表（20b +4.9 / 120b +4.2 皆正）的差異：那是 **4o-mini `correctness_new` 單票 + 對「純 filter ctrl」** 口徑；本表是 **3vote + 對「adjn3 base」**。口徑不同、base 不同，故 120B 由正翻負。**hint 的效果高度依賴 judge 口徑**（見上位文 §3 vs `grep-vs-adjudicate-cross-model.md` §四「指標依賴律」）——這本身就是 20B/120B 差異的一部分：120B 的增益脆弱到換口徑就翻負，20B 穩健為正。

---

## 2. 四個差異維度

### 2.1 hint 品質：20B 反而更準（65% vs 51%）

emit hint 與 gold 一致率（norm 後子字串比對）：

| 模型 | emit 題數 | hint 命中 gold | 命中率 |
|---|---:|---:|---:|
| **20B** | 100 | 65 | **65%** |
| **120B** | 74 | 38 | **51%** |

**反直覺**：更強的 120B agent，emit 的 hint 品質**更差**。原因見 §2.3——120B 過度自信、搜尋更少就 early FINAL，在難題上把「自己推的錯答案」當 hint。

### 2.2 emit 傾向：120B 更保守（74 vs 100）

120B 只在 74 題 emit HYPOTHESIS（20B 100 題）。120B 更常**直接 FINAL 不寫 hint**——它對「自己答得出」的題不 emit。但這個自我篩選**沒把難題篩乾淨**：120B emit 題在無 hint base 的正確率 74%（20B 80%），代表 120B 確實把 hint 導向更難的題，但那些難題正是它 hint 也抽不準的題。

### 2.3 弄壞題拆解：120B 的傷害來自「錯 hint」

弄壞題（base 對 → hint 弄壞）裡，hint 本身是對是錯：

| 模型 | 弄壞題 | **hint 本身就錯** | hint 對但仍弄壞(judge口徑/長句) |
|---|---:|---:|---:|
| **20B** | 6 | 3（含 2 個 `NONE then…` 抽取殘渣） | 3 |
| **120B** | 9 | **6（67%）** | 3 |

**120B 弄壞的主體是錯 hint**（6/9）——實例：

| id | 問題 | 120B agent hint | gold |
|---|---|---|---|
| `852ce960` | Wells Fargo 房貸預核額度 | `$350,000` | **$400,000** |
| `2698e78f` 類 | 看醫生頻率 | `twice a week` | **Three times a week** |
| `5c40ec5b` 類 | 完成幾門課/次數 | `4` | **six** |

錯 hint 把答題模型從「本來會答對」帶偏。20B 的 3 個「錯 hint」有 2 個其實是抽取殘渣（`NONE then…`、`NONE. But we need to output`）——20B 想寫 NONE 卻被正則抽出雜訊，屬 harness bug 非推理錯。

### 2.4 決定性 trajectory：同題 `852ce960`（房貸額度）

同一題，兩模型行為天差地別：

```
120B (fullfilt-120b-v2):
  GREP Wells Fargo pre-approved   ← 只搜 1 次
  HYPOTHESIS: $350,000            ← 推出錯答案（gold $400,000）
  FINAL answer_3a6f1e82_1:4:u     ← early FINAL，只留 1 條 seed
  → 錯 hint 餵給答題模型 → 弄壞 ❌

20B (fullfilt-20b-v2):
  3 個 command（搜得更徹底）
  hypothesis = None               ← 不 emit（沒把握就不寫）
  → 答題模型自推 → 對 ✅
```

**機制**：120B 過度自信 → 搜尋更少（early FINAL）→ 在難題上推出錯答案 → 又因自信而 emit → 錯 hint 反噬。20B 在同題上**搜得更徹底且知道自己沒把握（不 emit）**，反而讓答題模型自推答對。**「120B agent 更強」在 hint 這條線上是負資產：它更敢在證據不足時給出並 emit 錯答案。**

---

## 3. 為什麼 20B 是 hint 的最佳受益者

三個條件在 20B 上同時成立、在 120B 上部分失效（呼應上位文 §0「hint 有效三條件」）：

| 條件 | 20B | 120B |
|---|---|---|
| ① **base 低、有空間** | 77.6%，聚合題弱 → hint 補得到 | 81.7%，聚合題已較強 → 空間小 |
| ② **hint 落難題** | 是（emit 題 base 80%） | 是（74%，甚至更難） |
| ③ **hint 抽得準** | **65% 命中** ✅ | **51% 命中** ❌ |

20B：①②③ 全中 → **+2.8pp**。
120B：① 空間被高 base 吃掉、③ hint 品質不足 → 錯 hint 傷害 > 收益 → **−1.9pp**。

**核心律**：hint 是「用 agent 搜尋端的判斷補答題端的弱點」。這只在「**答題端弱（20B）× 搜尋端 hint 準**」時淨賺。120B 答題端已強（弱點少、補的空間小），而它的搜尋端 hint 反而**更不準**（過度自信 early FINAL）——兩頭都對它不利。

---

## 3.5 跨家族對照：Qwen 4b / 27b 的 hint 處理方式與結果

把 gpt-oss 的 20b/120b 放到更大的家族座標裡（皆 v2 精煉 prompt、純 filter 基底、v2hint vs ctrl 配對）。

### 3.5.1 hint 形態與品質（四模型全表）

| 模型 | emit 率 | median hint 長度 | **冗長率(>40字)** | **hint 命中 gold** | hint 形態 |
|---|---:|---:|---:|---:|---|
| gpt-oss-20b | 21% | 2 | 4% | 65% | 極精煉裸值 |
| gpt-oss-120b | 16% | 3 | 4% | 51% | 精煉但推理常錯 |
| **Qwen 4b** | **88%** | **20** | **29%** | 58% | **不守裸值指令、愛寫整句** |
| **Qwen 27b** | **91%** | 7 | 6% | **75%** | 精煉且推理最準 |

**兩條正交的軸：**

1. **emit 傾向（守不守「選擇性 emit」）**：gpt-oss 選擇性 emit（16–21%，沒把握就不寫）；**Qwen 兩者都幾乎逢題必 emit（88–91%）**——Qwen 家族傾向對每題都給假說，不做自我篩選。
2. **格式服從（守不守「裸值、禁整句」）**：gpt-oss 與 27b 守（median 2–7、冗長 <6%）；**4b 不守**——median 20、29% 是 >40 字的整句，實例：
   - `25 new postcards (latest update on 2023/11/30) and 17 new postcards (previous count on 2023/08/11).`
   - `5 engineers (at 2023/10/24) and 4 engineers (at 2023/05/11)`

   4b 把 v2 few-shot 當耳邊風，退化成 v1 式的冗長帶時間戳整句（正是上位文 §2 定調的「毒 hint」形態）。

### 3.5.2 結果對照（accuracy，全部 3vote 同口徑）

| 模型 | ctrl 無 hint | v2 hint | **Δ** | hint 救場 | 弄壞 | 淨 | 配對 n |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | 77.6% | 80.4% | **+2.8pp** ✅ | 10 | 6 | +4 | 469 |
| gpt-oss-120b | 81.7% | 79.7% | **−1.9pp** ❌ | 7 | 9 | −2 | 469 |
| **Qwen 4b** | 74.4% | **78.6%** | **+4.2pp** ✅ | 30 | 14 | +16 | 332 |
| **Qwen 27b** | — | — | **無 accuracy** | — | — | — | — |

**全表已統一 3vote 口徑**（`correctness_3vote`，temp 0/0.3/0.6 多數決，判分模型皆 gpt-4o-mini）。4b 用 3vote 重判後（`qwen4b-filt-{ctrl,v2hint}`，2026-07-21 補跑），Δ **+4.2pp** 與單票同方向同幅度——**穩健，不像 120b 換口徑就翻負**：

| 4b 口徑 | ctrl | v2hint | Δ | 配對 n |
|---|---:|---:|---:|---:|
| 4o-mini 單票（`correctness_new`） | 78.6% | 82.8% | +4.2pp | 262 |
| **3vote（`correctness_3vote`）** | 74.4% | 78.6% | **+4.2pp** | **332** |

3vote 把 ctrl 與 hint 兩臂**一起**往下拉 ~4pp（單票偏寬鬆），但 **Δ 幅度不變**——證明 4b 的 hint 增益不是單票寬鬆判出來的假象，換嚴口徑照樣 +4.2pp。這與 120b 形成鮮明對照：**120b 的 +4.2（單票）→ −1.9（3vote）翻負，4b 的 +4.2 兩口徑穩定**。

> ⚠️ **27b 目前無 accuracy**：`qwen27b-filt-ctrl` / `qwen27b-filt-v2hint` 兩 run 的 correctness 欄全空（未跑判分），本文只能報它的 hint 品質（trace 抽），不能報增益。這與上位文 §3「Qwen 27b 跑中」一致。
> ℹ️ 配對 n 差異：4b（332）< gpt-oss（469）因 `qwen4b-filt-ctrl` 只跑完部分類別（`knowledge_update` ctrl 為 0 檔），配對取兩 run 交集。

### 3.5.3 判讀：三條規律在家族層級的體現

1. **「emit 率高 ≠ hint 有效」**：Qwen 逢題必 emit（88–91%），但 4b 因格式不守（冗長 29%）把品質稀釋到 58%；gpt-oss 選擇性 emit（16–21%）反而 20b 命中 65%。**選擇性 emit + 短裸值** 才是品質關鍵，不是 emit 量。
2. **「模型越大、hint 品質未必越高——但格式服從度越高」**：27b 命中 75%（家族最高）且守裸值（冗長 6%），4b 命中 58% 且冗長 29%。同 Qwen 家族內，大模型主要贏在**能遵守 v2 裸值指令**（4b 讀不進 few-shot）。這與 gpt-oss 內部「120b 反而比 20b 差」不衝突——那裡差在**推理正確率**（120b early FINAL 推錯），這裡差在**格式服從**（4b 不守裸值）。兩種失敗模式獨立。
3. **4b 是 v2 hint 的家族最大受益者（3vote +4.2pp、淨 +16）**：即便它 hint 冗長、命中僅 59%（3vote 配對），因為 **4b 答題端最弱（聚合題弱點最多、3vote base 僅 74.4%，家族最低）**，hint 補的空間最大——呼應 §3「hint 有效 = 答題端弱 × hint 夠用」，4b 答題端最弱這一項壓過了它 hint 品質的不足。且 **4b 的增益跨口徑穩健（單票/3vote 皆 +4.2）**，與 120b 的口徑脆弱性（+4.2→−1.9）恰成兩極。

---

## 4. 結論

1. **hint 在 20B/120B 上方向相反**：20B +2.8pp、120B −1.9pp（3vote 口徑）。
2. **分歧根因是 hint 品質**：20B hint 命中 gold 65%、120B 只 51%。更強的 agent hint 品質**更差**——因為 120B 過度自信、搜尋更少就 early FINAL 並 emit，在難題上把錯答案當 hint（弄壞題 67% 是錯 hint）。
3. **120B base 高壓縮了 hint 的邊際空間**（81.7% vs 20B 77.6%），錯 hint 的傷害更容易壓過稀薄的收益。
4. **hint 是 20B-favoring lever**：它補的是「弱答題模型的聚合弱點」，而補的來源要求「搜尋端 hint 準」。20B 兩條件皆滿足，120B 兩條件皆偏弱 → hint 是 20B 專屬的增益、對 120B 是淨傷害。
5. ⚠️ **效果對 judge 口徑敏感**：120B 在 4o-mini 單票口徑報 +4.2pp、在 3vote 口徑翻 −1.9pp——120B 的 hint 增益脆弱到換口徑就消失，這脆弱性本身就是它與 20B 的關鍵差異。

## 相關文件
- hint 機制主文：[`../hypothesis-hint-cross-model.md`](../hypothesis-hint-cross-model.md)
- hint trajectory 剖析：[`case-study-hypothesis-hint-trajectory.md`](case-study-hypothesis-hint-trajectory.md)
- 20B vs 120B 通用逐題對照：[`case-study-20b-vs-120b-hitmiss.md`](case-study-20b-vs-120b-hitmiss.md)
