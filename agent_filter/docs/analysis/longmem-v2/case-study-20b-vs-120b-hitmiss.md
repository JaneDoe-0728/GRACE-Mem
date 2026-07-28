# LongMem case study：20B vs 120B 答題模型的 hit/miss 逐題差異

日期：2026-07-21
分支：`codex/agent-filter-coverage-session`
資料：`experiment/longmem/output/grep-lm20b-r1` vs `grep-lm120b-r1`（grep filter 版，499 題對齊，judge=`correctness_3vote`）

> 目的：對照同一 pipeline 下換答題模型（本地 gpt-oss-20b → 120b）**逐題**哪些進步、哪些退步，找出增益/退步的機制來源。這是 [`../grep-vs-adjudicate-cross-model.md`](../grep-vs-adjudicate-cross-model.md) §二「裁決效果 × 答題模型強度」的逐題底料。

---

## 一、總覽：淨 +9 題 = 進步 49 − 退步 40

| | 20B 對 | 20B 錯 |
|---|---:|---:|
| **120B 對** | 348（both_right） | **49（120B 進步）** |
| **120B 錯** | **40（120B 退步）** | 62（both_wrong） |

- 整體 accuracy：**20B 77.8% → 120B 79.6%（+1.8pp）**，淨 +9 題。
- **翻盤題高達 89 題（17.8%）**——換模型不是均勻抬升，而是大量互相抵消的 swap。這解釋了為何 §二「相依對照」下 20B↔120B 的裁決 Δ 幅度都在 ±3pp 內：兩模型能力接近，差異是**題型敏感的洗牌**，非全面碾壓。

### ⚠️ 關鍵方法學caveat：這不是純答題模型對照

**499 題沒有一題的 retrieved context 完全相同**（median |Δlen|≈1200 字元）。因為 grep filter 版**每個答題模型跑自己的 agent**（20B agent / 120B agent 各自搜尋、各自 FINAL）。所以本文的 hit/miss 差異是**「agent 檢索 + 答題」端到端**的，不是隔離的答題能力。

- 要純隔離「答題模型」須固定 context（如 §二 4o-mini 用 120b retrieve 的 16-seed 定樁）。
- 本文價值在**端到端逐題歸因**：把每題翻盤拆成「檢索差」還是「推理差」兩來源（見 §三）。

---

## 二、翻盤集中在數值/計數/計算題（~78%）

按問題類型分（numeric = how many/how much/how long/how often 或 gold 含數字）：

| bucket | numeric/count/compute | factual/other |
|---|---:|---:|
| **120B 進步（49）** | 38（78%） | 11 |
| **120B 退步（40）** | 31（78%） | 9 |

**兩桶都被數值題主導。** LongMem 的翻盤幾乎全發生在「答案需要從證據**算出來**（加總、相減、數天數、數次數）」的題上，而非「查一個字面事實」的題。這與「gold 集中 1–2 條、agent 裸選常已命中」的 LongMem 特性一致——字面查找題兩模型都對（進 both_right 348），拉開差距的是**推理/聚合**環節。

---

## 三、逐題歸因：檢索差 vs 推理差

把每題翻盤按「對的那版 context 有沒有 gold 證據」拆來源（gold 字面比對為 recall proxy；數值 gold 多為推導值故「neither literal」佔比高）：

| bucket | 兩版 context 都有 gold（**推理差**） | 對版有/錯版無（**檢索差**） | 兩版都無字面（推導/聚合題） |
|---|---:|---:|---:|
| 120B 進步（49） | 21 | 1 | 26 |
| 120B 退步（40） | 7 | 1 | 31 |

**判讀：翻盤幾乎不是「檢索差」（各僅 1 題）。** 主體是兩類：

1. **推理差（進步桶 21 題）**：兩版 context 都含 gold 證據，20B 挑錯/算錯、120B 挑對。這是 120B 淨賺的來源——同樣的證據，強模型抽取/推理更準。
2. **推導聚合題（都無字面 gold）**：答案要跨多條 evidence 計算，勝負取決於**誰的 context 湊齊了計算所需的全部分項**——這才是退步桶的主因（見下）。

---

## 四、進步案例（20B 錯 → 120B 對）

同樣的 evidence，120B 抽對數字、20B 抽錯：

| id | 問題 | gold | 20B（錯） | 120B（對） |
|---|---|---|---|---|
| `945e3d21` | 多久做一次瑜珈緩解焦慮 | 三次/週 | twice a week | three times/week |
| `8fb83627` | 讀完幾期 National Geographic | Five | three（issues 1-3） | 5 issues |
| `affe2881` | 公園看過幾種鳥 | 32 | 27 | 32 |
| `a2f3aa27` | 現在 IG 幾個追蹤者 | 1300 | 1,250 | ~1,300 |
| `0f05491a` | Starbucks Gold 要幾顆星 | 120 | 300（幻覺） | 120 |
| `7a87bd0c` | 每日整理習慣維持多久 | 4 weeks | 算成 19 weeks（日期算錯） | four weeks |

**機制**：這些題 evidence 裡有正確數字（或算它的分項），20B 抽成鄰近的錯值/舊值、或日期減法算錯；120B 抽取與算術更穩。`7a87bd0c` 尤其典型——20B 把「4 週」硬算成「140 天≈19 週」，120B 正確給 4 週。

---

## 五、退步案例（20B 對 → 120B 錯）——多為「聚合湊不齊」

| id | 問題 | gold | 20B（對） | 120B（錯） | 根因 |
|---|---|---|---|---|---|
| `67e0d0f2` | 完成幾門線上課 | 20 | 20（12 Coursera + 8 edX） | **12**（只 Coursera） | 120B 的 agent summary 丟了 edX「8 門」分項→漏加 |
| `157a136e` | 阿嬤比我大幾歲 | 43 | 43（75−32） | 45–55（沒抓到「我 32 歲」） | 120B context 缺「我的年齡」分項→給區間 |
| `46a3abf7` | 共幾個魚缸（含幫朋友設的） | 3 | 3（列出三個） | 2（漏 5-gallon） | 120B 漏一個分項 |
| `5c40ec5b` | 跟德國 Alex 見過幾次 | 二次 | two | three（把首次+兩次疊算） | 120B 多算一次 |
| `852ce960` | Wells Fargo 房貸預核額度 | $400,000 | $400,000 | $350,000 | 抽錯金額 |
| `41698283` | 最近買的鏡頭 | 70-200mm 變焦 | Canon EF 70-200mm | 50mm 定焦 | 抽到舊/錯的一筆 |

**機制**：退步桶的聚合題（`67e0d0f2`/`157a136e`/`46a3abf7`）幾乎都是**120B 那版 agent 的 context 漏了計算所需的某個分項**。以 `67e0d0f2` 為例，兩版 context 逐字對照：

- **20B 版** entity summary 同時保留 Coursera 與 edX 兩線 → 答 20（12+8）✅
- **120B 版** entity 只寫「completed 12 courses on Coursera」、edX 線被壓成「foundational courses」沒帶數字 → 答 12 ❌

**這正是 §三caveat 的實證**：退步不是「120B 答題笨」，而是「120B agent 的檢索/摘要把某個分項壓掉了」，聚合題對缺任何一個分項零容忍。**端到端對照下，agent 檢索的抖動被記到答題模型頭上**——要分離得固定 context 重跑。

---

## 六、結論

1. **20B→120B 是 +1.8pp 的淨值，但底下是 89 題（17.8%）互相抵消的翻盤**，兩模型能力接近、差異是題型敏感的洗牌。
2. **翻盤 ~78% 集中在數值/計數/計算題**——字面查找題兩模型都對，拉開差距的是推理/聚合。
3. **進步桶主因是「同 evidence，120B 抽/算更準」（推理差）**；**退步桶主因是「120B 那版 agent 漏了聚合分項」（檢索差偽裝成答題差）**。
4. **方法學鐵律**：grep filter 版每模型各跑 agent、context 逐題不同，故本文是端到端歸因，**不能當純答題能力對照**。純答題對照須固定 context（見 §二 4o-mini 定樁法）。這是後續若要「乾淨隔離 20B vs 120B 答題」的必要改法。

## 相關文件
- 跨模型機制主文：[`../grep-vs-adjudicate-cross-model.md`](../grep-vs-adjudicate-cross-model.md)
- LongMem n=3 主結果：[`../../result/longmem-adjudicate-n3-20b-120b.md`](../../result/longmem-adjudicate-n3-20b-120b.md)
- 錯題全歸因：[`../longmem-adjudicate-20b-wrong-answers.md`](../longmem-adjudicate-20b-wrong-answers.md)
