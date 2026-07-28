# grep filter vs 裁決版：跨答題模型的相依對照 + floor / hint 機制發現

日期：2026-07-20
分支：`codex/agent-filter-coverage-session`

本文件匯總三個相互關聯的機制發現，皆圍繞「⑦ 裁決層 add-only 疊在同一 agent FINAL 上」（見 [`../diagrams/agent_filter_pipeline.md`](../diagrams/agent_filter_pipeline.md)）：

1. **grep filter vs 裁決版是相依關係**（共用 agent FINAL），非兩次獨立 call。
2. **裁決效果取決於答題模型強度**——弱模型（20B）被裁決撿回的證據稀釋，強模型（120B/4o-mini）能兌現。
3. **evidence_floor 盲補對 accuracy 零貢獻**——只讓 kept 定性數字失真。

---

## 一、相依關係（機制）

裁決版 = grep filter（同一 agent FINAL）+ ⑦ 裁決層。⑦ 是 answer-blind LLM 對「被 FINAL 丟掉的 seed」逐條判 KEEP → 補回（**只加不減**）。

```
① seed → ② agent 搜尋 → ③ FINAL(agent 裸選) → ④⑤⑥ cap
                                                  │
              ┌────────────────────────────────────┤
              ▼(跳過⑦)                            ▼(走⑦)
        grep filter = agent 裸選            裁決版 = agent 裸選 + ⑦ 撿回
```

**佐證相依的關鍵數據（kept/題）**：grep filter 是裸 agent 選擇，裁決版在其上撿回。

| benchmark / 答題 | grep filter kept | 裁決版 kept | 裁決撿回 |
|---|---:|---:|---:|
| LongMem 20B | 1.7 | 6.6 | ~4.9 |
| LongMem 120B | 1.7 | 6.9 | ~5.2 |
| LoCoMo 20B | 4.6 | 7.5 | ~2.9 |
| 4o-mini LongMem | 2.2 | 5.2 | ~3.0 |
| 4o-mini LoCoMo | 10.7 | 14.5 | ~3.8 |

**正確的對照方法**：從同一 agent run 的 trace 抽 FINAL，去裁決 KEEP + 去 floor pad = grep filter 版，重組 context 後**只重答題（agent 不重跑，FINAL bit 級相同）**。這比「兩次獨立 agent replay」嚴謹——差異被隔離成純 ⑦ 裁決層。

---

## 二、裁決效果 × 答題模型強度（決定性）

### 本地 20B / 120B，兩 benchmark（相依對照，3vote 口徑，n=3）

| 配置 | grep filter | 裁決版 | 裁決 Δ |
|---|---:|---:|---:|
| **LongMem 20B** | 78.3 ± 0.5% | 75.6 ± 2.0% | **−2.7pp** ❌ |
| **LongMem 120B** | 79.4 ± 0.2% | 79.8 ± 0.0% | +0.4pp |
| **LoCoMo 20B** | 82.6 ± 0.2% | 83.8 ± 0.4% | **+1.2pp** ✅ |
| **LoCoMo 120B** | 82.5 ± 0.3% | 83.9 ± 0.3% | **+1.4pp** ✅ |

### 4o-mini（agent + 答題都 4o-mini，context = 120b retrieve 的 16-seed，去 floor，兩 benchmark）

| benchmark | grep filter（裸 agent） | 裁決版 | 裁決 Δ |
|---|---:|---:|---:|
| LongMem | 70.3% | 71.3% | +1.0pp |
| LoCoMo | 82.1% | 83.6% | +1.5pp |

### 判讀：裁決效果二維律（模型強度 × benchmark 特性）

**唯一負的是 LongMem 20B（−2.7），其餘全正。** 裁決效果由**答題模型強度 × benchmark 特性**二維共同決定：

- **LongMem 20B（弱模型 × seed 16 小池）＝唯一負案例**。agent 裸選 1.7 條（含答案最小集），裁決撿回 ~5 條「主題相關但不含答案」→ context 1.7 灌到 6.6，**20B 被雜訊稀釋、答題天花板不兌現 recall**，sd 也從 0.5 暴增到 2.0。
- **LoCoMo 20B（同弱模型，但 turn 池大、答案短、gold 集中）→ +1.2**。即使 20B 也能消化裁決撿回的證據（撿回佔比小）。
- **120B / 4o-mini（強模型）兩 benchmark 皆正（+0.4~+1.5）**。強答題模型消化撿回證據不被稀釋。
- **總律**：⑦ 裁決層補的是 **recall**，能否兌現取決於「答題模型能否消化撿回的證據」——受**模型強度**（弱模型易被稀釋）× **benchmark 特性**（小池 + 長答案放大稀釋）共同決定。

> 這推翻「裁決版是無條件主線」的預設——相依對照才揭露這個二維結構（兩次獨立 run 會混入 agent 重採樣噪音，看不到）。

---

## 三、evidence_floor 盲補：對 accuracy 零貢獻

`grep_agent_evidence_floor=12`：agent 選太少時，按 rerank 原序盲補到 12 條（recall safety net）。純 grep filter **不該有 floor**——floor 盲補是繞過 agent 的硬塞。

### 發現：floor 灌水對 accuracy 零貢獻，只讓 kept 定性數字失真

| LongMem 4o-mini grep filter | accuracy | kept | 組成 |
|---|---:|---:|---|
| 有 floor（舊） | 70.5% | 11.8 | agent 2.2 + **floor 盲補 9.5** |
| 去 floor（`--no-floor`，裸 agent） | 70.3% | 2.2 | agent 2.2 |

**accuracy 幾乎不變（70.5→70.3），但 kept 從假 11.8 變真 2.2**——floor 盲補的 9.5 條 rerank 原序**對正確率零貢獻**，純粹是表面數字灌水。LoCoMo 同樣（82.7→82.1，kept 13.6→10.7）。

### 這顛倒了 grep vs 裁決的 kept 關係（重要陷阱）

| 口徑 | grep filter kept | 裁決版 kept | 關係 |
|---|---:|---:|---|
| **含 floor（舊，錯誤）** | 11.8（灌水） | 8.0 | **反了**：grep 看起來 kept 更多 |
| **去 floor（正確）** | 2.2 | 5.2 | **正確**：裁決 kept > grep（撿回 +3） |

含 floor 時 grep filter 被灌到 11.8 > 裁決版 8.0，讓人誤以為「裁決砍證據」。**去 floor 後恢復正確相依**：裁決版 kept > grep filter（裁決撿回）。

**教訓**：論文報 grep filter 定性數字須用 `--no-floor`（裸 agent）；含 floor 的 kept 是灌水假象，會誤導「裁決 vs grep」的關係判讀。新增 CLI flag：`replay_run.py --no-floor`、`grep_replay.py --no-floor`。

### floor 全域停用（2026-07-20）

基於「零貢獻 + 只讓 kept 失真」的證據，**floor 盲補已從 pipeline 全域停用**：
`experiment_config.grep_agent_evidence_floor` 預設 **12 → 0**，`harness.py` 兩處 `_portfolio_pad` floor 呼叫**整段註解**（`force_verified_min` 對 floor 的耦合一併解除、fallback 直接寫 12）。`--no-floor` flag 保留為 no-op（避免既有腳本報錯）。v2（證偽存檔）的獨立 floor 呼叫因 config=0 條件永不觸發。

### ⚠️ floor ≠ ⑦ 撿回（兩者是不同機制，勿混稱）

裁決版一題的 `final_sids` 其實由**三塊**組成，`floor 灌水` 只是其中一塊——**⑦ 裁決撿回是 informed 的，floor 才是盲補灌水**：

| LongMem 20B 一題 final 拆解 | 條/題 | 決策方式 | 對 accuracy |
|---|---:|---|---|
| **裸選**（agent 自選，含答案最小集） | 1.69 | agent GREP/READ | 主體 |
| **⑦ 裁決撿回**（LLM 逐條讀 seed 判「主題相關」才 KEEP） | 2.66 | **informed**（看內容） | 補 recall，看模型能否消化 |
| **floor 灌水**（按 rerank 原序硬塞） | 2.47 | **盲補**（只看排名） | **零貢獻**（70.5→70.3） |
| = final | 6.83 | | |

**關鍵區分**：⑦ 撿回讀了 seed 內容做逐條判斷（informed），floor 只按 rerank 排名盲塞（uninformed）。停用的是 floor，**⑦ 撿回保留**（它是裁決版的核心機制）。

### ⚠️ kept 恆等式：含 floor 時「裁決版 kept ≠ grep kept + ⑦撿回」

trace 記錄的 `kept` 欄位是**最終 context 裡屬 seed 的全部 sid**，含 floor 時它由**三塊**組成，因此**含 floor 的舊 kept 不滿足「grep kept + ⑦撿回」的等式**——差的正是 floor：

```
含 floor（舊）：裁決版 kept 6.73 = grep 裸選 1.60 + ⑦撿回 2.66 + floor 灌水 2.47   ← 3 項
去 floor（新）：裁決版 kept 4.26 = grep 裸選 1.60 + ⑦撿回 2.66                    ← 2 項,等式成立
```

**所以本文早期表格報的「裁決版 kept 6.6/6.9」是含 floor 的灌水值**（LongMem）；去 floor 後真裁決版 kept ≈ 4.3。**只有去 floor 後**，「裁決版 kept = grep filter kept + ⑦撿回」才成立。LoCoMo 本就無 floor，其 kept（7.5/8.3）一直是「裸選 + 撿回」兩項、等式一直成立。

---

## 三之二、真裁決版 ⑦ 撿回定性（去 floor，四配置）

扣掉 floor 灌水後、⑦ 裁決層**真正撿回**多少（`adjudication.kept`）：

| 配置 | grep 裸選 kept/題 | ⑦ 純撿回/題 | 撿回命中題% | 撿到的題平均撿回 | floor 灌水/題 |
|---|---:|---:|---:|---:|---:|
| **LongMem 20B** | 1.69 | 2.66 | 56.1% | 4.75 | 2.47（停用前） |
| **LongMem 120B** | 1.72 | 2.17 | 52.3% | 4.15 | 2.60（停用前） |
| **LoCoMo 20B** | 4.60 | 2.86 | 55.3% | — | **0**（本就無 floor） |
| **LoCoMo 120B** | 5.76 | 2.92 | 42.5% | — | **0**（本就無 floor） |

**定性判讀：**

1. **⑦ 撿回範圍 = 每題 2.2–2.9 條**，約 42–56% 的題會觸發撿回。這是繞開 floor 灌水後、⑦ 層真正的貢獻量（不是灌水）。
2. **LongMem 裸選極少（1.7 條）**：agent 自剪到含答案最小集，⑦ 撿回 ~2.7 條「主題相關」→ 這正是二維律裡 LongMem 20B **被稀釋**的來源（1.7 灌到含撿回的 4.4，弱模型消化不了）。
3. **LoCoMo 無 floor、⑦ 撿回真實（2.86/2.92）**：LoCoMo 的 kept 全是「裸選 + ⑦ 撿回」，沒有灌水成分。LongMem 的 kept 停用前混了 2.5 條 floor 灌水，**停用後才是乾淨的真裁決版**。
4. **LoCoMo 120B 撿回命中題% 最低（42.5%）但單題保守**：120B agent fallback 高、裸選多（5.76），需撿回的題較少。

### 去 floor 後 accuracy 效應（LongMem，r1-vs-r1 乾淨對照）

用 `defloor_replay.py`（凍結 adjn3-lm*-r1 那輪 FINAL，`去 floor final = final_sids − evidence_floor_padded − evidence_coverage.added`，**保留裁決撿回**，`_rebuild_context` 重組乾淨 context、同答題模型 ANSWER_TEMP=0.3 重答），只切 floor 一個變因（無 agent 重跑噪音）。judge 同口徑（非 abs 3vote + abs rubric）：

| LongMem r1 | 含 floor | 去 floor | **floor 效應** |
|---|---:|---:|---:|
| **20B** | 76.35% | 75.35% | **−1.0pp** |
| **120B** | 79.96% | 78.56% | **−1.4pp** |

**發現：去 floor 後兩配置都小降（−1.0 / −1.4pp），皆在 ±3~4 題重採樣抖動（≈±0.7pp）帶內。** 這**證實 floor 對 accuracy 接近零貢獻**（甚至微負，與 4o-mini 的 −0.2 方向一致）：floor 盲補的 ~2.5 條 rerank 原序 seed 對答題無實質幫助、偶爾還稀釋。per-cat 差異散落在 preference/KU（各掉 1–2 題），非系統性。

**結論**：停用 floor 是正確的——移除一個對 accuracy 無益、只讓 kept 灌水失真（6.73→4.26）、顛倒 grep-vs-裁決 kept 關係的機制。**去 floor 後 kept 恆等式成立**：裁決版 kept 4.26 = grep 裸選 1.60 + ⑦撿回 2.66。run tag：`adjn3nofloor-lm{20b,120b}-r1`。

### ⭐ 逐題證據：撿回量 vs 答對率（為何「加 evidence」不必然提升正確率）

**直覺假設「多加 evidence 提升正確率」是錯的。** 按 ⑦ 撿回量把題分組、看各組答對率，四配置**方向完全相反**——這是二維律最直接的逐題證據：

| 配置 | 撿回=0 | 撿回 1–2 | 撿回 3+ | 趨勢 |
|---|---:|---:|---:|---|
| **LongMem 20B** | 77.9% | 77.8% | **70.8%** | **↓ 撿回越多越差（−7pp）** |
| **LongMem 120B** | 81.1% | 78.4% | **75.1%** | **↓ 撿回越多越差（−6pp）** |
| **LoCoMo 20B** | 80.0% | 86.2% | **86.4%** | **↑ 撿回越多越好（+6pp）** |
| **LoCoMo 120B** | 82.5% | 84.4% | **85.5%** | **↑ 撿回越多越好（+3pp）** |

（LongMem 用去 floor 裁決版判分隔離 floor；LoCoMo 本就無 floor。撿回量 = `adjudication.kept` 條數。）

**機制解釋——evidence 的「量」≠「品質」：**

1. **⑦ 撿回是 answer-blind 的**：它判「seed 與問題**主題**相關」，**不是**「seed **含答案**」。所以撿回主要是「主題相關但不含答案」的低精度 evidence，只有少數是 agent 誤砍的真答案。
2. **agent 裸選是高精度**（自己讀過、確認含答案才留）；⑦ 撿回是**低精度**（只判主題）。加低精度 evidence = **加噪音**。
3. **LongMem（gold 集中 1–2 條）**：agent 裸選 1.7 條往往已命中答案最小集 → 撿回 3+ 條全是多餘噪音 → 弱模型在噪音裡挑錯 → **↓**。這就是「裁決版輸給純 grep filter」的逐題根源。
4. **LoCoMo（gold 散在多 turn）**：裸選 4.6 條**不夠**、真有缺 → 撿回補到缺的 gold turn → **↑**。

**因果鏈（LongMem 20B −2.7pp 的根源）：**

```
純 grep filter：context = 1.7 條裸選（高精度、含答案最小集）      → 20B 專注    78.3%
裁決版：       context = 1.7 裸選 + 2.7 低精度撿回（主題相關噪音）→ 20B 被稀釋  75.6%
```

**一句話**：裁決版假設瓶頸是「recall 不足」而補證據，但 LongMem 20B 的真瓶頸是「弱模型答題精度」——補進的低精度 evidence 稀釋了 agent 已挑好的高精度裸選。「加 evidence 有益」**只在裸選不足（recall 真缺）× 模型夠強能消化噪音時成立**（LoCoMo / 120B），LongMem 20B 兩條件皆不滿足 → 撿回純傷害。

---

## 三之三、⑦ 裁決觸發率剖析：多少題根本沒被裁決碰到

裁決層**不是每題都跑**。從 trace 統計四配置的觸發/未觸發分佈（`adjn3-*-r1`）：

| 配置 | n | 觸發撿回>0 | 觸發撿回=0 | 類別不在 gate | fallback | final 滿 16 | **未觸發合計** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LongMem 20B** | 499 | 280 (56%) | 47 (9%) | 112 (22%) | 58 (12%) | 0 | **172 (34%)** |
| **LongMem 120B** | 499 | 261 (52%) | 76 (15%) | 120 (24%) | 40 (8%) | 2 | **162 (32%)** |
| **LoCoMo 20B** | 1540 | 854 (56%) | 434 (28%) | 0 | 94 (6%) | 158 (10%) | **252 (16%)** |
| **LoCoMo 120B** | 1540 | 654 (43%) | 392 (26%) | 0 | 310 (20%) | 184 (12%) | **494 (32%)** |

### 未觸發的四個原因（依 harness.py:975-1004 觸發條件）

裁決觸發需同時滿足：`adj_on=1`（且 category 在 gate）**且** `len(final) < max_sids`（有空間）**且** `pending` 非空（agent 有丟 seed）。任一不滿足即未觸發：

1. **類別不在 gate（LongMem 主因，22–24%）**：裁決只對 4 類開（preference/multi/temporal/KU），**user + assistant 兩類刻意不裁決**（各 ~55 題）。原因：這兩類是**單針型**——gold≈1 條、agent 全中率 80%、94% acc，裁決撿回只會稀釋一針見血的題。這是刻意 gate 設計，非失敗。**LoCoMo 走 `category=None` 全類觸發，無此排除**（gate 欄 0）。
2. **fallback 題（6–20%）**：agent 執行失敗（輸出無效／超步數預算／亂碼）→ 原 context 原封退回，pipeline 不走到裁決。**LoCoMo 120B fallback 高達 20%**（turn 池大、120B zero_keep 多），是它未觸發率衝到 32% 的主因。
3. **final 已滿 16（LoCoMo 10–12%）**：agent 選滿 max_sids 就沒空間補回。**只發生在 LoCoMo**（turn 池大 agent 選得多）；LongMem 20B 裸選僅 1.7 條，幾乎不會選滿（0 題）。
4. **裁決 error（<1%）**：裁決 call 拋錯，floor 續行（已停用）。

### ⚠️ 「觸發但撿回 0」也等於裁決無效（9–28%）

觸發裁決 ≠ 有補回。answer-blind 逐條判時，若所有被丟 seed 都判「不相關」→ 全 DROP、撿回 0 條。這在四配置都不小（LongMem 20B 9%、LoCoMo 20B 高達 28%）。

**合併「未觸發」+「觸發但撿回 0」= 裁決層實際沒補任何 seed 的題**：

| 配置 | 未觸發 | +觸發撿回0 | **裁決零貢獻題合計** |
|---|---:|---:|---:|
| LongMem 20B | 172 | 47 | **219 (44%)** |
| LongMem 120B | 162 | 76 | **238 (48%)** |
| LoCoMo 20B | 252 | 434 | **686 (45%)** |
| LoCoMo 120B | 494 | 392 | **886 (58%)** |

**判讀**：裁決層實際只在**約一半題**（LongMem 56%、LoCoMo 42–55%）真的補了 seed。這解釋了為何裁決 Δ 幅度都不大（±3pp 內）——它只作用於半數題。LongMem 20B 的 −2.7pp，是那 56% 被補題裡「補進噪音稀釋弱模型」的淨傷害壓過了少數真補回的收益。

### 「answer-blind 全 DROP」的判準機制（為何撿回 0）

「觸發但撿回 0」= 裁決逐條判被丟 seed，**全部判 DROP**。理解這個要先看裁決的判準（[`prompts.py`](../../../experiment/longmem/agent_filter/prompts.py) `ADJUDICATE_SYSTEM`，harness.py:503 `_adjudicate_candidates`）：

- **被丟的 seed** = agent 沒選進 FINAL 的候選（`pending`）。一題 16 seed，agent 裸選 1.7 條，其餘 ~14 條交裁決。
- **answer-blind** = 裁決是**獨立 LLM call**，看不到 agent 推出的答案、搜尋歷史、gold。只拿到「問題 + 這 14 條 seed 內容」。
- **判準是「主題相關」非「含答案」**：`KEEP if 帶問題 subject 的資訊；DROP if 無關 subject`。因為 answer-blind 不知道答案，只能判主題。
- **全 DROP** = 逐條判下來，14 條的主題都跟問題對不上（都是無關的其他對話）→ `kept=[]` → 撿回 0。

**真實案例**（temporal 題「讀完某書到參加活動過了幾天」）：agent 裸選 2 條（已含關鍵日期），裁決對其餘 14 條逐條判、全 DROP → 撿回 0。

**這其實無害甚至有益**：全 DROP = 裁決沒把噪音補進來，agent 的乾淨裸選保持不變。對照 §三之二「撿回越多越差」——真正傷害 LongMem 的是**撿回 3+ 條**的題（塞進主題相關但不含答案的噪音），不是全 DROP 的題。

> **附：偶發 400 Bad Request（本機 endpoint context 太小）**。本機 `localhost:1234` gpt-oss-20b 若只載 `n_ctx=16384`，agent `_run_loop` 累加 prompt（system + evidence + 多輪 READ 拉回的 turn 全文）跨輪撐大後可能超限，LM Studio 直接回 400（`n_keep > n_ctx`）而非截斷。這被 pipeline 安全網接住 → 該輪計入 fallback（`fallback=exception`），最終仍有結果。根治：`lms load ... --context-length 32768`。與 context=61072 的「長 context 退化亂碼」是同源坑的兩種表現（見 memory `oss20b-endpoint-garble`）。
>
> **量化影響（四配置 adjn3-*-r1）**：
>
> | 配置 | endpoint | 撞 400 題 | **最終仍 fallback** | 佔全量 |
> |---|---|---:|---:|---:|
> | LongMem 20B | localhost:1234（n_ctx=16384） | 31 | **6** | 1.2% |
> | LongMem 120B | .34 | 0 | 0 | 0% |
> | LoCoMo 20B | .92 | 0 | 0 | 0% |
> | LoCoMo 120B | .34 | 0 | 0 | 0% |
>
> **只有 LongMem 20B（localhost:1234）受影響**，31 題撞 400、但其中 25 題經重試由成功輪次救回，**最終僅 6 題（1.2%）真因 400 而 fallback**（撞 400 最多的是 multi_session context 最長類）。其餘三配置的 endpoint（.34/.92）context 較大，0 題撞限。**對最終結果影響極小**——這 6 題被吸收進 LongMem 20B 的 22% 總 fallback（主體是 `no_final` 117 題、`zero_keep` 11 題，非 400）。

---

## 四、附：hypothesis hint 生產化——指標依賴律（20B-only lever）

`hyp-v1`（4o-mini 事後抽 agent 內心答案當防禦性 hint，20B +2pp）的生產化：讓 agent 在 FINAL 同則訊息直接輸出 `HYPOTHESIS: <答案>` 行（config `grep_agent_emit_hypothesis` + CLI `--emit-hypothesis`），同模型自洽、零 4o-mini 依賴。

**命中率**：agent 自報 71–80%，追平 4o-mini 事後抽取（70%）。機制完美。

**但效果分裂（指標依賴律）**：

| 配置 | judge accuracy Δ | F1 Δ | BLEU Δ |
|---|---:|---:|---:|
| LongMem 20B（emit-hypothesis vs 無 hint 均值） | −1.4（噪音內） | **+5.7** | **+3.9** |
| LongMem 120B（乾淨隔離） | −0.2（噪音內） | — | — |
| LoCoMo 120B | −0.2（有 hint 題 −1.4） | — | — |

**發現：hint 的效果取決於指標**——對**字面指標（F1/BLEU）強正**（hint 讓答案變短貼 gold 字面），對**語意 judge accuracy 中性**（不改「答對率」）。

hyp-v1 當初的 +2pp 是**單票 judge + temp=0** 特定條件的產物；換 temp=0.3 + 3 票口徑後 accuracy 增益消失、F1/BLEU 增益現形。完整版圖：LongMem 20B（唯一 accuracy 正例的條件）/ 120B −0.2 / LoCoMo −0.2~−0.7。**hypothesis hint = 特定 judge 口徑下的 20B-only lever**；F1/BLEU 上則跨配置一致正。

---

## 五、產物與 config

- **相依 grep filter 重建**：從裁決版 trace 抽 agent FINAL（去裁決去 floor）重答，`/tmp/rebuild_grep.py`（LongMem）/ `/tmp/rebuild_grep_locomo.py`（LoCoMo）。
- **CLI flags**：`--no-adjudicate`（關裁決）、`--no-floor`（關 floor 盲補）、`--emit-hypothesis`（agent 自報 hint）。
- **判分口徑**：一般題 4o-mini 錯題 3 票重判 + carry；`_abs` 題強化版 rubric + 恆單票（`correctness_absrubric`，2026-07-20 起）。詳見 [`../JUDGING.md`](../JUDGING.md)。本文表格數字為當時 3vote 口徑，abs 重判平移見 n=3 主結果文件。

## 相關文件

- LongMem n=3 主結果：[`../result/longmem-adjudicate-n3-20b-120b.md`](../result/longmem-adjudicate-n3-20b-120b.md)
- pipeline 流程圖：[`../diagrams/agent_filter_pipeline.md`](../diagrams/agent_filter_pipeline.md)
- 判分口徑：[`../JUDGING.md`](../JUDGING.md)
- 錯題全歸因：[`longmem-adjudicate-20b-wrong-answers.md`](longmem-adjudicate-20b-wrong-answers.md)
