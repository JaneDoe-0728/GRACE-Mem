# 五配置阶梯：baseline / filter_fetch / 裁决 / ff+hint / 裁决+hint × 跨模型 × 跨 benchmark

> 把 agent filter 的完整配置阶梯（无 filter → grep filter → 裁决 → +hint 两型）在 **gpt-oss 20b/120b × Qwen 4b/27b** 四答题模型、**LongMem × LoCoMo** 两 benchmark 上统一口径跑齐。
> 姊妹文件 [`hypothesis-hint-cross-model.md`](hypothesis-hint-cross-model.md) 专注 hint 一条线；本文是完整五配置矩阵。
> 建立：2026-07-21。

## 0. 一句话结论

**LongMem：配置阶梯有效（4b 73.6→79.8 单调递增，裁决+hint 最高）；120b 强模型下 filter/裁决/裁决+hint 全打平 ~82。**
**LoCoMo：baseline 反而最高（filter 砍 recall 有害），所有 agent filter 配置都追不平 baseline。**
两 benchmark 的结论方向相反——LongMem 有 filter headroom，LoCoMo 没有。

---

## 1. 五配置定义

| 配置 | 定义 | 跑法 |
|---|---|---|
| **baseline** | 无 agent filter，原始 retrieval 直接答题 | `replay_run --no-agent`（LongMem）/ `grep_replay --no-agent`（LoCoMo） |
| **filter_fetch** | grep agent 精炼 context（filter 后允许再 fetch 补证据） | `--mode filter_fetch`，不加裁决 |
| **裁决** | filter_fetch + answer-blind 逐条裁决撿回 | `replay_run`（默认带 adjudicate）/ `--adjudicate-all`（LoCoMo） |
| **ff+hint** | **ff + 裁决 + self-emit hint 三疊加**（filter_fetch context 上 agent FINAL emit `HYPOTHESIS:`，**同 pipeline 再开 answer-blind 裁决撿回**；run 內 `adjudicate=1`，kept 1.49→4.08 即撿回证据） | `--mode filter_fetch --emit-hypothesis`（run 內含 adjudicate） |
| **裁决+hint** | 裁决 context + **外贴 hint**（快版：凍结**无 hint 的裁决 context**、事后贴一句 hint 字串重答；**与 ff+hint 的 self-emit 机制不同**——hint 未参与选证据，故此格无 agent trace） | `longmem_adj_hint_fast.py` / `locomo_hint_fast.py` |

> **两条 hint 路线的本质差异（2026-07-22 查实）**：
> - **`ff+hint` 格 = 理想的「ff + 裁决 + self-emit hint」三疊加**。hint 由 agent 在 filter_fetch 选证据过程中 self-emit（`ff-hint-20b` 实测 emit 率 20.8%、`adjudicate=1` 撿回令 kept 1.49→4.08），**hint 参与了选证据、裁决又撿回**。这一格有完整 agent trace，fb%/kept/add/drop **全有**（见 §6.1）。
> - **`裁决+hint` 格 = 外贴 hint 快版**。凍结的是**无 hint 的裁决 context**（证据已在无 hint 下选定），事后把一句 hint 字串贴到 context 尾巴重答（`longmem_adj_hint_fast.py` 的 `NOTE_TMPL`）。hint **未参与选证据**，且**不跑 agent → 无 trace → 天生无 fb%/kept/add/drop**（这就是这四栏空白的根因）。
> - **结论**：若要「ff+裁决+hint 三疊加带完整定性数据」，直接看 `ff+hint` 格即是；`裁决+hint` 是另一条「事后外贴 hint」对照线，四栏空白是机制使然、非缺实验。

---

## 2. LongMem 五配置阶梯（3vote+abs 口径）

judge = JUDGING.md 全口径（非 abs 题 `correctness_3vote` 错题 3 票重判+carry、`_abs` 题 `correctness_absrubric` 恒单票）。
数字为各 run 全量 ALL accuracy（%），分母见括号（题集不完全一致，见 §4 口径说明）。

| 答题模型 | baseline | filter_fetch | 裁决 | ff+hint | 裁决+hint |
|---|---:|---:|---:|---:|---:|
| **gpt-oss 20b** | 76.15 (499) | 77.76 (499) | 76.95 (499) | **79.80** (495) | 80.00 (470) |
| **gpt-oss 120b** | 75.20 (500) | 79.56 (499) | **80.96** (499) | 78.59 (495) | 82.34 (470) |
| **Qwen 4b** | 73.55 (499) | 75.20 (500) | 76.40 (500) | 76.16 (495) | **79.83** (471) |
| **Qwen 27b** | 77.24 (499) | 81.04 (480) | 81.80 (489) | 82.74 (452)¹ | **84.80** (460)² |
| **gpt-4o-mini** | ⬜ 无run | 72.55 (470) | 72.98 (470) | 73.19 (470) | — ³ |

¹ 27b ff+hint = 外部 `adjn3-lm20b-hyp` hint 池注入（非 self-emit）、452 题无 abs，与其他列不同口径。
² 27b 裁决 = 从头跑 agent（.34+.52，.52 中途卡死改 .34 独占）；裁决+hint = 快版凍结裁决 context @.34，无 abs（纯 3vote，460题）。2026-07-22 完成。
³ **gpt-4o-mini backbone**（答题模型，非 judge；走 OpenAI API 避本地端点长 context 坑）：三格用**三 run 共同题集 470 题**（filter/裁决/ff+hint = `4omini-lm-grep`/`4omini-lm-adj`/`4omini-emit-lm`，同 source `rr16-120b-base`）。ff+hint=self-emit（emit 率 62%，全模型最高）。**⚠️ 2026-07-22 修正：ff+hint 曾误用 `rr16-base-split` source（与对照组不同源、seed 仅 10/16），已重跑对齐（seed 16/16），73.19 为修正后值（旧 source-错值 73.54 作废）。** 裁决+hint 未跑。另有**强制全覆盖 hint**（`ffhint-forcegen-4omini`，100% 覆盖=**74.04**，470题无 abs），见 §7.5。4o-mini 是四模型里最弱的答题 backbone（filter 72.55 < Qwen 4b 75）。

### 参考：另一套 filter 配置（mode=filter，非 filter_fetch）
- `qwen27b-filt-v2hint`（filter + v2 hint）= **81.24%**（469题，无 abs）。属 hint 文档 §3 的 filter 矩阵，非本阶梯。

### 读法
- **单调递增最漂亮的是 4b**：baseline 73.55 → filter 75.20 → 裁决 76.40 → ff+hint 76.16 → **裁决+hint 79.83**。裁决+hint 最高，配置阶梯在弱模型上收益最明显。
- **120b 强模型压缩配置差异**：filter/裁决/裁决+hint 分别 79.56/80.96/82.34，裁决+hint 微胜；但 ff+hint 78.59 反而低于纯 filter——self-emit hint 触发率低（20%）+ 强模型自推不需 hint。
- **20b 中间态**：ff+hint 79.80 与裁决+hint 80.00 基本打平，都比 filter/裁决（77.76/76.95）高约 2-3pp——hint 是 20b 破 79 的主力 lever（呼应 hint 文档 §3）。
- **裁决净效应二维律**：20b 裁决 76.95 < filter 77.76（唯一负案例），其余模型裁决皆 ≥ filter（120b +1.4、4b +1.2）。
- **⚠️ `ff+hint` 列即「ff+裁决+self-emit hint」三疊加**（run 內 adjudicate=1，见 §1 註记）：故它与纯 `裁决` 列的差 = 「加 self-emit hint」的净效应（20b +2.85、120b −2.37、4b −0.24）——hint 在弱模型（20b）加分、强模型（120b）因触发率低反减。**`裁决+hint` 列则是另一条「裁决 context 外贴 hint」路线，非同一机制，两列不可直接相减。**

---

## 3. LoCoMo 五配置阶梯（3vote 口径，与 LongMem 对齐）

judge = 4o-mini 单票判过后，错题以 temp 0/0.3/0.6 三票多数决重判（carry 对题），取 `correctness_3vote`（`.ladder_work/locomo_3vote.py`，复用 LoCoMo standard judge prompt）。LoCoMo 无 `_abs` 棄答题，故无 abs rubric。全 10 samples ~1540 题。**2026-07-21 全部补齐 3vote，与 LongMem 口径对齐（原为 4omini 单票）。**

| 答题模型 | baseline | filter_fetch | 裁决 | ff+hint | 裁决+hint |
|---|---:|---:|---:|---:|---:|
| **gpt-oss 20b** | **84.22** | 82.79 | 83.51 | 82.27 | 84.35 |
| **gpt-oss 120b** | **87.40** | 82.53 | 83.64 | 84.61 | 81.04 |
| **Qwen 4b** | 84.68 | 81.36 | 84.01 | 81.82³ | 84.09⁴ |
| **Qwen 27b** | ⬜ 无run | 85.66⁵ | ⬜ | ⬜ | ⬜ |
| **gpt-4o-mini** | ⬜ 无run | 82.66 | 83.03 | 81.30⁶ | — |

³ 4b ff+hint 快版（`locomo_hint_fast.py`，凍结 `locomo-qwen4b-grep-r1` context + model_answer 抽 hint）@.100，2026-07-21。快版无 trace 定性数据。
⁴ 4b 裁决/裁决+hint = 8-sample 子集（1226题，base `locomo-qwen4b-adjudicate-r1` 只到 s7）。
⁵ 27b LoCoMo filter_fetch 从头跑 agent（.34+.52 双端点独占，systemd-run 保活，2026-07-22），1.4/min 全量 1540 题。fb%=74.1%（27b 在 LoCoMo turn 粒度大量 fallback，与 120b ff+hint 类似）。裁决/裁决+hint 仍无 run（需再跑 agent）。
⁶ gpt-4o-mini backbone（`4omini-lc-grep`/`4omini-lc-adj`/`4omini-emit-lc`，同 source `locomo-n8-120b` + turn 粒度）：ff+hint=self-emit 但 LoCoMo **emit 率 0%**（turn 粒度天生不 emit，见 §7.5）。81.30 的降幅（−1.36pp）来自 `--emit-hypothesis` 的非 hint 副作用——**温和**：final_sids 13.8→10.5（少选 3 条）、fb% 0.1→1.7（几乎没升）。**⚠️ 2026-07-22 修正：曾误用 chunk 粒度导致 fb% 假象飙到 32%，同源 turn 粒度重跑后 fb 仅 1.7%。** 裁决/裁决+hint 未跑。

### 单票对照（历史记录，2026-07-21 补 3vote 前的 `correctness_4omini` 单票口径）

保留原始单票值作对照：3vote 相对单票普遍 +0.2~0.5pp（错题重判回收误杀），方向结论完全一致。

| 答题模型 | baseline | filter_fetch | 裁决 | ff+hint | 裁决+hint |
|---|---:|---:|---:|---:|---:|
| **gpt-oss 20b** | 83.96 | 82.60 | 83.38 | 82.08 | 84.03 |
| **gpt-oss 120b** | 87.27 | 82.34 | 83.64 | 84.55 | 80.82 |
| **Qwen 4b** | 84.42 | 80.71 | 83.77 | 81.43 | 83.61 |

（每格 3vote−单票 Δ 见上表对应格；如 20b baseline 84.22 vs 83.96 = +0.26pp。）

### 读法
- **baseline 反而最高**（20b 84.22、120b 87.40）——filter 是 precision stage，砍 recall 换 precision；LoCoMo turn 全给反而 accuracy 高。**LoCoMo 无 filter headroom**。
- **所有 agent filter 配置都追不平 baseline**：20b 裁决+hint 84.35 略超 baseline 0.13pp（噪音内），其余全负。
- **120b 裁决+hint 最低（81.04）**——hint 在裁决高基底上纯锚定伤害（hint 文档 §5 定律：基底越干净 hint 越毒）。
- **3vote vs 单票**：普遍 +0.2~0.5pp（回收误杀），方向结论不变（baseline 仍最高、filter 仍有害）——见上方单票对照表。
- **⚠️ LoCoMo 的 `ff+hint` 列**不含裁决（`lc-ff-hint-20b`/`120b` 实测 `adjudicate` 全 0，与 LongMem 的 ff+hint **含裁决**不对称）：LoCoMo ff+hint = 纯 ff + self-emit hint。故 LoCoMo 此表**没有** LongMem 那种「ff+裁决+hint 三疊加」列——若要该配置需另跑 `--emit-hypothesis` + adjudicate。`裁决+hint` 列同 LongMem 是外贴 hint 快版。

---

## 3.9 裁决真实效应：乾净单变因对照（deadjudicate，2026-07-22）

> **问题**：「裁决」这个动作本身有没有效益？§2/§3 的「filter vs 裁决」两列相减混了 **agent 重跑噪音**（同题同输入,答题模型重采样仍 ~14.5% 翻转）——不是单变因。
> **乾净口径**：凍结裁决版**同一轮 agent FINAL**,只把裁决 answer-blind 撿回的 sid **拿掉**重答（裁决是 add-only,可逆）。两臂只差裁决、无 agent 重跑噪音。
> **跑法**：LongMem `experiment/longmem/agent_filter/deadjudicate_replay.py`；LoCoMo `experiment/locomo/deadjudicate_replay.py`（2026-07-22 新写,复用同一 harness `_rebuild_context` + `build_chunk_corpus(unit=turn)`）。判分同裁决臂口径（LongMem `correctness_new` 4o-mini 单票、LoCoMo `correctness_4omini`）,取两臂共同题集。

| benchmark | 答题模型 | 裁决臂 | 不裁决臂 | **裁决真实效应** | 修对/改坏(净) | 共同题 |
|---|---|---:|---:|---:|---|---:|
| **LongMem** | gpt-oss 20b | 78.40 | 81.74 | **−3.34pp**（有害） | 22修/37坏(−15) | 449 |
| **LongMem** | Qwen 4b | 74.70 | 72.87 | **+1.82pp**（有益） | — | 494 |
| **LongMem** | Qwen 4b(重现) | 74.70 | 73.08 | **+1.62pp**（有益） | — | 494 |
| **LoCoMo** | gpt-oss 20b | 86.05 | 83.45 | **+2.60pp**（有益） | 53修/31坏(+22) | 846 |

### 读法（核心发现）
- **裁决效益取决于 benchmark,不只取决于模型**：**同一个 gpt-oss 20b**,裁决在 **LongMem 有害（−3.34）**、在 **LoCoMo 有益（+2.60）**,方向相反。LongMem 撿回的证据引入噪音 > 救回;LoCoMo 撿回真补上缺证据。
- **乾净口径把两个方向都放大**（对比 §2/§3 的两独立 run 相减）：LongMem 20b −0.81→**−3.34**、LoCoMo 20b +0.72→**+2.60**。agent 重跑噪音(~14.5%)在两 benchmark 都**稀释**了裁决真实效应——真实信号比文件旧记录强得多,方向不变。
- **4b 裁决有益(+1.82)**：与 gpt-oss 20b 在 LongMem 相反,呼应 §2「裁决净效应二维律」（20b 是 LongMem 唯一裁决有害的模型,乾净量测后伤害被放大到 −3.34）。
- **方法可靠性(4b 重现)**：4b deadjudicate 重跑一次 = **+1.62pp**（vs 原 +1.82pp,差 0.20pp）,两次不裁决臂**一致性 91.3%**（8.7% 翻转 = 答题重采样噪音底）。**重现差(0.2pp)远小于测到的效应量级**（20b −3.34 / LoCoMo +2.60）→ 那些结论不是噪音,deadjudicate 单变因方法本身稳健。

### 3.9.1 LongMem 20b −3.34pp 归因（2026-07-22 深挖）

把 −3.34pp（净 −15 题 = 37 改坏 − 22 修对）按「裁决是否真改 context」分解（比对两臂 `Retrieved_Context` 逐字）:

| 来源 | 改坏 | 修对 | 净 | 换算 |
|---|---:|---:|---:|---|
| **裁决真改 context**（撿回真的进了答题 context） | 27 | 18 | **−9** | **≈−2.0pp** 真实伤害 |
| **纯噪音**（context 逐字相同也翻,答题重采样） | 10 | 4 | **−6** | ≈−1.3pp 运气偏负 |
| 合计 | 37 | 22 | −15 | −3.34pp |

**归因结论:**
- **真实裁决伤害 ≈ −2.0pp**（27 题真改坏）,剩 −1.3pp 是这一轮答题重采样**偶然偏负**（坏侧噪音 10 > 好侧 4,不对称）。真伤害比 −3.34 温和,但**方向确实为负**。
- **伤害机制 = answer-blind 裁决撿回「主题相关但答案错误的干扰证据」**:改坏集中在 **temporal(12)+multi_session(10)**（占 22/37）——这两类有**多个主题相关候选**（多个日期/多个事件）,裁决不知答案、按主题撿回,把干扰项也撿进来,答题模型在有干扰的 context 里**挑错日期/事件**。实例:「读 The New Yorker 几天前」不裁决答 12 天(对)、裁决答 17 天(错);「feedback 日期」不裁决 March 17(对)、裁决 April 15(错)。
- **不是「量」的问题**:改坏题裁决撿回均值 3.4 < 修对题 4.1（撿回越多≠越坏）,也**无 cap 截断**（0 题）。差别纯在撿回证据的**质**（真证据 vs 干扰项）,answer-blind 裁决无法先验区分——**这是裁决的本质限制**。
- 对比 LoCoMo 20b +2.60（53修/31坏,净+22）:同机制在 LoCoMo 撿回的多是**真补缺证据**（LoCoMo turn 粒度、filter 砍太狠留证据不足,撿回补上;LongMem summary 已够、撿回只添干扰）——印证 §0「LongMem 有 filter headroom / LoCoMo 无」的反向律在裁决层同样成立。

（待补:120b/27b 两 benchmark 的 deadjudicate 未跑。）

---

## 4. 口径说明（重要，格间可比性）

> **两 benchmark 表内各自可比，跨表不可比。**
> - LongMem = **3vote+abs** 合成；LoCoMo = **3vote**（无 abs）。**2026-07-21 已对齐，两 benchmark 主口径皆 3vote**（LongMem 多 abs 合成一步）。
> - 判分方式不同源于 benchmark 性质：LongMem 有 `_abs` 棄答题需专用 rubric，LoCoMo 无。

**LongMem 题集不完全一致**（分母差异，格间严格相减需用共同题集）：
- baseline/filter/裁决 = 499-500；ff+hint 系列 = 495；裁决+hint（adjhint）= 470（无 abs）；27b = 452/480（子集）。
- 用 `ladder_score.py` 取共同题集交集才严格可比：核心 495 共同；加裁决+hint（无abs）= 465 共同。
- **本表用各 run 全量分**（便于横看趋势），严格相减请用共同题集重算。

**已知不完全对齐点：**
- **裁决+hint（adjhint-*、adjv2hint120b）无 abs 题**：ctx-run（裁决版）重命名了 abs 文件（去 `_abs` 后缀），继承时丢了 abs 标记 → 470 题纯 3vote。与含 abs 的裁决格（500）口径略异。
- **27b ff+hint** 是外部 hint 池注入（非 self-emit）、452 题子集，与 4b/20b/120b 的 self-emit ff+hint 不同源。

---

## 5. 实验日志（2026-07-21 ~ 22 战役）

### 5.1 run 清单与端点

| 配置 | run tag | 端点 | 判分 |
|---|---|---|---|
| 20b baseline | `rerank16-rr2` | (2026-07-07 产) | 本次补 3vote+abs |
| 20b filter/裁决 | `grep-lm20b-r1` / `adjn3-lm20b-r1` | .92 | 3vote+abs |
| 20b ff+hint | `ff-hint-20b` | 本机 .76 | 3vote+abs |
| 20b 裁决+hint | `adjhint-20b-full` | 本机 .76（用户指定） | 3vote |
| 120b 全系列 | `rr16-120b-base`/`grep-lm120b-r1`/`adjn3-lm120b-r1`/`ff-hint-120b`/`adjv2hint120b-full` | .34 | 3vote+abs |
| 4b baseline | `baseline-4b-full` | .100 | 3vote+abs |
| 4b filter/裁决 | `qwen4b-grep-r1` / `qwen4b-adjudicate-r1` | .100/.86 | 3vote+abs |
| 4b ff+hint | `ff-hint-4b` | .100+.86 双端点 | 3vote+abs |
| 4b 裁决+hint | `adjhint-4b-full` | .99/.100/.86 三端点 | 3vote+abs |
| 27b filter | `qwen27b-grep-r1` | .52 | 3vote+abs |
| 27b ff+hint | `ff-hint-27b-fast` | (外部hint) | 3vote+abs |
| 27b baseline/裁决/裁决+hint | `baseline-27b-full`/`adj-27b-full`/`adjhint-27b-full` | .52/.34 | 3vote(+abs) ✅ |
| LoCoMo 27b filter_fetch | `lc-ff-27b-full`（source=**locomo-n8-turnk24** ⚠️非 locomo-n8） | .34+.52 systemd | 3vote ✅ |
| LoCoMo 全系列 | `locomo-n8`/`grep-lc*`/`adjn3-lc*`/`lc-ff-hint-*`/`lc3combo-*`/`baseline-lc-4b`/`lc-ff-hint-4b-fast` | .92/.34/.100/.86 | 4omini |

### 5.2 端点陷阱（本次踩坑）

- **`.34`/`.99` 的 400 Bad Request = context length 配置太小**，非端点坏。`.34` 重新下载调大 context length 后恢复（长 18k 请求 2.0s 无 400）。**跑前用长 context 健检**。
- **`.86` 会过载卡死**：LoCoMo 4b ff+hint 快版 @.86 跑 5:25 只用 6s CPU（死等 I/O）→ 换 .100（2.5s 快）正常。
- **27b 端点承载低、推理慢**：长 context 8-15s/请求；高并发（workers=3 × 多进程打同端点）会把 `.52` 打到 **ReadTimeout 卡死**（停负载后自愈需时间）。跑 27b agent 用 **workers=2、避免多进程共享同端点**。
- **后台进程保活**：setsid/harness 后台的子进程会被 turn 清理（SIGHUP）；唯一可靠方式是 **`systemd-run --user`**（被 `systemd --user` 领养，ppid→systemd，不随 session 死）。⚠️ 必须用**绝对 python 路径**（`.venv/bin/python`），systemd 环境无 PATH（否则 `exit 127: python not found`）。
- **27b 端点清单**（2026-07-22）：`.34`(`qwen3.5-27b`,可用~1-8s)、`.52:8000`(`Qwen/Qwen3.5-27B`,可用但重载易卡)、`.99`(慢/ReadTimeout)、`.76`(间歇 400,curl 正常但 LLMClient 忽好忽坏)、`.32`(DOWN)。**只 `.34`/`.52` 可靠**。
- Qwen 端点：4b `.100/.86/.99`、27b 见上。

### 5.3 判分流程坑（详见 [[judging-pipeline-gotchas]]）

- LongMem：① 单票 → ② prep carry → ③ 3vote 重判错题 → ④ abs-only 判分（**须自写脚本只判 `*_abs.csv`**，工具 `--dirs` 会污染 non-abs）→ ⑤ 合成。**3vote 与 abs 判分须串行**（并发 to_csv 互相覆盖）。**score 用 `float(s) in (0,1)` 解析**（工具写 `0.0`/`1.0` float，字符串 `=="0"` 会漏 → 虚高 96%）。
- LoCoMo：**首判用 `judge_eval_4omini.py`**（raw eval → `correctness_4omini`）；**重判才用 `rejudge_4omini.py`**（对无 `_judge.csv` 的 run 静默无输出）。

### 5.4 复用现成 run 走快版（省重跑 agent）

- **baseline / ff+hint / 裁决+hint 可快版**：凍结现成 run 的 context + 只重答（`longmem_adj_hint_fast.py` / `locomo_hint_fast.py`），比重跑 agent 快 3-12×。
- **裁决须从头跑 agent**：无现成裁决 context 时（如 27b），只能重跑。裁决+hint 依赖裁决先跑完。
- 慢版（重跑 agent）vs 快版对比见 [[hypothesis-hint-cross-model]] §8。

---

## 6. LongMem 详细分析（per-category 分数 + 定性数据）

> 定性数据来源 = `_grep_agent_traces.jsonl`（`kept`/`added`/`dropped` sid 列表长度 + `fallback` 非空率）。
> **baseline 与快版（裁决+hint）无 trace**（无 agent / 只重答）→ 只有 per-category 分数，无 kept/added/dropped。
> 生成：`/tmp/detailed_analysis.py <run-tag>`。ALL% = 3vote+abs 合成；fb% = fallback 率；kept/add/drop = 每题均值。

### 定性数据的核心洞察

**kept/dropped 揭示各配置在"砍多少"上的策略差异（同 ~16 seed 输入）：**

| 配置 | kept/題 | dropped/題 | 策略 |
|---|---:|---:|---|
| grep filter（20b/120b） | ~1.5 | ~14.4 | **激进砍**：16 seed 只留 1.5，砍 90% |
| grep filter（**4b/27b**） | **~11.5** | **~4.0** | **几乎不砍**：留 11.5，与 gpt-oss 相反！ |
| 裁决（20b/120b） | ~6.4 | ~8.5 | **撿回**：kept 从 1.5→6.4，dropped 减半 |
| 裁决（4b） | 7.7 | 8.2 | 从 11.8 反而砍到 7.7（4b 裁决在删） |
| ff+hint（20b/120b） | ~3.8 | ~10.5 | 中间态 |

- **模型家族分裂**：gpt-oss（20b/120b）grep filter **激进砍**（kept ~1.5）；Qwen（4b/27b）grep filter **几乎不砍**（kept ~11.5）。同一 `--mode filter_fetch` prompt，两家族 filter agent 行为截然相反——**Qwen filter agent 倾向保留，gpt-oss 倾向精炼**。这解释了为何 4b/27b 的 grep 与裁决分数接近（都留很多）。
- **裁决对 gpt-oss 是"撿回"**（kept 1.5→6.4），对 Qwen 4b 是"再砍"（11.8→7.7）——裁决在两家族做的事相反，因为起点不同。

> **kept+dropped 守恒律（2026-07-22 核）**：以 `kept=final∩seed`、`dropped=seed−final` 定义，两 benchmark **全模型全配置 kept+dropped ≡ 16**（seed 均 15.98，少的 0.02 是极少数题 retrieval 只召回 15；added=final−seed 均仅 0.05~0.17，fetch 极少）。
> - **LongMem 天生守恒**：seed（16 summary）与 final 同一套字串 sid，直接 `len(kept)+len(dropped)=16`。（注：trace 里**存的** kept/dropped 字段对裁决/ff 会把 fetch 撿回的 added 塞进 kept 致 >16，那是记录口径；按集合定义重算即守恒。）
> - **LoCoMo 需先对齐粒度**：seed 是 **chunk 级**（`0__4:1`）、final 是 **turn 级**（`0__4:1t2`），必须把 final 截回 chunk 前缀再比，kept+dropped 才 =16（否则字串永不相交、全崩为 0——即本文档修正前的 bug）。

**fallback 率揭示难题分布：**
- **multi_session / preference 最高**（20b ff+hint 26.7% / 23.3%），**assistant 最低**（~1-4%）——多跳/开放题 agent 更常"凑不齐不敢交 FINAL"。
- **强 filter agent fallback 低**：120b grep 8.0% < 20b grep 16.2% < 4b grep 0.8%（4b 几乎不 fallback 因为它不砍、总有东西留）。
- **裁决降 fallback**：撿回证据让 agent 更敢收尾。

### 6.1 gpt-oss 20b 五配置详细

| 配置 | ALL | assistant | user | temporal | multi | pref | KU | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 76.2 | 89.3 | 88.6 | 78.8 | 62.4 | 83.3 | 71.8 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 77.8 | 92.9 | 92.9 | 83.3 | 63.2 | 83.3 | 66.7 | 16.2 | 1.49 | 0.07 | 14.49 |
| 裁决 | 77.0 | 94.6 | 91.4 | 78.8 | 63.2 | 73.3 | 73.1 | 16.2 | 6.32 | 0.07 | 7.18 |
| ff+hint | **79.8** | 94.5 | 95.7 | 78.8 | 66.4 | 83.3 | 78.2 | 19.2 | 4.08 | 0.09 | 8.99 |
| 裁决+hint | 80.0 | 92.9 | 93.8 | 80.2 | 71.3 | 70.0 | 76.4 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

ᵃ baseline 无 agent filter → 无 kept/added/dropped/fallback 概念（类别分数正常有）。
ᵇ 裁决+hint 走**外贴 hint 快版**（凍结无 hint 的裁决 context、事后贴 hint 只重答，不跑 agent）→ 无 trace 定性数据（类别分数正常有）。**若要「ff+裁决+hint」带完整 fb%/kept/add/drop，见上方 `ff+hint` 行**（那格才是 self-emit hint 参与选证据 + 裁决撿回的三疊加，机制见 §1 註记）。

### 6.2 gpt-oss 120b 五配置详细

| 配置 | ALL | assistant | user | temporal | multi | pref | KU | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 75.2 | 80.4 | 97.1 | 73.7 | 62.4 | 66.7 | 79.5 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 79.6 | 92.9 | 95.7 | 78.0 | 64.7 | 86.7 | 80.8 | 8.0 | 1.65 | 0.07 | 14.33 |
| 裁决 | **81.0** | 87.5 | 95.7 | 77.3 | 71.4 | 86.7 | 83.3 | 8.0 | 6.42 | 0.07 | 8.52 |
| ff+hint | 78.6 | 96.4 | 94.2 | 79.5 | 64.9 | 83.3 | 71.8 | 11.5 | 3.75 | 0.05 | 10.51 |
| 裁决+hint | 82.3 | 91.1 | 96.9 | 78.6 | 73.8 | 80.0 | 84.7 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

（ᵃ baseline 无 agent、ᵇ 裁决+hint 快版无 trace，同 §6.1 注。）

### 6.3 Qwen 4b 五配置详细

| 配置 | ALL | assistant | user | temporal | multi | pref | KU | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 73.5 | 85.7 | 94.3 | 73.5 | 54.9 | 83.3 | 74.4 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 75.2 | 96.4 | 97.1 | 66.9 | 56.4 | 86.7 | 82.1 | 0.8 | 11.78 | 0.15 | 4.05 |
| 裁决 | 76.4 | 100.0 | 98.6 | 66.9 | 60.9 | 90.0 | 76.9 | 1.2 | 7.67 | 0.17 | 8.20 |
| ff+hint | 76.2 | 98.2 | 95.7 | 64.4 | 64.1 | 86.7 | 79.5 | 4.8 | 3.79 | 0.12 | 11.42 |
| 裁决+hint | **79.8** | 94.6 | 100.0 | 74.0 | 63.1 | 93.3 | 83.3 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

（ᵃ baseline 无 agent、ᵇ 裁决+hint 快版无 trace，同 §6.1 注。定性数据缺，类别分数正常有。）

### 6.4 Qwen 27b 详细（已判分格）

| 配置 | ALL | assistant | user | temporal | multi | pref | KU | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 77.2 | 89.3 | 94.3 | 78.0 | 64.7 | 83.3 | 70.5 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 81.0 | 92.9 | 94.3 | 79.7 | 67.7 | 90.0 | 82.8 | 3.3 | 11.42 | 0.15 | 4.01 |
| 裁决 | 81.8 | 98.2 | 95.7 | 80.3 | 68.8 | 86.7 | 79.5 | 8.8 | 4.55 | 0.12 | 10.13 |
| ff+hint¹ | 82.7 | 98.2 | 96.9 | 78.0 | 69.7 | 93.3 | 84.9 | — | — | — | — |
| 裁决+hint | **84.8** | 98.1 | 98.4 | 78.6 | 75.9 | 93.3 | 84.7 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |
| filter+v2hint² | 81.2 | — | — | — | — | — | — | — | — | — | — |

¹ 外部 hint 池、452题无abs、无 trace（快版）。²`qwen27b-filt-v2hint`，mode=filter（非 filter_fetch），另一套配置。
（baseline/裁决/裁决+hint 跑中，见 §8 待办。）

### 共同弱项（跨模型）
- **multi_session 最难**（所有配置 55-74%），**temporal 次之**——多跳聚合 + 时序题是硬底，与 hint 文档 §5 病灶同构（多相似候选挑一）。
- **assistant/user 最易**（90-100%）——单会话直答，任何配置都高。
- **add/題 普遍 ~0.1**：filter_fetch 的 fetch 补证据几乎不触发（agent 很少主动补），说明 "fetch" 一步实际作用小。

---

## 7. LoCoMo 详细分析（per-category + 定性数据）

> LoCoMo category = 数据集 1-5 类（`stages/judge.py::CATEGORY_MAP`）：Single-hop / Multi-hop / Temporal / Open-domain / Adversarial（Adversarial 仅 1 题，忽略）。
> 定性数据字段语义**与 LongMem 不同**（见下）。生成：`/tmp/locomo_detailed.py <run-tag>`。分数 = `correctness_3vote`（与 §3 对齐），全 10 samples ~1540 题。

### LoCoMo 定性字段语义（与 LongMem 不同）

**LoCoMo 输入 = fixed 16 chunk seed**（source `locomo-n8` / `locomo-n8-120b`，1540 题全 16，与 LongMem 的 16-summary 同规格）。grep agent 在 **turn 粒度** 上 filter：把每个 seed chunk（`0__4:1`）展开成 turns，选中其中的 turn（`0__4:1t2`）。

**本表 filter_fetch / 裁决 / ff+hint 三行的 kept/add/drop 一律 = chunk 级、口径统一**（`/tmp/lc_recompute_kad.py`，把 final turn sid 截回 chunk 前缀 `0__4:1t2`→`0__4:1` 再跟 16 seed 比）：
- `kept` = seed∩final_chunks、`dropped` = seed−final_chunks、`added` = final_chunks−seed（真 fetch）。
- **三行都 kept+dropped ≡ 16**（与 seed 数守恒），可同表纵比。另有 **turn 级 final 数**（实际进答题 context 的 turn 量）：filter 20b=4.6/120b=5.8/4b=12.4。（27b source 不同，见 §7.4 ⚠️。）

> ⚠️ **裁决/ff+hint 的 dropped 曾虚高到 43-109（2026-07-22 修正）**：**不是砍得多，是粒度放大**。裁决走 [`harness.py:990`](../../experiment/longmem/agent_filter/harness.py#L990) `pending = seed_norm − final`，而 `seed_norm = corpus.normalize_sids(seed)` 把 16 chunk seed **展开成 turn 池**（20b ~50 turn、120b ~107 turn），裁决对整个 turn 池逐条判 → trace 存的 dropped 是 **turn 级**（≈50/107），与 filter 的 chunk 级（16）不同粒度、**不可直接比**。本表已全部换算成 chunk 级（裁决 20b drop 43.44→13.72、120b 108.97→14.22），三行同口径。

> ⚠️ **过去的坑（2026-07-22 修正）**：早前 filter_fetch 行记录 `kept=0/dropped=0/added=全部`，被误注为"grep 只记 added 是设计本意"。**真相是 instrumentation bug + 粒度错配**：① `harness.refine_context` 的 kept/added/dropped 依赖 `seed_norm = corpus.normalize_sids(seed)`，seed sid resolve 失败 → seed_norm 空 → 三值全崩；② 更根本的是 **seed 是 chunk 级（`0__4:1`）、final 是 turn 级（`0__4:1t2`），直接字串比永远不相交**（这也是我第一版用输出 CSV 重算得到假 kept 的原因）。**正解：从 fixed-16 source 抽 seed，把 final 截回 chunk 前缀再比**。修法见 [`../../experiment/locomo/grep_replay.py`](../../experiment/locomo/grep_replay.py) 写 trace 处。

### 7.1 gpt-oss 20b LoCoMo 五配置详细

| 配置 | ALL | Single | Multi | Temporal | Open-dom | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | **84.2** | 90.4 | 81.6 | 74.8 | 69.8 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 82.8 | 87.9 | 81.2 | 75.7 | 66.7 | 6.1 | 1.25 | 0.09 | 14.75 |
| 裁决 | 83.5 | 89.9 | 81.2 | 75.4 | 61.5 | 6.1 | 2.28 | 0.13 | 13.72 |
| ff+hint | 82.3 | 86.8 | 83.0 | 75.7 | 62.5 | 11.4 | 1.21 | 0.10 | 14.79 |
| 裁决+hint | 84.3 | 90.2 | 82.6 | 75.7 | 66.7 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

### 7.2 gpt-oss 120b LoCoMo 五配置详细

| 配置 | ALL | Single | Multi | Temporal | Open-dom | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | **87.4** | 91.7 | 86.2 | 81.6 | 72.9 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 82.5 | 85.5 | 83.7 | 76.3 | 74.0 | 20.2 | 1.01 | 0.19 | 14.99 |
| 裁决 | 83.6 | 88.3 | 83.0 | 76.0 | 69.8 | 20.2 | 1.78 | 0.95 | 14.22 |
| ff+hint | 84.6 | 87.9 | 83.0 | 82.2 | 68.8 | **50.6** | 0.66 | 0.69 | 15.34 |
| 裁决+hint | 81.0 | 84.5 | 78.4 | 78.5 | 67.4 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

### 7.3 Qwen 4b LoCoMo 五配置详细

| 配置 | ALL | Single | Multi | Temporal | Open-dom | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | **84.7** | 92.6 | 80.9 | 77.6 | 50.0 | —ᵃ | —ᵃ | —ᵃ | —ᵃ |
| filter_fetch | 81.4 | 88.8 | 77.3 | 76.9 | 42.7 | 0.1 | 6.72 | 0.10 | 9.28 |
| 裁决 | 84.0³ | 92.1 | 80.3 | 80.1 | 35.5 | 1.8 | 4.90 | 0.09 | 11.10 |
| ff+hint | 81.8 | 87.9 | 79.1 | 77.3 | 52.1 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |
| 裁决+hint | 84.1³ | 91.0 | 78.9 | 79.7 | 51.3 | —ᵇ | —ᵇ | —ᵇ | —ᵇ |

³ 裁决/裁决+hint = 8-sample 子集（1226题，base 只到 s7）。

ᵃ baseline 无 agent、ᵇ 裁决+hint 快版无 trace（同 §6 注，类别分数正常有）。

### 7.4 Qwen 27b LoCoMo（仅 filter_fetch，2026-07-22）

| 配置 | ALL | Single | Multi | Temporal | Open-dom | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filter_fetch | 85.66 | 91.5 | 84.8 | 80.4 | 55.2 | **74.1** | ⚠️待核 | ⚠️待核 | ⚠️待核 |

（baseline/裁决/ff+hint/裁决+hint 仍无 run，需再跑 agent；见 §8 待办。）

> ⚠️ **27b source 口径与其他三格不同（2026-07-22 查实）**：systemd 启动命令 = `grep_replay.py --source-run **locomo-n8-turnk24** --chunk-turns 1`，**不是** 20b/120b/4b 用的 `locomo-n8`（fixed 16 chunk seed）。27b 整个系列（`lc-ff-27b-full`/`lc-baseline-27b`/`lc-adj-27b`）都跑在 **turnk24 基底 + chunk-turns=1 turn 粒度 corpus**：source seed 是 chunk-turns=24 粗块索引（`4__6:1`）、final 是 turn 粒度索引（`4__6:14`），两者 sid 编号体系不同、需 chunk→turn 展开映射才能对齐，无法与其他三格同口径算 kept/dropped。**该格 kept/add/drop 标待核，ALL accuracy（85.66）不受影响**（judge 独立于 trace）。
- **fb%=74.1% 全场最高**：27b 在 LoCoMo turn 粒度大量 fallback（凑不齐 FINAL），与 120b ff+hint（50.6%）同型放大。

### LoCoMo 关键洞察

- **Open-domain 最难**（35-73%，跨模型跨配置全线最低）——LoCoMo 的硬底，需常识推理非纯检索。Single-hop 最易（85-93%）。
- **120b ff+hint 的 fb%=50.6%（异常高）**：一半题 fallback——120b self-emit 触发多但 agent 常凑不齐 FINAL，说明 emit+fetch 在 LoCoMo turn 粒度下决策困难。对比 LongMem 120b ff+hint fb 仅 11.5%。
- **baseline 各类别几乎全线领先**：LoCoMo filter 砍 recall 在**每个类别**都有害（不只总分）——印证 LoCoMo 无 filter headroom 是类别级普遍现象，非某类拖累。
- **Qwen 4b filter kept≈6.7/dropped≈9.3（chunk 级）**：16 seed chunk 保留 6.7 个（turn 级 final=12.4），远高于 gpt-oss 20b/120b 的 kept≈1——**Qwen filter agent 强保留倾向**，与 LongMem 家族律一致。（旧记录 dropped≈94 是把整个 turn 池误算的 bug，已修正为 chunk 级 seed−final。）
- **fb% 与 benchmark 难度正相关**：Open-domain fb% 最高（17-55%），Single-hop 最低——难题 agent 更常 fallback。

---

## 7.5 gpt-4o-mini backbone × hint 三臂（2026-07-21）

> 用 gpt-4o-mini 当**答题 backbone**（非 judge；judge 仍是 4o-mini）跑 hint 三臂：无 hint / self-emit hint / 强制全覆盖 hint。
> 揭示「hint 覆盖率」与「hint 效应」的关系，以及 `--emit-hypothesis` 指令的**非 hint 副作用**。
> 跑法见 [[4omini-emit-hypothesis-cross-benchmark]]：4o-mini 走 OpenAI API（避开本地端点长 context 壊掉的坑），env 前缀 `LLM_API=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini OPENAI_API_KEY=$KEY` 指定。

### 三臂定义

| 臂 | 定义 | run tag |
|---|---|---|
| **无 hint** | 4o-mini filter，不带 hint | `4omini-lm-grep` / `4omini-lc-grep` |
| **self-emit hint** | agent 带 `--emit-hypothesis`，FINAL 时自报 HYPOTHESIS 当 hint | `4omini-emit-lm` / `4omini-emit-lc` |
| **强制全覆盖 hint** | 冻结 filter context，4o-mini 从 context **强制生成** hint 再答（100% 覆盖） | `ffhint-forcegen-4omini`（仅 LongMem） |

### LongMem（470 共同题，correctness_new 3vote；⚠️ self-emit 已用同源 `rr16-120b-base` 重跑，seed 16/16）

| 配置 | emit 覆盖 | ALL | vs 无hint |
|---|---:|---:|---:|
| 无 hint | — | 72.55% | — |
| self-emit hint | **62%** | 73.19% | **+0.64** |
| 强制全覆盖 hint | 100% | 74.04% | **+1.49** |

**单调递增**：hint 覆盖越高分越高（无 → 62% → 100%）。self-emit 修正后增益缩到 +0.64（原 source-错值 +1.07，偏乐观），方向不变。

per-category（无 → self-emit → 强制，同源修正后）：
- **KU/temporal hint 受益**：knowledge_update 73.6→79.2→75.0（self-emit 大赚 +5.6）、temporal 69.0→71.4→73.0（单调 +4）
- **开放/易题 hint 有毒**：preference 56.7→43.3→46.7（self-emit **−13.4**、强制 −10）、assistant 96.4→96.4→92.7（强制 −3.7）
- multi_session self-emit 微降（56.6→55.7），强制才回升（60.7）——self-emit 的难题增益不如强制全覆盖稳。
- 双面律：有唯一答案但难拼的题 hint 是救援；开放题（preference）强制生成具体答案反而锚定错误。

### LoCoMo（1540 题，correctness_4omini 单票）

| 配置 | emit 覆盖 | ALL | vs 无hint |
|---|---:|---:|---:|
| 无 hint | — | **82.66%** | — |
| self-emit hint | **0%** | 81.30% | **−1.36** |
| 强制全覆盖 hint | — | 未跑 | — |

### per-category + 定性数据（全量口径，对齐 §6.1）

> ALL/类别分 = 各 run 全量（LongMem 499/495/470、LoCoMo 1540）；fb%/kept/add/drop 来自 trace（生成 `/tmp/detail_4omini.py`）。
> **摘要表（§7.5 上方）用三臂共同题集严格可比；此处用全量看类别趋势 + agent 定性，两套口径并存。**

**LongMem 三臂 detailed：**

| 配置 | ALL | assistant | user | temporal | multi | pref | KU | fb% | kept | add | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 无 hint | 70.5 | 96.4 | 91.4 | 68.9 | 53.4 | 56.7 | 70.5 | 0.0 | 11.76 | 0.21 | 4.20 |
| self-emit hint | 72.0 | 96.4 | 94.3 | 71.4 | 54.9 | 43.3 | 75.6 | 1.8 | 5.31 | 0.22 | 10.65 |
| 强制全覆盖 | 74.0 | 92.9 | 96.9 | 73.0 | 60.7 | 46.7 | 75.0 | —ᶜ | —ᶜ | —ᶜ | —ᶜ |

ᶜ 强制全覆盖走快版（冻结 context 只重答，无 agent）→ 无 trace 定性数据（类别分正常有）。

**LoCoMo 三臂 detailed：**

| 配置 | ALL | Single | Multi | Temporal | Open-dom | fb% | final_sids | kept·ᵈ | drop·ᵈ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 无 hint | 82.7 | 90.1 | 79.1 | 72.3 | 62.5 | 0.1 | 13.8 | 13.57 | 92.85 |
| self-emit hint | 81.3 | 87.3 | 74.8 | 77.6 | 60.4 | 1.7 | 10.5 | 1.63 | 14.37 |

ᵈ **⚠️ LoCoMo 无 hint run 的 kept/drop 与 self-emit 记法仍不同源**（无 hint `dropped`=从全 turn 池砍的量~93、self-emit `dropped`≈14 只记本轮）——**可比信号是 fb% 与 final_sids**。

### 定性数据揭示 self-emit 的真实机制（⚠️ 2026-07-22 同源重跑修正）

- **LongMem self-emit：agent 砍更多（可比，守恒律成立）**——kept 11.76→5.31、dropped 4.20→10.65（kept+drop≡16 seed，99-100% 守恒）。emit 指令让 4o-mini filter 更激进，净效应 +0.64（小，难题 KU +5.6 盖过 pref −13.4）。
- **LoCoMo self-emit：温和负效应，非「filter 崩坏」**——**同源修正后**：final_sids **13.8→10.5**（略少证据）、fb% **0.1→1.7**（几乎没升）。`--emit-hypothesis` 只让 agent 稍微少选一点证据（−3 条）+ 极小幅 fallback → 掉 1.36pp，是**温和负效应**。**（⚠️ 早期用错 source/chunk 粒度时报的「fb%=31.9、final_sids=1、filter 崩坏 kept 掉 94%」全是 source/粒度错造成的假象——同源 turn 粒度重跑后 fb 仅 1.7%、证据数正常，机制完全不同，已更正。）** 类别上 Multi-hop 掉（79.1→74.8）、Open-dom 掉（62.5→60.4），但 Temporal 反升（72.3→77.6）。

### 两个核心发现

- **4o-mini emit 率两 benchmark 天差地别**：LongMem **63%**（比 20b 的 26% 高一倍多——4o-mini 最听 emit 指令），LoCoMo **0%**（1540 题全不 emit）。原因：LoCoMo 是 chunk 粒度，agent 交 chunk sid 就收尾，`--emit-hypothesis` 指令在这语境下被完全忽略（呼应 §7 「LoCoMo 120b ff+hint fb=50.6% 异常高、emit+fetch 在 turn/chunk 粒度决策困难」）。
- **LoCoMo 0% emit 却掉 1.36pp，机制=温和少选证据（非崩坏）**：self-emit 0 覆盖（无任何 hint）却掉 1.36pp。**同源修正后可比信号**：final_sids **13.8→10.5**（少选 3 条）、fb% **0.1→1.7**（几乎没升）。`--emit-hypothesis` 只让 agent 稍微少选一点证据 + 极小幅 fallback → 温和 −1.36pp。**结论：emit-hypothesis 指令本身会改 agent 的选证据行为（不只是产 hint），但在 LoCoMo 是温和负效应，非崩坏。**（⚠️ 早期用错 source/chunk 粒度报的「fb%=31.9、final_sids=1、filter 崩坏」是假象，同源 turn 粒度重跑已更正。）

### backbone 对比（self-emit 效应）

| backbone | LongMem emit 率 | 单题增益 |
|---|---:|---|
| gpt-oss 20b | 26% | ff+hint +1.6pp（含 agent 重跑） |
| **gpt-4o-mini** | **63%** | self-emit +1.07pp |

4o-mini emit 更勤，但单题增益反而没 20b 大——因为 4o-mini 连「没把握的题」也 emit，那些 hint 品质差。

---

## 8. 待办

- 🏃 **27b 裁决 / 裁决+hint**（LongMem）跑中（.34+.52 双端点），完成后判分填格。
- 🏃 **LoCoMo 4b ff+hint** 快版跑中，完成判分填最后一格。
- ⬜ **27b baseline**（LongMem）快版跑中。
- ✅ **LoCoMo 27b filter_fetch = 85.66%**（2026-07-22 跑完，systemd 保活 + .34/.52 双端点，1.4/min）。
- ⬜ **LoCoMo 27b 其余 4 格**（baseline/裁决/ff+hint/裁决+hint）：需再跑 agent（27b 慢 1.4/min，每格 ~1.5h）。
- ⬜ **abs 口径对齐**：裁决+hint 系列缺 30 个 abs 题（ctx 重命名丢标记），若要严格同口径需从 source 映射补判。
- ⬜ **LoCoMo 4o-mini 强制全覆盖 hint**（§7.5）：LongMem 已跑（74.03%），LoCoMo 未跑——补齐可验证「强制 hint 能否救 LoCoMo self-emit 0% 覆盖」。用 `/tmp/locomo_hint_fast.py` 或 forcegen 法。

## 相关文件

- hint 专线：[`hypothesis-hint-cross-model.md`](hypothesis-hint-cross-model.md)（§3.6 filter_fetch+hint、§4 三叠加、§5 LoCoMo hint 归因）
- 判分口径：[`../JUDGING.md`](../JUDGING.md)
- 裁决 n=3：[`../result/longmem-adjudicate-n3-20b-120b.md`](../result/longmem-adjudicate-n3-20b-120b.md)、[`../result/locomo-adjudicate-n3-20b-120b.md`](../result/locomo-adjudicate-n3-20b-120b.md)
- grep vs 裁决二维律：[`grep-vs-adjudicate-cross-model.md`](grep-vs-adjudicate-cross-model.md)
