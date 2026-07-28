# LongMem Error Analysis Pipeline

針對 LongMem run 中答錯的題目，自動追蹤失敗發生在 pipeline 的哪一層。

## 輸出結構

```text
experiment/longmem/error_analysis/<run_tag>/<type>/
├── summary.csv
└── cases/
    └── {dataset_id}.json
```

`summary.csv` 提供 overview；`cases/` 提供逐題細節。

## 執行方式

### 收集 case

```bash
uv run experiment/longmem/analysis/collect.py \
  --run-tag my-run \
  --type temporal_reasoning
```

### 彙整 summary

```bash
uv run experiment/longmem/analysis/summarize.py \
  --run-tag my-run \
  --type temporal_reasoning
```

### 常用選項

```bash
uv run experiment/longmem/analysis/collect.py \
  --run-tag my-run \
  --type multi_session \
  --dataset_id 099778bb \
  --no_llm

uv run experiment/longmem/analysis/collect.py \
  --run-tag my-run \
  --type temporal_reasoning \
  --no_overwrite
```

## 路徑規則

- dataset CSV：`experiment/longmem/script_data/<type>/`
- run output：`experiment/longmem/output/<run_tag>/<type>/`
- analysis output：`experiment/longmem/error_analysis/<run_tag>/<type>/`
