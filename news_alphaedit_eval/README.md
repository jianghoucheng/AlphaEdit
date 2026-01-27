# News → AlphaEdit sequential editing eval

Small adapter around **AlphaEdit** for running sequential edits on `news_dataset_v1` (and larger variants).

Repository layout assumed:

```
KnowledgeTimeline/
  AlphaEdit/                 # upstream AlphaEdit clone/copy
  news_alphaedit_eval/       # this folder
```

## Inputs

`NEWS_DATASET_ROOT` points at a folder containing `dataset.jsonl` (preferred) or `dataset.json`.

Example:

```
/N/project/cogllm/ztiganj/news_datast/outputs/news_dataset_v1
```

Each `dataset.jsonl` record is expected to contain:

- `extracted.subject`
- `extracted.relation`
- `extracted.obj`
- `extracted.fact_date` (ISO `YYYY-MM-DD`)
- `extracted.ripples[]` with `{question, expected_answer, category, answer_type}`

## What gets generated

For each condition:

- `work/news_<condition>.json` (CounterFact/MCF-like JSON list)
- `work/news_<condition>.meta.jsonl` (sidecar containing the original ripple Q/A)

The dataset is installed into AlphaEdit under several plausible locations (`data/`, `dsets/`, etc.) using `ds_name=mcf`.

## Conditions

- `temporal`: sort by `fact_date` ascending
- `random`: shuffle with a fixed seed
- `temporal_tag`: same as `temporal`, but prefixes prompts/questions with `[YYYY-MM-DD]`

## HuggingFace token

This wrapper sources `~/.secrets/hf.env` if present.

Example setup:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets

cat > ~/.secrets/hf.env << 'ENV'
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
ENV

chmod 600 ~/.secrets/hf.env
```

## Running

### 1) Activate an env that can run AlphaEdit

AlphaEdit has its own dependency stack (torch/transformers/etc.). Use the same env that already runs:

```bash
cd AlphaEdit
python3 -m experiments.evaluate --help
```

### 2) Run all models × conditions

From `KnowledgeTimeline/news_alphaedit_eval`:

```bash
export NEWS_DATASET_ROOT=/N/project/cogllm/ztiganj/news_datast/outputs/news_dataset_v1

# Optional overrides
# export NUM_EDITS=10
# export DOWNSTREAM_EVAL_STEPS=5
# export DATASET_SIZE_LIMIT=0   # 0 => use full dataset
# export MOVE_RESULTS=0          # 1 => move AlphaEdit results out of AlphaEdit/results

./scripts/run_all_conditions.sh
```

Results are written under:

```
news_alphaedit_eval/runs/<condition>/<model_tag>/
```

The corresponding AlphaEdit run directory is copied/moved under the same folder.

### 3) SLURM

Edit `scripts/slurm_run_all_conditions.sbatch` for the cluster account/partition, then:

```bash
sbatch scripts/slurm_run_all_conditions.sbatch
```
