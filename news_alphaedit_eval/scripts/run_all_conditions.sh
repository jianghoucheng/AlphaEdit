#!/usr/bin/env bash
set -euo pipefail



ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$ROOT_DIR/.." && pwd)

# Activate uv venv if it exists
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# HF token (optional)
if [[ -f "$HOME/.secrets/hf.env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets/hf.env"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

ALPHAEDIT_REPO=${ALPHAEDIT_REPO:-"$REPO_ROOT"}
NEWS_DATASET_ROOT=${NEWS_DATASET_ROOT:-"/u/demistry/news_datast/outputs/news_dataset_v3"}

WORK_DIR=${WORK_DIR:-"$ROOT_DIR/work"}
OUT_ROOT=${OUT_ROOT:-"$ROOT_DIR/runs"}

MOVE_RESULTS=${MOVE_RESULTS:-0}  # 1 => move AlphaEdit results out of AlphaEdit/results

DATASET_SIZE_LIMIT=${DATASET_SIZE_LIMIT:-0}   # 0 => full dataset
NUM_EDITS=${NUM_EDITS:-10}
DOWNSTREAM_EVAL_STEPS=${DOWNSTREAM_EVAL_STEPS:-5}
SEED=${SEED:-7}

# hparams files under AlphaEdit/hparams/AlphaEdit/
HPARAMS_LLAMA3=${HPARAMS_LLAMA3:-"Llama3-8B.json"}
HPARAMS_GPT2XL=${HPARAMS_GPT2XL:-"gpt2-xl.json"}
HPARAMS_GPTJ=${HPARAMS_GPTJ:-"EleutherAI_gpt-j-6B.json"}

# Models used in the AlphaEdit paper
# MODELS=(
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "gpt2-xl"
#   "EleutherAI/gpt-j-6B"
# )

MODELS=($1)


model_to_hparams() {
  local model_name="$1"
  case "$model_name" in
    meta-llama/Meta-Llama-3-8B-Instruct)
      echo "$HPARAMS_LLAMA3" ;;
    gpt2-xl)
      echo "$HPARAMS_GPT2XL" ;;
    EleutherAI/gpt-j-6B)
      echo "$HPARAMS_GPTJ" ;;
    *)
      echo "" ;;
  esac
}

model_to_tag() {
  local model_name="$1"
  case "$model_name" in
    meta-llama/Meta-Llama-3-8B-Instruct)
      echo "llama3" ;;
    gpt2-xl)
      echo "gpt2-xl" ;;
    EleutherAI/gpt-j-6B)
      echo "gpt-j" ;;
    *)
      echo "model" ;;
  esac
}

mkdir -p "$WORK_DIR" "$OUT_ROOT"

calc_size_limit() {
  local dataset_json="$1"
  local limit="$2"
  if [[ "$limit" -gt 0 ]]; then
    echo "$limit"
    return
  fi

  python3 - <<PY
import json
p = "$dataset_json"
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
PY
}

run_one() {
  local condition="$1"
  local order="$2"
  local temporal_tag="$3"
  local model_name="$4"

  local model_tag
  model_tag=$(model_to_tag "$model_name")

  local hparams
  hparams=$(model_to_hparams "$model_name")
  if [[ -z "$hparams" ]]; then
    echo "Unknown model: $model_name" >&2
    exit 1
  fi

  local ds_json="$WORK_DIR/news_${condition}.json"
  local meta_jsonl="$WORK_DIR/news_${condition}.meta.jsonl"

  echo "==> Build dataset: condition=$condition order=$order temporal_tag=$temporal_tag"
  build_args=(
    --news_dataset_root "$NEWS_DATASET_ROOT"
    --out_json "$ds_json"
    --out_meta_jsonl "$meta_jsonl"
    --order "$order"
    --seed "$SEED"
    --temporal_tag "$temporal_tag"
  )
  if [[ "$DATASET_SIZE_LIMIT" -gt 0 ]]; then
    build_args+=(--limit "$DATASET_SIZE_LIMIT")
  fi
  python3 -m news_alphaedit_eval.make_alphaedit_dataset "${build_args[@]}"

  echo "==> Install dataset into AlphaEdit (ds_name=mcf)"
  python3 -m news_alphaedit_eval.install_into_alphaedit \
    --alphaedit_repo "$ALPHAEDIT_REPO" \
    --dataset_json "$ds_json" \
    --ds_name "mcf"

  local effective_limit
  effective_limit=$(calc_size_limit "$ds_json" "$DATASET_SIZE_LIMIT")

  local out_dir="$OUT_ROOT/$condition/$model_tag"
  mkdir -p "$out_dir"
  cp -f "$ds_json" "$out_dir/dataset_for_alphaedit.json"
  cp -f "$meta_jsonl" "$out_dir/news_meta.jsonl"

  local log_file="$out_dir/alphaedit_eval.log"

  echo "==> Run AlphaEdit: model=$model_name hparams=$hparams dataset_size_limit=$effective_limit"
  run_args=(
    --alphaedit_repo "$ALPHAEDIT_REPO"
    --alg_name "AlphaEdit"
    --model_name "$model_name"
    --hparams_fname "$hparams"
    --ds_name "mcf"
    --dataset_size_limit "$effective_limit"
    --num_edits "$NUM_EDITS"
    --downstream_eval_steps "$DOWNSTREAM_EVAL_STEPS"
    --out_root "$out_dir"
    --log_file "$log_file"
  )
  if [[ "$MOVE_RESULTS" == "1" ]]; then
    run_args+=(--move)
  fi

  python3 -m news_alphaedit_eval.run_alphaedit_eval \
    "${run_args[@]}"

  echo "==> Done: condition=$condition model=$model_name"
}

CONDITIONS=(
  "temporal temporal none"
  "random random none"
  "temporal_tag temporal bracket_date"
)

for model_name in "${MODELS[@]}"; do
  for triple in "${CONDITIONS[@]}"; do
    # shellcheck disable=SC2206
    parts=($triple)
    condition=${parts[0]}
    order=${parts[1]}
    temporal_tag=${parts[2]}

    run_one "$condition" "$order" "$temporal_tag" "$model_name"
  done
done

echo "All runs finished. Output: $OUT_ROOT"
