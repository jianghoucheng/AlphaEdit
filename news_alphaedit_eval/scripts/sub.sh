#!/bin/bash
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "gpt2-xl"
  "EleutherAI/gpt-j-6B"
)

for m in "${MODELS[@]}"; do
    echo $m
    sbatch slurm_run_all_conditions.sbatch $m
done