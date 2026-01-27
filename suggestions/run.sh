#!/bin/bash
#SBATCH -p general
#SBATCH --job-name=AlphaEdit
#SBATCH --account=cogneuroai
#SBATCH --output=./logs/%j.txt
#SBATCH --error=./logs/%j.err
#SBATCH --mem=64G
#SBATCH --tasks-per-node=10
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=36:00:00

set -eou pipefail

source /u/demistry/AlphaEdit/.venv/bin/activate

python -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=2000 \
    --num_edits=100 \
    --downstream_eval_steps=5