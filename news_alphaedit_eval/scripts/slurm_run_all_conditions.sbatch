#!/usr/bin/env bash
#SBATCH -p general
#SBATCH --job-name=AlphaEdit
#SBATCH --account=cogneuroai
#SBATCH --output=./logs/%j.txt
#SBATCH --error=./logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Adjust these on BigRed200 if needed:
#   #SBATCH -A <account>
#   #SBATCH -p <partition>

if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT/news_alphaedit_eval"

if [[ -f "$HOME/.secrets/hf.env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets/hf.env"
fi

# Activate env here if needed
# source /path/to/venv/bin/activate
# module load ...

./scripts/run_all_conditions.sh $1
