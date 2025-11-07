#!/bin/bash -xl
#
#SBATCH --gres=gpu::1
#SBATCH --time=01:00:00
#SBATCH --output=./slurm/train_victim/logs/%x.%j.log


unset SLURM_EXPORT_ENV

export https_proxy=http://proxy:80
export http_proxy=http://proxy:80

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

module load python/3.12-conda
module load cuda/12.6

conda activate draem
python3 "$SLURM_SUBMIT_DIR/train_DRAEM.py" "${@:1}" --data_path "$SLURM_SUBMIT_DIR/datasets/mvtec/" --anomaly_source_path "$SLURM_SUBMIT_DIR/datasets/dtd/images/" --checkpoint_path "$SLURM_SUBMIT_DIR/checkpoints/" --log_path "$SLURM_SUBMIT_DIR/logs/"