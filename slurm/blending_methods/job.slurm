#!/bin/bash -xl
#
#SBATCH --gres=gpu::1
#SBATCH --time=12:00:00
#SBATCH --output=./slurm/default/logs/%x.%j.log


unset SLURM_EXPORT_ENV

export https_proxy=http://proxy:80
export http_proxy=http://proxy:80

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

module load python/3.12-conda
module load cuda/12.6

conda activate draem

# First argument should be comma-separated list of object IDs
OBJ_IDS=$1
shift  # Remove first argument so remaining args can be passed through

# Split comma-separated list and spawn a process for each
IFS=',' read -ra OBJ_ARRAY <<< "$OBJ_IDS"
for obj_id in "${OBJ_ARRAY[@]}"; do
    echo "Starting training for object ID: $obj_id"
    python3 "$SLURM_SUBMIT_DIR/train_DRAEM.py" --obj_id "$obj_id" "$@" --data_path "$SLURM_SUBMIT_DIR/datasets/mvtec/" --anomaly_source_path "$SLURM_SUBMIT_DIR/datasets/dtd/images/" --checkpoint_path "$SLURM_SUBMIT_DIR/checkpoints/" --log_path "$SLURM_SUBMIT_DIR/logs/" &
done

# Wait for all background processes to complete
wait

echo "All training jobs completed!"
