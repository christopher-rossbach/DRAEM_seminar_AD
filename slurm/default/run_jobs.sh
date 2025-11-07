#!/bin/bash
# Auto-generated grid search for train_victim.py parameters
# --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/

# Grid search parameters
BATCH_SIZES=(8)
LRS=(0.0001)
EPOCHS=(700)
OBJ_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)
#OBJ_IDS=(0)

# Parse arguments
DRY_RUN=0
for arg in "$@"; do
	if [[ "$arg" == "--dry-run" ]]; then
		DRY_RUN=1
	fi
done

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Calculate total number of jobs
NUM_JOBS=$(( ${#BATCH_SIZES[@]} * ${#LRS[@]} * ${#EPOCHS[@]} * ${#OBJ_IDS[@]} ))

echo "Number of jobs to start: $NUM_JOBS"
if [ "$NUM_JOBS" -gt 20 ]; then
	read -p "Are you sure you want to start $NUM_JOBS
 jobs? (y/N): " CONFIRM
	if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
		echo "Aborting."
		exit 1
	fi
fi

# Create logs directory if it doesn't exist
mkdir -p "$DIR/logs"


# Grid search loop
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
	for LR in "${LRS[@]}"; do
		for EPOCH in "${EPOCHS[@]}"; do
			for OBJ_ID in "${OBJ_IDS[@]}"; do
				echo "sbatch job.slurm --bs $BATCH_SIZE --epochs $EPOCH --lr $LR --obj_id $OBJ_ID"
				if [ "$DRY_RUN" -eq 0 ]; then
					sbatch.tinygpu $DIR/job.slurm --bs $BATCH_SIZE --epochs $EPOCH --lr $LR --obj_id $OBJ_ID
				fi
			done
		done
	done
done

if [ "$DRY_RUN" -eq 1 ]; then
	echo "Dry run complete. No jobs were submitted."
else
	echo "All $NUM_JOBS jobs submitted successfully!"
fi