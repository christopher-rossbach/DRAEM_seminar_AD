#!/bin/bash

# Grid search parameters
BATCH_SIZES=(8)
LRS=(0.0001)
EPOCHS=(700)
OBJ_IDS=("0,0" "1,1" "2,2" "3,3" "4,4" "5,5" "6,6" "7,7" "8,8" "9,9" "10,10" "11,11" "12,12" "13,13" "14,14")
EXTRA_TAGS="blending_methods"
EXTRA_FLAGS=("--amp")
LR_SCHEDULERS=("multi_step_0.1_0.572_0.858")
BLEND_METHODS=("blurred_beta" "perlin_beta" "texture_beta" "uniform_beta" "poisson")

# Parse arguments
DRY_RUN=0
for arg in "$@"; do
	if [[ "$arg" == "--dry-run" ]]; then
		DRY_RUN=1
	fi
done

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Calculate total number of jobs
NUM_JOBS=$(( ${#BATCH_SIZES[@]} * ${#LRS[@]} * ${#EPOCHS[@]} * ${#OBJ_IDS[@]} * ${#EXTRA_FLAGS[@]} * ${#LR_SCHEDULERS[@]} * ${#BLEND_METHODS[@]} ))

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
				for EXTRA_FLAG in "${EXTRA_FLAGS[@]}"; do
					for LR_SCHEDULER in "${LR_SCHEDULERS[@]}"; do
						for BLEND_METHOD in "${BLEND_METHODS[@]}"; do
							echo "sbatch job.slurm $OBJ_ID --bs $BATCH_SIZE --epochs $EPOCH --lr $LR --lr_scheduler $LR_SCHEDULER $EXTRA_FLAG --extra_tags \"$EXTRA_TAGS\" --blend_method $BLEND_METHOD"
							if [ "$DRY_RUN" -eq 0 ]; then
								sbatch.tinygpu $DIR/job.slurm $OBJ_ID --bs $BATCH_SIZE --epochs $EPOCH --lr $LR --lr_scheduler $LR_SCHEDULER $EXTRA_FLAG --extra_tags "$EXTRA_TAGS" --blend_method $BLEND_METHOD
							fi
						done
					done
				done
			done
		done
	done
done

if [ "$DRY_RUN" -eq 1 ]; then
	echo "Dry run complete. No jobs were submitted."
else
	echo "All $NUM_JOBS jobs submitted successfully!"
fi