#!/bin/bash
# Submit fMRIPrep jobs in batches of 20, each batch waits for the previous to finish.
# Usage: bash submit_batches.sh

SCRIPT=SeqMemTask_fmriprep_hummel.sh

# Define batches as "start-end" index ranges (total 111 subjects, indices 0-110)
BATCHES=(
  "0-19"
  "20-39"
  "40-59"
  "60-79"
  "80-99"
  "100-110"
)

PREV_JOB_ID=""

for BATCH in "${BATCHES[@]}"; do
  if [ -z "$PREV_JOB_ID" ]; then
    # Submit first batch with no dependency
    JOB_ID=$(sbatch --array=${BATCH} --parsable ${SCRIPT})
  else
    # Submit subsequent batches only after previous batch fully completes
    JOB_ID=$(sbatch --array=${BATCH} --dependency=afterok:${PREV_JOB_ID} --parsable ${SCRIPT})
  fi
  echo "Submitted batch ${BATCH} as job ${JOB_ID}"
  PREV_JOB_ID=${JOB_ID}
done
