#!/bin/bash
set -euo pipefail

# ---------- if submit all participants together ----------
# bash AgingReplay_glm_mask_cvLOBO_singleSub_loop.sh

# ------ submit from the 4th participant ------
N_SUBS=110
START_IDX=3
LAST_IDX=$((N_SUBS-1))

echo "Submitting individual jobs from ${START_IDX} to ${LAST_IDX}"

for i in $(seq ${START_IDX} ${LAST_IDX}); do
  sbatch --export=SUBJ_IDX=${i} AgingReplay_glm_mask_cvLOBO_singleSub_submit.sbatch
done