#!/bin/bash
set -euo pipefail

# ------ submit from the 4th participant ------
N_SUBS=110
START_IDX=3
LAST_IDX=$((N_SUBS-1))

# ------ Max number of simultaneous running tasks (tune this!) ------
MAX_CONCURRENT=5

# Memory per task (tune this!)
MEM_PER_TASK="24G"

echo "Submitting array ${START_IDX}-${LAST_IDX} with max ${MAX_CONCURRENT} concurrent tasks, mem=${MEM_PER_TASK}"

sbatch \
  --array="${START_IDX}-${LAST_IDX}%${MAX_CONCURRENT}" \
  --mem="${MEM_PER_TASK}" \
  AgingReplay_glm_mask_cvLOBO_singleSub_submit.sbatch


# ---------- The following command has memory issue if running simultaneously ----------
# ---------- if submit all participants together ----------
# bash AgingReplay_glm_mask_cvLOBO_singleSub_loop.sh

# ------ submit from the 4th participant ------
# N_SUBS=110
# START_IDX=3
# LAST_IDX=$((N_SUBS-1))

# echo "Submitting individual jobs from ${START_IDX} to ${LAST_IDX}"

# for i in $(seq ${START_IDX} ${LAST_IDX}); do
#   sbatch --export=SUBJ_IDX=${i} AgingReplay_glm_mask_cvLOBO_singleSub_submit.sbatch
# done