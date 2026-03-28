#!/bin/bash
#SBATCH --job-name=reliabilityDecoding_%A_%a
#SBATCH --output=/beegfs/u/bbc7806/logs/FineTuned-%j.out
#SBATCH --error=/beegfs/u/bbc7806/logs/FineTuned-%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1            # one job per ROI (adjust length)
#SBATCH --time=24:00:00        # <-- 3 days (D-HH:MM:SS)
#SBATCH --mail-type=NONE

# Submit the bash file: TOP_FRAC=0.30 sbatch AgingReplay_LocalizerDecodingReliability_submit.sh

# --- Load modules ---
module load conda/24.3.0
conda activate nipype_env

# --- Move to your working directory ---
cd /home/mpib/ren/rxj-neurocode/AgingStudy/AgingStudy-fMRI-data/fMRI-code

# --- Define ROI names (in bash array) ---
#ROIs=(VISventral VISlow MTL HPC ERH PFCdv PFCdorsoL PFCventroL ventricles)
ROIs=(HPC ERH)

# --- Select ROI based on array index ---
ROI_NAME=${ROIs[$SLURM_ARRAY_TASK_ID]}

# default if not provided
TOP_FRAC=${TOP_FRAC:-0.20}

echo "Reliability-based decoding for ROI: $ROI_NAME"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "top-frac: $TOP_FRAC"

# --- Run Python script with ROI name ---
python AgingReplay_LocalizerDecoding_EightCateory_ROILoop_HPC.py \
  --roi "$ROI_NAME" \
  --top-frac "$TOP_FRAC"
  
