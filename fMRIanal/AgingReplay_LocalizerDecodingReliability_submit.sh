#!/bin/bash
#SBATCH --job-name=reliabilityDecoding_%A_%a
#SBATCH --output=/home/mpib/ren/logs/reliabilityDecoding_%A_%a.out
#SBATCH --error=/home/mpib/ren/logs/reliabilityDecoding_%A_%a.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6            # one job per ROI (adjust length)
#SBATCH --time=24:00:00        # <-- 3 days (D-HH:MM:SS)
#SBATCH --mail-type=NONE

# Submit the bash file: TOP_FRAC=0.30 sbatch AgingReplay_LocalizerDecodingReliability_submit.sh

# --- Load modules ---
module load conda/24.3.0
conda activate nipype_env

# --- Move to your working directory ---
cd /home/mpib/ren/rxj-neurocode/AgingStudy/AgingStudy-fMRI-data/fMRI-code

# --- Define ROI names (in bash array) ---
ROIs=(VISventral VISlow MTL PFCdv PFCdorsoL PFCventroL ventricles)

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
  
