#!/bin/bash
#SBATCH --job-name=ROIExtractSMT
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=NONE
#SBATCH --output=/beegfs/u/bbc7806/logs/ROIExtractSMT-%a-%j.out
#SBATCH --error=/beegfs/u/bbc7806/logs/ROIExtractSMT-%a-%j.err
#SBATCH --time=24:00:00

# ------ Load conda base environment ------
CONDA_PATH="/beegfs/g/schuck/bbc7806/miniconda3"
BASE_DIR="/beegfs/g/schuck/bbc7806"
ENV_PATH="${BASE_DIR}/envs/DecodingReplay_env"
source "${CONDA_PATH}/etc/profile.d/conda.sh"

# ------ Activate the corresponding conda environment ------
conda activate "${ENV_PATH}"

# --- Move to your working directory ---
cd /beegfs/u/bbc7806/AgingStudy/AgingStudy-fMRI-data/fMRI-code

# --- Run Python script with ROI name ---
python localizer_maskROI_SMT.py
