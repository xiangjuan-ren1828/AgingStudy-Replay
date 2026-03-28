#!/bin/bash
#SBATCH --job-name=SeqMemTask-pre
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=NONE
#SBATCH --output=/beegfs/u/bbc7806/logs/SeqMemTask-pre-%a-%j.out
#SBATCH --error=/beegfs/u/bbc7806/logs/SeqMemTask-pre-%a-%j.err
#SBATCH --array=0-110

# Subject list (111 subjects, indices 0-110)
SUBJECTS=(
  00 02 03 04 05 06 07 08 09
  11 12 13 14 15 16 17 18
  20 21 22 23 24 25 26 27 28 29
  30 31 32 33 34 35 36 37 38 39
  40 41 42 43 44 45 46 47 48 49
  50 51 52 53 54 55 56 57 58 59
  60 61 62 63 64 65 66 67 68 69
  70 71 72 73 74 75 76 77 78 79
  80 81 82 83 84 85 86 87 89 
  90 91 92 93 94 96 97 99
  100 101 102 103 104 105 106 107 108 109 
  110 111 112 113 114 115 116
)

SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Define paths
IMG=/usw/u/bbc7806/fmriprep-23.1.3.sif
BIDS_DIR=/beegfs/u/bbc7806/AgingStudy/AgingStudy-fMRI-data/fMRI-PreprocessedData
OUT_DIR=${BIDS_DIR}/sub-${SUBJ}/output
LICENSE_SRC=/beegfs/u/bbc7806/AgingStudy/AgingStudy-fMRI-data/license.txt
LICENSE_DEST=/opt/freesurfer/license.txt

# initialization
source /sw/batch/init.sh
module switch env apptainer/1.2.5

# Create output directory if it doesn't exist
mkdir -p ${OUT_DIR}

# Remove macOS metadata files that break BIDS indexing
find ${BIDS_DIR} -name "._*" -delete

# Run fMRIPrep via Apptainer
apptainer run --home /beegfs/u/bbc7806 --cleanenv \
  -B ${BIDS_DIR}:${BIDS_DIR} \
  -B ${OUT_DIR}:${OUT_DIR} \
  -B ${LICENSE_SRC}:${LICENSE_DEST} \
  ${IMG} \
  ${BIDS_DIR} ${OUT_DIR} participant \
  --participant-label ${SUBJ} \
  --fs-license-file ${LICENSE_DEST} \
  --mem_mb 50000 --nthreads 12 --omp-nthreads 8 \
  --output-spaces T1w MNI152NLin6Asym MNI152NLin2009cAsym fsnative fsaverage
