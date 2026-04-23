#!/bin/bash
set -euo pipefail

CONDA_PATH="/beegfs/g/schuck/bbc7806/miniconda3"
BASE_DIR="/beegfs/g/schuck/bbc7806"
ENV_PATH="${BASE_DIR}/envs/DecodingReplay_env"

source "${CONDA_PATH}/etc/profile.d/conda.sh"

export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export CONDARC="${BASE_DIR}/.condarc"
export CONDA_PKGS_DIRS="${BASE_DIR}/pkgs"
export CONDA_ENVS_PATH="${BASE_DIR}/envs"
export XDG_CACHE_HOME="${BASE_DIR}/.cache"
export XDG_CONFIG_HOME="${BASE_DIR}/.config"

mkdir -p "${BASE_DIR}/pkgs" "${BASE_DIR}/envs" "${BASE_DIR}/.cache" "${BASE_DIR}/.config"

conda create -y \
  --solver=classic \
  --override-channels -c conda-forge \
  -p "${ENV_PATH}" \
  python=3.10

conda activate "${ENV_PATH}"

conda install -y \
  --solver=classic \
  --override-channels -c conda-forge \
  numpy scipy pandas seaborn matplotlib scikit-learn nilearn nibabel

python - <<'PY'
import numpy, scipy, seaborn, pandas, sklearn, nilearn, nibabel
print("All imports OK")
PY