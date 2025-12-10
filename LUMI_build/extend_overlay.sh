#!/bin/bash
#SBATCH --job-name=extend_overlay
#SBATCH --account=project_465001695
#SBATCH --output=err_out/extend_overlay_%j.out
#SBATCH --error=err_out/extend_overlay_%j.err
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

# -- 0. Load modules --------------------------------------------------------------
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# -- 1. Paths and variables -------------------------------------------------------
CONTAINER=/scratch/project_465001695/containers/images/my_torch_container_with_plotting.sif
OVERLAY_DIR=/scratch/project_465001695/containers/overlays
OVERLAY_IMG=$OVERLAY_DIR/my_overlay.img
OVERLAY_SIZE_MB=5000

mkdir -p "$OVERLAY_DIR"
if [ ! -f "$OVERLAY_IMG" ]; then
  echo "Creating overlay image at $OVERLAY_IMG with size $OVERLAY_SIZE_MB MB..."
  singularity overlay create --size "$OVERLAY_SIZE_MB" "$OVERLAY_IMG"
else
  echo "Re-using existing overlay image at $OVERLAY_IMG."
fi

# -- 2. Install into overlay (RW) --------------------------------------------------
echo "Installing CDO and dependencies into overlay at $OVERLAY_IMG..."
singularity exec --overlay "$OVERLAY_IMG":rw "$CONTAINER" bash -eu <<EOF
  set -o pipefail

  # 2.1 bootstrap micromamba once
  if ! command -v micromamba &>/dev/null; then
      echo "[bootstrap] Micromamba not found, installing..."
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest |
          tar -xvj bin/micromamba
      mkdir -p \$HOME/micromamba/bin
      mv bin/micromamba \$HOME/micromamba/bin/
  fi
  export PATH=$HOME/micromamba/bin:$PATH
  export MAMBA_ROOT_PREFIX=$HOME/micromamba

  # 2.2 Ensure coreutils present
  micromamba install -y -n base -c conda-forge coreutils 

  # 2.3 Create or update env 'era5' with CDO + netCDF4
  echo "[bootstrap] Creating or updating the 'era5' environment..."
  micromamba create -y -n era5 -c conda-forge python=3.10 cdo netcdf4 
  echo "[bootstrap] Environment 'era5' created or updated successfully."
  
  # 3) Extra python wheels inside that env
  micromamba run -n era5 python -m pip install --upgrade pip scikit-learn omegaconf
  echo "[bootstrap] Pip packages upgraded successfully."
EOF

# -- 3. Finalize overlay ----------------------------------------------------------
echo "Overlay now contains CDO.  Current size:"
du -h "$OVERLAY_IMG"