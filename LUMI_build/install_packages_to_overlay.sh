#!/bin/bash
#SBATCH --job-name=install_pkgs
#SBATCH --account=project_465001695
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=install_pkgs_%j.out
#SBATCH --error=install_pkgs_%j.err

# === Load Singularity Environment ===
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Define paths ===
CONTAINER="/scratch/project_465001695/containers/images/my_torch_container_with_plotting.sif"
OVERLAY_DIR="/scratch/project_465001695/containers/overlays"
OVERLAY_IMG="$OVERLAY_DIR/my_overlay.img"
OVERLAY_SIZE_MB=5000

# === Create overlay if it doesn't exist ===
mkdir -p "$OVERLAY_DIR"
if [ ! -f "$OVERLAY_IMG" ]; then
    echo "Creating overlay image..."
    singularity overlay create --size $OVERLAY_SIZE_MB "$OVERLAY_IMG"
else
    echo "Overlay already exists."
fi

# === 3. Install packages in the container using the overlay ===
singularity exec --overlay "$OVERLAY_IMG":rw "$CONTAINER" bash -c "
    set -e 
    pip install --upgrade pip && pip install scikit-learn omegaconf"