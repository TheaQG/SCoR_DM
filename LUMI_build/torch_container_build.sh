#!/bin/bash
#SBATCH --job-name=container_build
#SBATCH --output=cotainr.out
#SBATCH --error=cotainr.err
#SBATCH --account=project_465000956
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:45:00


# Setup software environment
module use /appl/local/training/modules/AI-20240529/
module load LUMI cotainr

# Point to the container
CONTAINER=/scratch/project_465000956/containers/images/my_torch_container_with_pandas.sif

# Point to the environment yml file
ENV_YML=/scratch/project_465000956/containers/build_files/torch_lumi_w_pandas.yml

srun cotainr build my_torch_container_with_pandas.sif --system=lumi-g --conda-env=/scratch/project_465000956/containers/build_files/torch_lumi_w_pandas.yml #cotainr build $CONTAINER --system=lumi-g --conda-env=$ENV_YML --accept-licenses