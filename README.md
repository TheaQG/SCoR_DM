# CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall

> **Status:** Research code accompanying an ongoing PhD project.
> The repository is stable enough to reproduce main experiments and results, but may undergo further refinement and development.

---

## 1. Overview
### **What is CEDDAR?**
  CEDDAR (Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall) is a diffusion-based statistical downscaling framework designed to enhance the spatial resolution of precipitation data - and in the future other climate variables(/full regional climate emulation). 
  It is built on the Elucidated Diffusion Model (EDM) setup and currently targets **precipitation** over Denmark, with ongoing work extending the framework to other variables and regions.

### **Problem setting**
Global reanalysis products such as ERA5 [REFERENCE] provide large-scale atmospheric dynamic information but at spatial resolutions (~31 km at DK latitude) that are insufficient for local-scale climate impact modelling.
High-resolution reanalyses like DANRA [REFERENCE] (2.5 km) offer more detailed spatial structure, but are costly to produce and not globally available, while only being single-realization datasets (no ensembles).
CEDDAR aims to bridge this resolution gap by learning a generative mapping from Low-Resolution (LR) ERA5 precipitation data to High-Resolution (HR) DANRA-like fields while attempting to capture (1) fineScale spatial variability and structures, (2) extreme-event behaviour, (3) multi-scale physical structure, and (4) uncertainty quantification through daily ensembles.

### **Key ideas**
- Elucidated Diffusion Model (EDM) based generative modelling for high-resolution precipitation fields
- Residual learning framework: model learns to generate the residual between bilinearly upsampled LR input and HR target, improving learning efficiency
- Multi-conditional inputs: geographical (topography, land-sea mask), temporal (day-of-year, time-of-day), and climatic (large-scale variables) conditions (*ongoing*)
- Scale-controllable sampling via $\sigma$* (sigma-star) parameter, allowing control over power spectrum characteristics of generated fields at inference time
- Ensemble generation for uncertainty quantification, enabling multiple plausible HR realizations for a given LR input
- Comprehensive evaluation metrics suite covering distributional characteristics, extreme event statistics, spatial metrics, power spectra and temporal behaviour

## 2. Features

### **Core Model**
    - [x] Elucidated Diffusion Model (EDM) for spatial precipitation downscaling
    - [x] Residual learning formulation
    - [x] Multi-conditional inputs (geographical, temporal, climatic)
    - [x] Scale-controllable sampling via $\sigma$* (sigma-star) for power spectrum control
    - [x] Ensemble approach for uncertainty quantification
    - [x] RainGate module for drizzle reduction and precipitation correction

### **Evaluation Suite**
    - [x] Distributional metrics (PDF, CDF, quantiles, pixel statistics)
    - [x] Extreme event statistics (RX1day, RX5day, R95p, R99p, (POT, GPD fitting) )
    - [x] Spatial metrics (FSS, ISS, spatial mean/std maps, bias ratios)
    - [x] Power spectrum analysis 
    - [x] Temporal consistency metrics 
    - [x] Automated evaluation pipeline with visualizations

### **Data and (pre)processing pipelines**
    - [x] ERA5 and DANRA preprocessing scripts (CDO-based)
    - [x] Daily .nc/.npz -> .zarr conversion
    - [x] Train/valid/test split creation
    - [x] Statistics computation 
    - [x] Correlation analysis across HR/LR variables

### **Baseline Methods for comparison**
    - [x] Bilinear upsampling
    - [x] Quantile mapping (QM)
    - [ ] UNet-SR baseline (ongoing)
    - [x] Baseline evaluation matching the full CEDDAR suite

## 3. Repository Structure
High-level directory overview:
```
.
├── baselines/                  # Baseline downscaling methods and evaluations
├── bash_scripts/               # Example Slurm/helper scripts (LUMI/HPC)
├── ceddar/                     # Core CEDDAR model, training, generation, evaluation code
├── data_analysis_pipeline/     # Statistics, comparisons, correlations, preprocessing, splits
├── data_examples/              # Tiny example data (ERA5/DANRA)
├── era5_download_pipeline/     # ERA5 data download + CDO-based preprocessing
├── models_and_samples/         # Pretrained models, sample outputs (optional, may be empty)
├── notebooks/                  # Jupyter notebooks for exploratory analysis
├── requirements.txt            # Minimal Python dependencies
└── README.md                   # This file
```

### Component descriptions

#### **`ceddar/` - Core model**
- Diffusion model implementation (EDM)
- UNet-based score model (`score_unet.py`)
- EDM sampler and $\sigma$* control (`score_sampling.py`)
- Small data utilities (`data/create_train_valid_test.py`, `data/datasets_loading.py`)
- Training pipeline (`training_main.py`, `training.py`, `training_utils.py`)
- Generation tools (`generate/`)
- Evaluation tools (`evaluate/`)
- Special modules (e.g., RainGate in `heads/`, transformations in `special_transforms.py`)
- Utilities (`utils.py`, `logging_utils.py`, `scale_utils.py`)
- Unified controller script in `cli/` controlled through .yaml config files in `configs/`

#### **`baselines/`**
- Bilinear upsampling baseline (`bilinear.py`)
- Quantile mapping baseline (`quantile_mapping.py`)
- UNet-SR (WIP) baseline (`unet_sr/`)
- Baseline evaluation scripts (`baseline_eval.py`, `plotting.py`)
- Helpers for baseline data handling (`build_dataset.py`, `adapter.py`)
- Controlled through ´baseline_main.py` and .yaml config files in ceddar/configs/

#### **`era5_download_pipeline/`**
- ERA5 (single-level and pressure-level) data download scripts using CDS API [REFERENCE TO CDS API] (`pipeline/`)
  - Necessary CDO commands for initial preprocessing (regridding, variable extraction, unit conversions) (`cdo_utils.py`)
  - ERA5 data request script in `download.py`
  - Utilities for organizing and managing downloaded data both locally and on HPC storage system (LUMI) (`remote_utils.py`, `utils.py`)
  - Transfer script for moving data between local and HPC storage (`transfer.py`)
  - Pipeline setup for streamlined data download and preprocessing (`stream.py`)
- Controller script in `cli/` controlled through .yaml files in `cfg/`
- Slurm job scripts for HPC execution in `slurm/`
- Logging and log utilities in `utils/` and `era5_logs/`

#### **`data_analysis_pipeline/`**
- Data preprocessing/preparation and filtering scripts (`preprocess/`)
  - Daily .nc/.npz to .zarr conversion (`daily_files_to_zarr.py`)
  - Data filtering (`filter_data.py`)
  - Small dataset creation for testing (`create_small_dataset.py`)
- Train/valid/test split generation (`splits/`)
- Dataset comparisons (between variables both in ERA5 and DANRA) (`comparisons/`)
- Correlation analysis, temporal and spatial, across HR/LR variables (`correlations/`)
- Statistics computation (`statistics/`)
- Controller scipts in `cli/`controlled through .yaml files in `configs/`

#### **`data_examples/`**
- Minimal example of DANRA and ERA5 files (in final .zarr format and raw .nc/.npz formats) for quick testing and debugging
- Demonstration of model input/output shapes and data structures

#### **`models_and_samples/`**
- Pretrained CEDDAR model checkpoints (optional, may be empty)
- Sample generated outputs/evaluations from pretrained models for quick visualization and inspection
- Organized by experiment/configuration names

#### **`bash_scripts/`**
- LUMI Slurm examples for training, generation, evaluation, data download 

## 4. Requirements

CEDDAR uses a minimal, platform-agnostic Python environment.

```
pip install -r requirements.txt
```

A CPU-only installation works for basic testing. 
Full training and generation require a CUDA-capable GPU (AMD or NVIDIA) with sufficient VRAM. See `bash_scripts/` for example Slurm scripts for LUMI/HPC setups and container-based workflows.



## 5. Installation

CEDDAR is designed to run on both standard local machines (for development and basic testing) and on HPC environments such as **LUMI** for full-scale training and generation.

### 5.1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CEDDAR.git
cd CEDDAR
```

### 5.2. Python Environment Setup (Local)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

This installs a minimal CPU-capable environment suitable for:
- code inspection,
- running preprocessing scripts,
- running the evaluation pipeline on stored results,
- debugging model components.

GPU acceleration is required for:
- training,
- generation,
- σ\*-controlled sampling,
- large-scale evaluation.

### 5.3. HPC / LUMI Installation

CEDDAR is typically executed on **LUMI-G** using:
- a **Singularity/Apptainer container** with ROCm+PyTorch installed, and  
- Slurm batch submission scripts.

The repository includes example scripts in:

```
bash_scripts/
```

Typical workflow on LUMI:

```bash
# Load appropriate modules
module load LUMI/22.12 partition/G

# Build or load container
bash bash_scripts/build_container.sh

# Run interactive shell
srun --account=<project> --partition=standard-g --gres=gpu:1 \
     --container-image=<container_path> bash
```

Training, generation, and evaluation commands inside the container  
are identical to local usage (see Section 7).

**Note:**  
Because LUMI uses AMD MI250X GPUs, PyTorch must be ROCm-compatible.  
The container handles this automatically.

---

## 6. Data

CEDDAR is trained on a combination of:

- **ERA5** (coarse-resolution, ~31 km)
- **DANRA** (high-resolution reanalysis over Denmark, ~2.5 km)

Due to licensing and size constraints, **full datasets are not included**.  
Instead, small example files are provided in:

```
data_examples/
```

### 6.1. Required Variables

#### ERA5 variables (typical configuration)
- Total precipitation (tp)
- 2m temperature (t2m)
- Mean sea level pressure (msl)
- CAPE (cape)
- Optional: pressure-level variables  
  (e.g. geopotential, humidity, wind components at 500/700/850 hPa)

#### DANRA variables
- Precipitation (`pr`)
- Temperature (`t2m`) (optional future work)
- Potential evaporation (`pev`) (optional future work)

### 6.2. File Structure Expected by CEDDAR

CEDDAR expects **Zarr-based datasets**, generated via the preprocessing pipeline:

```
/path/to/data/
    ERA5/
        zarr/
            prcp_ERA5.zarr/
            t2m_ERA5.zarr/
            ...
    DANRA/
        zarr/
            prcp_DANRA.zarr/
            ...
    splits/
        train_dates.txt
        valid_dates.txt
        test_dates.txt
```

The exact folder structure is controlled by YAML config files in:

```
data_analysis_pipeline/configs/
```

### 6.3. Preprocessing Workflow

All preprocessing actions are executed through the Data Analysis CLI:

- Convert `.nc` or `.npz` → Zarr  
- Compute daily aggregates  
- Filter invalid days  
- Construct train/valid/test splits  
- Compute statistics used for normalisation during training  
- Perform correlation analyses (optional)

A typical preprocessing command is:

```bash
python -m data_analysis_pipeline.cli.main_data_app \
    --config data_analysis_pipeline/configs/preprocess_default.yaml \
    --task preprocess
```

More examples are provided in `bash_scripts/` and in the config folder.

---

## 7. Usage

CEDDAR follows a configuration-driven design.  
All workflows (training, generation, evaluation) use YAML config files located in:

```
ceddar/configs/
```

Individual experiments are launched through the unified CLI in:

```
ceddar/cli/
```

---

### 7.1. Training CEDDAR

A typical training command:

```bash
python -m ceddar.cli.main_app \
    --config ceddar/configs/train_prcp_default.yaml \
    --mode train
```

This will:
- load ERA5+HR datasets through the configured data module,
- build the EDM+UNet architecture specified in the config,
- start logging and checkpointing,
- train until convergence.

Logs and checkpoints are saved under:

```
models_and_samples/
    <experiment-name>/
        checkpoints/
        logs/
        samples/
```

---

### 7.2. Generating Downscaled Samples

Generation uses the **same config**, but switches mode:

```bash
python -m ceddar.cli.main_app \
    --config ceddar/configs/train_prcp_default.yaml \
    --mode generate
```

Or a dedicated generation config:

```bash
python -m ceddar.cli.main_app \
    --config ceddar/configs/generate_prcp_default.yaml \
    --mode generate
```

Supports:
- multiple σ\* values (scale-control sweeps),
- ensemble generation,
- residual + absolute output modes,
- optional RainGate correction.

Generated samples are stored in:

```
models_and_samples/<experiment>/generated/
```

---

### 7.3. Evaluation Pipeline

Evaluation is fully modular and covers:
- distributional metrics,
- extremes (POT/GPD),
- spatial statistics,
- power spectra (PSD),
- correlation and temporal behaviour,
- visual diagnostics.

Run evaluation:

```bash
python -m ceddar.cli.main_app \
    --config ceddar/configs/eval_prcp_default.yaml \
    --mode evaluate
```

Each evaluation module writes results into:

```
models_and_samples/<experiment>/evaluation/
    metrics/
    figures/
    spectra/
    extremes/
```

---

### 7.4. Baselines

Baselines are run through:

```bash
python baselines/baseline_main.py \
    --config ceddar/configs/baseline_prcp_default.yaml
```

Produces:
- bilinear upsampling outputs,
- quantile-mapped fields,
- UNet-SR predictions (if enabled),
- full baseline evaluation suite.

Outputs follow the same directory convention as CEDDAR models.

---## 8. Citation