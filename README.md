# CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall

> **Status:** Research code accompanying an ongoing PhD project.
> The repository is stable enough to reproduce main experiments and results, but may undergo further refinement and development.

---

## 1. Overview
![alt text](introduction_long.png)
### **What is CEDDAR?**
  CEDDAR (Controllable Ensemble Difxfusion Downscaling for Atmospheric Rainfall) is a diffusion-based statistical downscaling framework designed to enhance the spatial resolution of precipitation data - and in the future other climate variables(/full regional climate emulation). 
  It is built on the Elucidated Diffusion Model (EDM) setup and currently targets **precipitation** over Denmark, with ongoing work extending the framework to other variables and regions.
  **KNOWN CURRENT LIMITATION**: The current precipitation-to-precipitation model exhibits a significant dry-bias in yearly totals, which is actively being investigated.

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
![alt text](Dates_final.png)

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
![alt text](preprocessing_pipeline.png)
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
├── Data_DiffMod_small/         # Tiny example data (ERA5/DANRA)
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

#### **`Data_DiffMod_small/`**
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
Data_DiffMod_small/
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
.
└── Data_DiffMod_small/
    ├── data_DANRA/
    │   └── size_589x789/
    │       └── prcp_589x789/
    │           ├── all/
    │           │   ├── tp_tot_19910101.npz
    │           │   ├── ...
    │           │   └── tp_tot_20201231.npz
    │           └── zarr_files/
    │               ├── train.zarr/
    │               │   ├── tp_tot_19910101
    │               │   ├── ...
    │               │   └── tp_tot_20151231
    │               ├── valid.zarr/
    │               │   ├── tp_tot_20160101
    │               │   ├── ...
    │               │   └── tp_tot_20181231
    │               └── test.zarr/
    │                   ├── tp_tot_20190101
    │                   ├── ...
    │                   └── tp_tot_20201231
    ├── data_ERA5/
    │   └── size_589x789/
    │       └── prcp_589x789/
    │           ├── all/
    │           │   ├── tp_589x789_19910101.npz
    │           │   ├── ...
    │           │   └── tp_589x789_20201231.npz
    │           └── zarr_files/
    │               ├── train.zarr/
    │               │   ├── tp_tot_19910101
    │               │   ├── ...
    │               │   └── tp_tot_20151231
    │               ├── valid.zarr/
    │               │   ├── tp_tot_20160101
    │               │   ├── ...
    │               │   └── tp_tot_20181231
    │               └── test.zarr/
    │                   ├── tp_589x789_20190101
    │                   ├── ...
    │                   └── tp_589x789_20201231
    ├── data_lsm/
    │   ├── truth_DK/
    │   │   └── lsm_dk.npz/
    │   └── truth_fullDomain/
    │       └── lsm_full.npz/
    └── data_topo/
        ├── truth_DK/
        │   └── topo_dk.npz/
        └── truth_fullDomain/
            └── topo_full.npz/
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

## 8. Configuration files and setup
CEDDAR is fully configuration-driven and all workfloes - training, generation, evaluation - are controlled through YAML config files located in:
```
ceddar/configs/
data_analysis_pipeline/configs/
era5_download_pipeline/cfg/
```
### 8.1 Structure of training/generation/evaluation configs
A typical training/generation/evaluation config includes:
- **model settings**: UNet depth, multipliers, attention, activation, conditioning variables
- **diffusion settings**: EDM parameters (sigma_min, sigma_max, sigma_data, sampling schedule)
- **data settings**: paths to LR/HR datasets, HR/LR variable selections, normalisation strategies and stats, residual vs absolute learning
- **training settings**: batch size, learning rate, optimizer, number of steps, logging, mixed precision, epochs
- **logging/checkpointing**: output directory, save frequency, experiment naming
- **generation settings**: number of samples, sigma-star values, ensemble size, RainGate usage
- **evaluation settings**: which metrics to compute, various thresholds, plotting options

### 8.2 Overriding config options
All parameters may be overridden from the Command Line Interface (CLI):
```bash
python -m ceddar.cli.main_app \
    --config ceddar/configs/train_prcp_default.yaml \
    --mode train \
    data.batch_size=16 \
    model.unet.attention=True \
```

## 9. Reproducibility
Reproducibility is a core design objective for CEDDAR. 

### 9.1 Practices for reproducibility
To ensure consistent results across runs and environments, CEDDAR incorporates the following practices:
- **Fixed random seeds**: All random number generators (PyTorch, NumPy, Python `random`) are seeded at the start of each run.
- **Deterministic operations**: PyTorch is configured to use deterministic algorithms where possible.
- **Environment logging**: The exact versions of all dependencies are logged at the start of each run.
- **Checkpointing**: Model checkpoints are saved regularly during training, allowing for exact resumption of experiments.
- **Config versioning**: All experiments are controlled through versioned YAML config files, ensuring that the exact settings used can be retrieved later.
- **Containerization**: For HPC environments, CEDDAR is designed to run within Singularity/Apptainer containers, encapsulating the entire software environment.
- **Self-contained experiments**: Each experiment creates a fully self-contained output directory with logs, checkpoints, samples, and evaluation results under ```models_and_samples/<experiment-name>/```. This contains:
  - Full config file snapshot
  - Model checkpoints
  - Training logs and samples
  - Generated samples
  - Evaluation metrics and figures


### 9.2 Environment snapshots
For scientific reproducibility, we recommend `requirements.txt` for general (local) setups, and `requirements_lumi.txt` (or container recipe) for exact HPC/LUMI setups.

### 9.3 Data-version reproducibility

## 10. Limitations and Future Work
CEDDAR is active research code and has several known limitations:

### 10.1 Current Limitations
- **Current precipitation-to-precipitation has shown a significant dry-bias in yearly totals, which is being investigated**
- Temperature is also implemented, but not fully tested or evaluated yet
- Multiple climatic condition variables is fully integrated, but not yet evaluated
- UNet-SR baseline is under development
- RainGate module is experimental and may require further tuning
- Evaluation metrics may be further expanded to include additional diagnostics
- $\sigma$* control is currently heuristic and may benefit from more principled approaches 

### 10.2 Planned improvements
- Multi-variable downscaling (precipitation + temperature + evaporation)
- Larger spatial domain conditioning and other region outputs
- Temporal context integration (lagged inputs or recurrent diffusion)
- Scale-aware domain adaptation for different regions and future climates
- Improved RainGate module and alternative post-processing methods
- Additional baseline methods (GANs, normalizing flows)
- More extensive hyperparameter sweeps and ablation studies
- Streamlined container build for LUMI with automated ROCm+PyTorch verification and setup
- User-friendly data download and preprocessing pipeline with GUI or web interface
- Notebook tutorials for common workflows
- Integration with climate impact models (hydrology) for end-to-end evaluation
- Scale-aware downscaling of future climate scenarios (RCPs, SSPs) (e.g. CORDEX --> DANRA-like) using methods such as [Hess et al. 2025].


## 11. Citation
Please cite this repository as:

```@misc{CEDDAR2025,
  author = {Quistgaard, T. et al.},
  title = {CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TheaQG/CEDDAR}}
}
```

A manuscript describing CEDDAR is in preparation and a preprint will be made available soon. A full peer-reviewed publication will be added once available.
Until then, use the above citation or a placeholder reference like:

```@article{CEDDAR2025,
  title={CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall},
  author={Quistgaard, T. et al.},
  journal={In preparation},
  year={2025}
}
```

## 12. Acknowledgements
CEDDAR was developed as part of a PhD project at Aarhus University. The author gratefully acknowledges:
- Funding from the **Danish Council for Independent Research** (DFF) under grant number xxxxxxx.
- Computational resources on LUMI provided by both **Danish e-Infrastructure Cooperation** (DeIC) and the local **HPC center at Aarhus University**.
- **DMI** for providing access to the DANRA reanalysis dataset, especially the help of Dr. Xiaohua Yang, Dr. Carlos Andrés Peralta Aros, and Søren Borg Thorsen.
- **ECMWF** for providing access to ERA5 data through the Copernicus Data Store (CDS).
- The open-source community for providing foundational libraries such as PyTorch, NumPy, and others that underpin this work.
- Colleagues, collaborators and supervisors for their valuable feedback, discussions and support during the development of CEDDAR.

## 13. License
CEDDAR is released under the **MIT License**. See the `LICENSE` file for details.

## 14. References
- Hersbach et al., 2020: The ERA5 global reanalysis. Q.J.R. Meteorol. Soc., 146, 1999–2049. https://doi.org/10.1002/qj.3803
- Yang et al., 2020: DANRA: A high-resolution dynamical downscaling reanalysis for Denmark. J. Geophys. Res. Atmos., 125, e2019JD032264. https://doi.org/10.1029/2019JD032264
- Karras et al., 2022: Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. https://arxiv.org/abs/2206.00364
- [CDS API Reference]: https://cds.climate.copernicus.eu/api-how-to
- CDO Documentation: https://code.mpimet.mpg.de/projects/cdo/wiki
- [Hess et al., 2025]: 
