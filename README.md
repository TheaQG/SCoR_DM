# SCoR-DM: Scale-Controllable Residual Diffusion Model
# CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall

### NOTE: This project is currently under development and the code is not yet fully functional. It is intended to be used as a reference for implementing score-based generative models for downscaling climate data. This README will be updated as the project progresses.


This repository contains the code for the SBGM_DANRA project, a work on using score-based generative models for downscaling of climate data. The project is structured into several directories, each serving a specific purpose in the workflow from data preparation to model training and evaluation.

The model downscales ERA5 [REF] data (~31 km spatial resolution) to a higher resolution corresponding to the DANRA [REF] target dataset (2.5 km spatial resolution). The model is trained with ERA5 variables as input and DANRA variables as target. The training is done using a score-based generative model, which learns to generate high-resolution climate data conditioned on low-resolution inputs. 
Multiple models have been trained, with different LR conditions to attempt a downscaling closer to an emulator, attempting to capture dynamic meteorological processes from various variables.

<!-- Examples of the generated samples can be found in the `models_and_samples/generated_samples` directory, and the trained models are stored in the `models_and_samples/trained_models` directory.

To run the code, you will need to set up the environment as described in the `requirements.txt` or `environment.yml` files. The project is structured to facilitate easy data handling, model training, and evaluation.
For training the model, you can use the scripts in the `scripts/model_runs` directory. The evaluation of the model can be done using the scripts in the `scripts/evaluation` directory. -->


## Directory Structure
```
.
├── baselines
│   ├── __init__.py
│   ├── adapter.py
│   ├── baseline_eval.py
│   ├── baseline_main.py
│   ├── bilinear.py
│   ├── build_dataset.py
│   ├── evaluate_baselines
│   │   ├── eval_dataresolver_baselines.py
│   │   ├── evaluation_baselines.py
│   │   └── launch_baseline_evaluation.py
│   ├── plotting.py
│   ├── quantile_mapping.py
│   └── unet_sr
│       ├── __init__.py
│       ├── model.py
│       └── train.py
├── bash_scripts
├── data_analysis_pipeline
│   ├── __init__.py
│   ├── cli
│   │   ├── __init__.py
│   │   ├── launch_comparison.py
│   │   ├── launch_correlations.py
│   │   ├── launch_small_batches_creation.py
│   │   ├── launch_split_creation.py
│   │   ├── launch_statistics.py
│   │   └── main_data_app.py
│   ├── comparison
│   │   ├── __init__.py
│   │   ├── compare_distributions.py
│   │   ├── compare_fields.py
│   │   ├── compare_timeseries.py
│   │   ├── comparison_main.py
│   │   ├── comparison_pipeline.py
│   │   └── plot_utils.py
│   ├── configs
│   │   ├── __init__.py
│   │   ├── comparison_config.yaml
│   │   ├── correlation_config.yaml
│   │   ├── default_data_config.yaml
│   │   ├── small_batches_config.yaml
│   │   ├── split_config.yaml
│   │   └── stats_config.yaml
│   ├── correlations
│   │   ├── correlation_main.py
│   │   ├── correlation_methods.py
│   │   ├── correlation_pipeline.py
│   │   ├── correlation_plotting.py
│   │   └── data_correlations.py
│   ├── preprocess
│   │   ├── __init__.py
│   │   ├── create_small_data_batches.py
│   │   ├── daily_files_to_zarr.py
│   │   ├── filter_data.py
│   │   └── small_batches_main.py
│   ├── splits
│   │   ├── __init__.py
│   │   ├── create_train_valid_test.py
│   │   └── splits_main.py
│   └── stats_analysis
│       ├── __init__.py
│       ├── data_loading.py
│       ├── data_stats_pipeline.py
│       ├── path_utils.py
│       ├── plotting.py
│       ├── statistics.py
│       └── stats_main.py
├── data_examples
│   ├── DANRA
│   ├── ERA5
│   └── figures
├── era5_download_pipeline
│   ├── __init__.py
│   ├── cfg
│   │   ├── era5_pipeline_testing.yaml
│   │   ├── era5_pipeline.yaml
│   │   ├── era5_pressure_pipeline_testing.yaml
│   │   └── era5_pressure_pipeline.yaml
│   ├── cli
│   │   ├── __init__.py
│   │   ├── run_local_pressure.py
│   │   ├── run_local.py
│   │   └── run_lumi.py
│   ├── era5_logs
│   │   ├── era5_download.log
│   │   └── era5_run_restartable.log
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── cdo_utils.py
│   │   ├── download.py
│   │   ├── remote_utils.py
│   │   ├── stream.py
│   │   ├── transfer.py
│   │   └── utils.py
│   ├── slurm
│   │   ├── local_pressure_process.sh
│   │   ├── local_process.sh
│   │   ├── lumi_process.sh
│   │   └── rename_zpl.sh
│   └── utils
│       ├── __init__.py
│       └── logging_utils.py
├── models_and_samples
├── README.md
├── requirements.txt
└── scor_dm
    ├── __init__.py
    ├── cli
    │   ├── __init__.py
    │   ├── launch_evaluation_sampler_grid.py
    │   ├── launch_evaluation_sigma_star.py
    │   ├── launch_evaluation.py
    │   ├── launch_generation_sampler_grid.py
    │   ├── launch_generation_sigma_star.py
    │   ├── launch_generation.py
    │   ├── launch_quicklook.py
    │   ├── launch_sbgm.py
    │   └── main_app.py
    ├── config
    │   ├── __init__.py
    │   ├── ..
    │   ├── ablations
    │   │   └── ..
    ├── data
    │   ├── __init__.py
    │   ├── create_train_valid_test.py
    │   └── datasets_loading.py
    ├── data_modules.py
    ├── evaluate
    │   ├── __init__.py
    │   ├── data_resolver.py
    │   ├── evaluate_main.py
    │   ├── evaluate_multivariate
    │   │   └── __init__.py
    │   ├── evaluate_prcp
    │   │   ├── __init__.py
    │   │   ├── eval_dates
    │   │   │   ├── __init__.py
    │   │   │   ├── evaluate_dates.py
    │   │   │   ├── metrics_dates.py
    │   │   │   └── plot_dates.py
    │   │   ├── eval_distributions
    │   │   │   ├── ...
    │   │   ├── eval_extremes
    │   │   │   ├── ...
    │   │   ├── eval_features
    │   │   │   ├── ...
    │   │   ├── eval_probabilistic
    │   │   │   ├── ...
    │   │   ├── eval_scale
    │   │   │   ├── ...
    │   │   ├── eval_sigma_star
    │   │   │   ├── ...
    │   │   ├── eval_spatial
    │   │   │   ├── ...
    │   │   ├── eval_temporal
    │   │   │   ├── ...
    │   │   ├── overlay_utils.py
    │   │   ├── plot_utils.py
    │   │   └── sigma_control
    │   ├── evaluate_prcp.zip
    │   ├── evaluate_sampler_grid_main.py
    │   ├── evaluate_temp
    │   │   └── __init__.py
    │   ├── evaluation.py
    │   └── summarize_sampler_grid.py
    ├── evaluate_sbgm
    │   ├── __init__.py
    │   ├── evaluation_main.py
    │   ├── evaluation.py
    │   ├── metrics_univariate.py
    │   └── plot_utils.py
    ├── generate
    │   ├── generation_main.py
    │   ├── generation_sampler_grid_main.py
    │   ├── generation_sigma_grid_main.py
    │   ├── generation.py
    │   └── quicklook.py
    ├── heads
    │   ├── __init__.py
    │   └── rain_gate.py
    ├── logging_utils.py
    ├── losses.py
    ├── monitoring.py
    ├── plotting_utils.py
    ├── scale_utils.py
    ├── score_sampling.py
    ├── score_unet.py
    ├── special_transforms.py
    ├── training_main.py
    ├── training_utils.py
    ├── training.py
    ├── utils.py
    └── variable_utils.py

```

# SBGM_DANRA
This directory contains the main code for the SBGM_DANRA project, including scripts for data handling, model training, and evaluation. The code is organized into several subdirectories, with a focus on modularity and reusability. The main components of the project are:

## bash_scripts
Contains bash scripts for setting up the environment and running the project. These scripts help automate the setup process and ensure that the necessary dependencies are installed and configured correctly.

## data_examples
Contains example datasets for the ERA5 and DANRA datasets, as well as figures for data comparison, correlations, distributions, and statistics. These examples are used to demonstrate the functionality of the project and provide a reference for users to understand the data formats and structures.
    - **ERA5**: Contains example data files from the ERA5 dataset, which is used as the low-resolution input for the model.
    - **DANRA**: Contains example data files from the DANRA dataset, which is used as the high-resolution target for the model.
    - **figures**: Contains figures generated from the data examples, including raw data examples, data comparisons, data correlations, data distributions, and data statistics. 

## models_and_samples
Contains the trained models and generated samples from the SBGM_DANRA project. This directory is organized into subdirectories for evaluations, generated samples, and trained models.
    - **evaluations**: Contains evaluation results and metrics for the trained models, including performance metrics and visualizations.
    - **generated_samples**: Contains samples generated by the trained score-based generative model. These samples are used to assess the quality of the model's downscaling capabilities.
    - **trained_models**: Contains the trained score-based generative models. These models can be used for generating new samples or for further evaluation and analysis.


## sbgm
This directory contains the main code for the SBGM_DANRA project, including scripts for data handling, model training, and evaluation. The code is organized into several subdirectories:
    - cli: Contains command-line interface scripts for launching the generation and training processes.
    - config: Contains configuration files for the project, including default settings and parameters for model training and evaluation.
    - data_scripts: Contains scripts for data preparation, including creating smaller data batches, splitting datasets into training, validation, and test sets, converting daily files to Zarr format, and performing data comparisons and correlations.
    - evaluation: Contains scripts for evaluating the model performance, including metrics computation and visualization.
    - __init__.py: Initializes the sbgm package.


The main scripts in this directory include:
    - **data_modules.py**: Contains the data modules used for loading and processing the datasets specifically for model training. It defines the structure and format of the data, as well as any transformations or preprocessing steps that need to be applied before training the model.
    - **generation.py**: Responsible for generating samples from the trained score-based generative model. It uses the trained model to produce high-resolution climate data conditioned on low-resolution inputs.
    - **main_sbgm.py**: The main entry point for training the score-based generative model. It orchestrates the training process, including data loading, model initialization, and training loop.
    - **score_sampling.py**: Responsible for sampling from the trained score-based generative model. It uses the model to generate new samples based on the learned distribution, allowing for the creation of high-resolution climate data.
    - **score_unet.py**: Contains the implementation of the score-based UNet architecture used in the generative model. This script defines the neural network structure, including the layers and operations that make up the model.
    - **special_transforms.py**: Contains special transformations that can be applied to the datasets during model training, such as normalization, scaling, or other preprocessing steps. These transformations are used to prepare the data for training the score-based generative model.
    - **training.py**: Contains the training loop for the score-based generative model. It handles the forward and backward passes, loss computation, and optimization steps to train the model on the provided datasets.
    - **utils.py**: Contains utility functions that are used throughout the model training scripts, such as file handling, data manipulation, and other common tasks. These functions help to streamline the model training workflow and reduce code duplication.



### cli
- Contains command-line interface scripts for launching the generation and training processes. These scripts allow users to run the model training and generation from the command line, providing a user-friendly interface for interacting with the project.

### config
- Contains configuration files for the project, including default settings and parameters for model training and evaluation. These files allow users to customize the behavior of the scripts without modifying the code directly, making it easier to adapt the project to different use cases or datasets.

### Data scripts
This directory contains scripts for data preparation, including creating smaller data batches, splitting datasets into training, validation, and test sets, converting daily files to Zarr format, and performing data comparisons and correlations. These scripts help to preprocess the data and prepare it for model training and evaluation.

- **create_small_data_batches.py**: This script is used to create smaller batches of data from larger datasets for easier handling and processing during model training and evaluation.

- **create_train_valid_test_data.py**: This script is responsible for splitting the dataset into training, validation, and test sets. It ensures that the data is properly partitioned for model training and evaluation.

- **daily_files_to_zarr.py**: This script converts daily downloaded ERA5/DANRA files (.npz or .nc) into the Zarr format, which is a format optimized for large datasets and allows for efficient storage and retrieval of multi-dimensional arrays.

- **data_comparison.py**: Compares the Low Resolution (LR) ERA5 and High Resolution (HR) DANRA datasets to visualise the differences and similarities between the two datasets. This is useful for understanding the characteristics of the data before training the model. 

- **data_correlations.py**: This script calculates and visualizes the correlations between different variables in the datasets, helping to identify relationships and dependencies that may be important for model training.

- **data_filter.py**: Filters the datasets based on specific criteria, such as date ranges or variable values, to prepare the data for training and evaluation. This is useful for focusing on relevant subsets of the data, or for excluding outliers or irrelevant data points.

- **data_stats.py**: This script computes and visualizes statistics of the datasets, such as mean, standard deviation, and distribution of variables. This is useful for understanding the data and identifying any potential issues or anomalies before training the model.

.
└── Data_DiffMod/
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
    └── data_ERA5/
        └── size_589x789/
            └── prcp_589x789/
                ├── all/
                │   ├── tp_589x789_19910101.npz
                │   ├── ...
                │   └── tp_589x789_20201231.npz
                └── zarr_files/
                    ├── train.zarr/
                    │   ├── tp_tot_19910101
                    │   ├── ...
                    │   └── tp_tot_20151231            
                    ├── valid.zarr/
                    │   ├── tp_tot_20160101
                    │   ├── ...
                    │   └── tp_tot_20181231            
                    └── test.zarr/
                        ├── tp_589x789_20190101
                        ├── ...
                        └── tp_589x789_20201231            

<!-- 
### evaluation
- Contains scripts for evaluating the model performance, including metrics computation and visualization. These scripts help to assess the quality of the generated samples and the effectiveness of the model in downscaling climate data.
 -->

<!-- ## model_runs

### data_modules.py
- Contains the data modules used for loading and processing the datasets specifically for model training. It defines the structure and format of the data, as well as any transformations or preprocessing steps that need to be applied before training the model.

### generation.py
- This script is responsible for generating samples from the trained score-based generative model. It uses the trained model to produce high-resolution climate data conditioned on low-resolution inputs.

### launch_generation.py
- This script is used to launch the generation process, which involves running the generation script with the appropriate parameters and configurations. It sets up the environment and initiates the sample generation from the trained model.

### launch_sbgm.py
- This script is used to launch the training of the score-based generative model (SBGM). It sets up the environment, loads the data, and initiates the training process with the specified parameters and configurations.

### main_sbgm.py
- The main entry point for training the score-based generative model. It orchestrates the training process, including data loading, model initialization, and training loop. This script is typically run to start the model training.

### score_sampling.py
- This script is responsible for sampling from the trained score-based generative model. It uses the model to generate new samples based on the learned distribution, allowing for the creation of high-resolution climate data.

### score_unet.py
- Contains the implementation of the score-based UNet architecture used in the generative model. This script defines the neural network structure, including the layers and operations that make up the model.

### special_transforms.py
- Contains special transformations that can be applied to the datasets during model training, such as normalization, scaling, or other preprocessing steps. These transformations are used to prepare the data for training the score-based generative model.

### test_data_transformations.py
- This script tests the data transformations applied to the datasets to ensure that they are correctly formatted and ready for model training. It checks for any issues or errors in the data transformation process.

### training.py
- This script contains the training loop for the score-based generative model. It handles the forward and backward passes, loss computation, and optimization steps to train the model on the provided datasets.

### utils.py
- Contains utility functions that are used throughout the model training scripts, such as file handling, data manipulation, and other common tasks. These functions help to streamline the model training workflow and reduce code duplication.

## evaluation -->