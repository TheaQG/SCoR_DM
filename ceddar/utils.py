'''
    Utility functions for the project.

    Functions:
    ----------
    - model_summary: Print the model summary
    - SimpleLoss: Simple loss function
    - HybridLoss: Hybrid loss function
    - SDFWeightedMSELoss: Custom loss function for SDFs
    - convert_npz_to_zarr: Convert DANRA .npz files to zarr files
    - create_concatenated_data_files: Create concatenated data files
    - convert_npz_to_zarr_based_on_time_split: Convert DANRA .npz files to zarr files based on a time split
    - convert_npz_to_zarr_based_on_percent_split: Convert DANRA .npz files to zarr files based on a percentage split
    - convert_nc_to_zarr: Convert ERA5 .nc files to zarr files
    - extract_samples: Extract samples from dictionary
    
    For argparse:
    - str2bool: Convert string to boolean
    - str2list: Convert string to list
    - str2list_of_strings: Convert string to list of strings
    - str2dict: Convert string to dictionary

    
    TODO:
        - Update extract_samples to also incorporate 'slope'
'''

import torch 
import zarr
import os
import logging

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union
# --------------------------------------------------------------------------------
# Helper: return numpy array with mask-channel removed, if shape == (2, H, W)
# --------------------------------------------------------------------------------
GEO_KEYS = {"lsm", "topo"}      # Plot-keys that carry value||mask
def _squeeze_geo_value(arr, key):
    """
        If *arr* is a torch/np array of shape (2,H,W) *and* identifies as a geo tensor
        assume it is [value, mask] and return only the first channel. Otherwise return 
        the array unchanged.
    """
    if key in GEO_KEYS and hasattr(arr, "shape") and arr.ndim == 3 and arr.shape[0] == 2:
        return arr[0]
    return arr

# Set up logging
logger = logging.getLogger(__name__)


def get_model_string(cfg):
    '''
        Generate a string representation of the model configuration for saving and logging.
        Args:
            cfg (dict): Configuration dictionary containing model settings.
        Returns:
            save_str (str): String representation of the model configuration.
    '''
    # Set image dimensions vased on onfig (if None, use default values)
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)

    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None    
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes (if verbose)
    if cfg['lowres']['resize_factor'] > 1:
        hr_data_size_use = (hr_data_size[0] // cfg['lowres']['resize_factor'], hr_data_size[1] // cfg['lowres']['resize_factor'])
        lr_data_size_use = (lr_data_size_use[0] // cfg['lowres']['resize_factor'], lr_data_size_use[1] // cfg['lowres']['resize_factor'])
    else:
        hr_data_size_use = hr_data_size
        lr_data_size_use = lr_data_size_use

    # Setup specific names for saving
    lr_vars_str = '_'.join(cfg['lowres']['condition_variables'])

    save_str = (
        f"{cfg['experiment']['config_name']}__"
        f"HR_{cfg['highres']['variable']}_{cfg['highres']['model']}__"
        f"SIZE_{hr_data_size_use[0]}x{hr_data_size_use[1]}__"
        f"LR_{lr_vars_str}_{cfg['lowres']['model']}__"
        f"LOSS_{cfg['training']['loss_type']}__"
        f"HEADS_{cfg['sampler']['num_heads']}__"
        f"TIMESTEPS_{cfg['sampler']['n_timesteps']}"
    )

    return save_str


def convert_npz_to_zarr(npz_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    logger.info(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')

    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Make iterator to keep track of progress
    i = 0

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            if VERBOSE:
                logger.info(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))            

            # Loop through all keys in the .npz file
            for key in npz_data:

                if i == 0:
                    logger.info(f'Key: {key}')
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)
            
            # Print progress if iterator is a multiple of 100
            if (i+1) % 100 == 0:
                logger.info(f'Converted {i+1} files...')
            i += 1


def create_concatenated_data_files(data_dir_all:list, data_dir_concatenated:str, variables:list, n_images:int=4):
    '''
        Function to create concatenated data files.
        The function will concatenate the data from the data_dir_all directories
        and save the concatenated data to the data_dir_concatenated directory.

        Parameters:
        -----------
        data_dir_all: list
            List of directories containing the data
        data_dir_concatenated: str
            Directory to save the concatenated data
        variables: list
            List of variables to concatenate
        n_images: int
            Number of images to concatenate
    '''
    logger.info(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

    # Loop through all directories in the data_dir_all list
    for data_dir in data_dir_all:
        # Loop through all files in the directory
        for data_file in os.listdir(data_dir):
            # Check if the file is a .npz file (not dir or .DS_Store)
            if data_file.endswith('.npz'):
                # Load the .npz file
                npz_data = np.load(os.path.join(data_dir, data_file))
                # Loop through all variables in the .npz file
                for var in variables:
                    # Select the data from the variable
                    data = npz_data[var][:n_images]
                    # Save the data as a zarr array
                    zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
    
    logger.info(f'Concatenated data saved to {data_dir_concatenated}...')




class data_preperation():
    '''
        Class to handle data preperation for the project.
        All data is located in an /all/ directory.
        This class can handle:
            - Check if data already exists
            - Creating train, val and test splits (based on years or percentages) - and saving to zarr
            - Clean up after training (remove zarr files)

    '''
    def __init__(self, args):
        self.args = args
        self.data_dir_all = args.data_dir_all

    def create_concatenated_data_files(self, data_dir_all, data_dir_concatenated, n_images=4):
        '''
            Function to create concatenated data files.
            The function will concatenate the data from the data_dir_all directories
            and save the concatenated data to the data_dir_concatenated directory.

            Parameters:
            -----------
            data_dir_all: list
                List of directories containing the data
            data_dir_concatenated: str
                Directory to save the concatenated data
            n_images: int
                Number of images to concatenate
        '''
        logger.info(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

        # Create zarr group (equivalent to a directory)
        zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

        # Loop through all directories in the data_dir_all list
        for data_dir in data_dir_all:
            # Loop through all files in the directory
            for data_file in os.listdir(data_dir):
                # Check if the file is a .npz file (not dir or .DS_Store)
                if data_file.endswith('.npz'):
                    # Load the .npz file
                    npz_data = np.load(os.path.join(data_dir, data_file))
                    # Loop through all variables in the .npz file
                    for var in npz_data:
                        # Select the data from the variable
                        data = npz_data[var][:n_images]
                        # Save the data as a zarr array
                        zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
        
        logger.info(f'Concatenated data saved to {data_dir_concatenated}...')


def convert_npz_to_zarr_based_on_time_split(npz_directory:str, year_splits:list):
    '''
        Function to convert DANRA .npz files to zarr files based on a time split.
        The function will select the files based on the year_splits list.
        Will create train, val and test splits based on the years specified in the list.

        Parameters:
        -----------
        npz_directory: str
            Directory containing all .npz data files
        year_splits: list
            List of 3 lists containing the years for train, val and test splits
    '''
    # Print the years for each split
    # Print number of files in the split years
    logger.info(f'\nTrain years: {year_splits[0]}')
    logger.info(f'Number of files in train years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[0])])}')

    logger.info(f'\nVal years: {year_splits[1]}')
    logger.info(f'Number of files in val years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[1])])}')

    logger.info(f'\nTest years: {year_splits[2]}')
    logger.info(f'Number of files in test years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[2])])}')

    # 



def convert_npz_to_zarr_based_on_percent_split(npz_directory:str, percent_splits:list, random_selection:bool=False):
    '''
        Function to convert DANRA .npz files to zarr files based on a percentage split.
        The function will randomly select the percentage of files specified in the percent_splits list
        or select the first files in the directory.
        Will create train, val and test splits based on the percentages specified in the list.
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        percent_splits: list
            List of floats representing the percentage splits
    
    '''
    logger.info(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')



def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    logger.info(f'Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            if VERBOSE:
                logger.info(os.path.join(nc_directory, nc_file))
            # Load the .nc file
            nc_data = nc.Dataset(os.path.join(nc_directory, nc_file), mode='r') # type: ignore
            # Loop through all variables in the .nc file
            for var in nc_data.variables:
                # Select the data from the variable
                data = nc_data[var][:]
                # Save the data as a zarr array
                zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)

def extract_samples(samples, device=None):
    """
        Extract samples from the dictionary returned by the dataset class.
        Expected keys:
            - HR image: key ending with '_hr' (e.g. 'prcp_hr') [ignoring keys ending with '_original']
            - Classifier: key 'classifier'
            - LR conditions: key(s) ending with '_lr' (e.g. 'prcp_lr')
            - HR mask: key 'lsm_hr'
            - Land/sea mask: key 'lsm'
            - SDF: key 'sdf'
            - Topography: key 'topo'
            - Points: keys 'hr_point' and 'lr_point'
        If multiple LR condition keys are present, they are concatenated along the channel dimension
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # HR image (choose key ending with '_hr' not containing 'original')
    hr_keys = [k for k in samples.keys() if k.endswith('_hr') and not k.endswith('_original')]
    # If key 'lsm_hr' in hr_keys, remove it
    if 'lsm_hr' in hr_keys:
        hr_keys.remove('lsm_hr')
    
    if len(hr_keys) == 0:
        raise ValueError('No HR image found in samples dictionary.')
    hr_img = samples[hr_keys[0]].to(device, non_blocking=True).float()
    # if len(hr_keys) > 1:
    #     logger.warning(f'Multiple HR images found. Using the first one: {hr_keys[0]}')
    
    # Seasonal label (unified): prefer 'y' only; warn and fallback to legacy keys if needed
    classifier = samples.get('y', None)
    if classifier is None:
        for _legacy_key in ['classifier', 'seasons', 'season']:
            if _legacy_key in samples:
                logger.warning("[seasonal] Found legacy key '%s' in samples; prefer using unified key 'y' in future.", _legacy_key)
                logger.warning("         For now: using '%s' as classifier.", _legacy_key)
                classifier = samples[_legacy_key]
                break
    if classifier is not None:
        classifier = classifier.to(device, non_blocking=True)
        # Normalzie accepatble shapes/dtypes
        #   - categorical: Long [B] or [B,1]
        #   - sin/cos: Float [B,2]
        if classifier.ndim == 1:
            classifier = classifier.unsqueeze(1)
        if classifier.ndim != 2:
            raise ValueError(f"[extract_samples] y has ndim={classifier.ndim}, but expected 2D [B,1] or [B,2].")
        C = classifier.shape[1]
        if torch.is_floating_point(classifier):
            if C != 2:
                raise ValueError(f"[extract_samples] float y must have shape [B,2] for sin/cos encoding; got {classifier.shape}.")
        else:
            if classifier.dtype not in (torch.long, torch.int64):
                raise TypeError(f"[extract_samples] categorical y must be Long; got {classifier.dtype}.")
            if C != 1:
                raise ValueError(f"[extract_samples] categorical y must have shape [B,1]; got {classifier.shape}.")

    # LR conditions: if multiple, stack along channel dimensio
    lr_keys = [k for k in samples.keys() if k.endswith('_lr') and not k.endswith('_original')]
    if len(lr_keys) == 0:
        lr_img = None
    elif len(lr_keys) == 1:
        lr_img = samples[lr_keys[0]].to(device, non_blocking=True).float()
    else:
        lr_list = [samples[k].to(device, non_blocking=True).float() for k in sorted(lr_keys)]
        lr_img = torch.cat(lr_list, dim=1)

    # HR mask (LSM)
    lsm_hr = samples.get('lsm_hr', None)
    if lsm_hr is not None:
        lsm_hr = lsm_hr.to(device, non_blocking=True).float()
    
    # Land/sea mask (LSM)
    lsm = samples.get('lsm', None)
    if lsm is not None:
        lsm = lsm.to(device, non_blocking=True).float()
    
    # SDF
    sdf = samples.get('sdf', None)
    if sdf is not None:
        sdf = sdf.to(device, non_blocking=True).float()

    # Topography
    topo = samples.get('topo', None)
    if topo is not None:
        topo = topo.to(device, non_blocking=True).float()

    # HR crop points (if available)
    hr_points = samples.get('hr_point', None)
    if hr_points is not None:
        hr_points = hr_points.to(device).float()

    # LR crop points (if available)
    lr_points = samples.get('lr_point', None)
    if lr_points is not None:
        lr_points = lr_points.to(device).float()

    # Return all extracted samples
    return hr_img, classifier, lr_img, lsm_hr, lsm, sdf, topo, hr_points, lr_points


def get_first_sample_dict(samples: dict) -> dict:
    """
        Return a new dictionary with only the first samples from a batch-wise sample dict
        - Tensors: sliced via [:1] to retain batch-dimension
        - List/tuples with length matching batch: take [0]
        - Other objects: kept as-is

    """
    single_sample = {}
    batch_size = None

    # Infer batch size from first tensor
    for v in samples.values():
        if torch.is_tensor(v):
            batch_size = v.shape[0]
            break

    if batch_size is None:
        raise ValueError("No tensor found in samples to determine batch size.")
    

    # Now build the single sample dict
    for k, v in samples.items():
        if torch.is_tensor(v):
            single_sample[k] = v[:1]
        elif isinstance(v, (list, tuple)) and len(v) == batch_size:
            single_sample[k] = [v[0]]
        else:
            single_sample[k] = v

    return single_sample



def build_data_path(base_path, model, var, full_domain_dims, split, zarr_file=True):
    """
    Construct a path for high-resolution data.
    Example: base_path + 'data_DANRA/size_589x789/temp_589x789/zarr_files/train.zarr'
    
    """
    if zarr_file:
        data_path = os.path.join(base_path, f"data_{model}", f"size_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{var}_{full_domain_dims[0]}x{full_domain_dims[1]}", "zarr_files", f"{split}.zarr")
    else:
        data_path = os.path.join(base_path, f"data_{model}", f"size_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{var}_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{split}/")
    
    return data_path







def crop_to_region(data, crop_region):
    """
    Crop the data to a specific subregion: [x_start, x_end, y_start, y_end].
    """
    [x_start, x_end, y_start, y_end] = crop_region
    return data[x_start:x_end, y_start:y_end]



from omegaconf import OmegaConf

def load_config(config_path):
    """
        Loads and resolves an OmegaConf configuration
    """
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda x: os.environ.get(x))
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    return cfg

def report_precip_extremes(x_bt: torch.Tensor, name: str, cap_mm_day: float = 500.0, logger=print):
    """
        Reports extremes in a back-transformed precipitation tensor.
        Values below 0 are counted as negative, values above cap_mm_day are counted as extreme.
    """
    flat = x_bt.flatten(1)
    p999 = torch.quantile(flat, 0.999, dim=1)
    mx = torch.max(flat, dim=1).values
    n_ex = 0
    vals_ex = []
    n_b0 = 0
    vals_b0 = []
    for i, (p, m) in enumerate(zip(p999.tolist(), mx.tolist())):
        if m > max(5.0 * p, cap_mm_day):
            logger(f"{name} sample {i} has extreme precipitation: max={m:.1f} mm/day > max(5xp99.9={p:.1f} mm/day)")
            n_ex += 1
            vals_ex.append(m)
        if m < 0:
            logger(f"{name} sample {i} has negative precipitation: max={m:.1f} mm/day < 0")
            n_b0 += 1
            vals_b0.append(m)
    if n_b0 > 0 and n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex,
                'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}
    if n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex}
    if n_b0 > 0:
        return {'has_below_zero': True, 'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}

    return {'has_extreme': False}


# def load_config(yaml_file):
#     """
#         Loads a YAML configuration file and returns a dictionary
#     """
#     with open(yaml_file, 'r') as f:
#         config = yaml.safe_load(f)
#         return config