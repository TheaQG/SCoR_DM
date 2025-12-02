"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset can be used for training and testing the SBGM_SD model.

    TODO:
        - Add static sampling (no crop + shift) option (fixed cutout)
        - Add multiple cutout domains (Northern Germany, Poland, Netherlands etc.)
        - Add option for Day-Of-Year conditional sampling
        - Add option for 'Slope' as geo variable
"""

# Import libraries and modules 
import zarr
import re
import random
import torch
import logging
import math
# import multiprocessing

import numpy as np
import torch.nn.functional as F

from typing import Optional
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from scipy.ndimage import distance_transform_edt as distance

from special_transforms import Scale, get_transforms_from_stats
from variable_utils import correct_variable_units

# Set logging
logger = logging.getLogger(__name__)

def preprocess_lsm_topography(lsm_path, topo_path, target_size, scale=False, flip=False):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to [0, 1] interval,
        and upscales the data to match the target size.

        Input:
            - lsm_path: path to lsm data
            - topo_path: path to topography data
            - target_size: tuple containing the target size of the data
    '''
    # 1. Load the Data and flip upside down if flip=True
    if flip:
        lsm_data = np.flipud(np.load(lsm_path)['data']).copy() # Copy to avoid negative strides
        topo_data = np.flipud(np.load(topo_path)['data']).copy() # Copy to avoid negative strides
        
    else:
        lsm_data = np.load(lsm_path)['data']
        topo_data = np.load(topo_path)['data']

    # 2. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    if scale: # SHOULD THIS ALSO BE A Z-SCALE TRANSFORM?
        # 3. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 4. Upscale the Fields to match target size
    resize_lsm = transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST, antialias=False) # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
    resize_topo = transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR, antialias=True) # Bilinear for continuous topo data
    lsm_tensor = resize_lsm(lsm_tensor)
    topo_tensor = resize_topo(topo_tensor)

    return lsm_tensor, topo_tensor

def preprocess_lsm_topography_from_data(lsm_data, topo_data, target_size, scale=True):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to[0, 1] interval (if scale=True)),
        and upscales the data to match the target size.

        Input:
            - lsm_data: lsm data
            - topo_data: topography data
            - target_size: tuple containing the target size of the data
            - scale: whether to scale the topography data to [0, 1] interval
    '''    
    # 1. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data.copy()).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data.copy()).float().unsqueeze(0)
    
    if scale:
        # 2. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 3. Upscale the Fields to match target size
    resize_lsm = transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST, antialias=False) # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
    resize_topo = transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR, antialias=True) # Bilinear for continuous topo data
    lsm_tensor = resize_lsm(lsm_tensor)
    topo_tensor = resize_topo(topo_tensor)

    return lsm_tensor, topo_tensor

def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Set land to 1 and subtract sea distances
    sdf = 10*binary_mask.float() - dist_transform_sea # NOTE: 10 is an arbitrary positive value for land - not a problem, when normalizing later

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    if isinstance(sdf, torch.Tensor):
        min_val = torch.min(sdf)
        max_val = torch.max(sdf)
    elif isinstance(sdf, np.ndarray):
        min_val = np.min(sdf)
        max_val = np.max(sdf)
    else:
        raise ValueError('SDF must be either torch.Tensor or np.ndarray')

    # Normalize the SDF
    sdf_normalized = (sdf - min_val) / (max_val - min_val)
    return sdf_normalized

class DateFromFile:
    '''
    General class for extracting date from filename.
    Can take .npz, .nc and .zarr files.
    Not dependent on the file extension.
    '''
    def __init__(self, filename):
        # Remove file extension
        self.filename = filename.split('.')[0]
        # Get the year, month and day from filename ending (YYYYMMDD)
        self.year = int(self.filename[-8:-4])
        self.month = int(self.filename[-4:-2])
        self.day = int(self.filename[-2:])

    def determine_season(self):
        # Determine season based on month
        if self.month in [3, 4, 5]:
            return 1
        elif self.month in [6, 7, 8]:
            return 2
        elif self.month in [9, 10, 11]:
            return 3
        else:
            return 4

    def determine_month(self):
        # Returns the month as an integer in the interval [1, 12]
        return self.month

    @staticmethod
    def is_leap_year(year):
        """Check if a year is a leap year"""
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        return False

    def determine_day(self):
        # Days in month for common years and leap years
        days_in_month_common = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_month_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Determine if the year is a leap year
        if self.is_leap_year(self.year):
            days_in_month = days_in_month_leap
        else:
            days_in_month = days_in_month_common

        # Compute the day of the year
        day_of_year = sum(days_in_month[:self.month]) + self.day # Now 1st January is 1 instead of 0
        return day_of_year
    
def FileDate(filename):
    """
    Extract the last 8 digits from the filename as the date string.
    E.g. for 't2m_ave_19910122' or 'temp_589x789_19910122', returns '19910122'
    """

    m = re.search(r'(\d{8})$', filename)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"Could not extract date from filename: {filename}")


def find_rand_points(rect, crop_size):
    '''
    Randomly selects a crop region within a given rectangle
    Input:
        - rect (list or tuple): [x1, x2, y1, y2] rectangle to crop from
        - crop_size (tuple): (crop_width, crop_height) size of the desired crop
    Output:
        - point (list): [x1_new, x2_new, y1_new, y2_new] random crop region (slice as [y1: y2, x1: x2])

    Raises: 
        - ValueError if crop_size is larger than the rectangle
    '''
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    crop_width = crop_size[0]
    crop_height = crop_size[1]

    full_width = x2 - x1
    full_height = y2 - y1

    if crop_width > full_width or crop_height > full_height:
        raise ValueError('Crop size is larger than the rectangle dimensions.')

    # Calculate available offsets
    max_x_offset = full_width - crop_width
    max_y_offset = full_height - crop_height

    offset_x = random.randint(0, max_x_offset)
    offset_y = random.randint(0, max_y_offset)

    x1_new = x1 + offset_x
    x2_new = x1_new + crop_width
    y1_new = y1 + offset_y
    y2_new = y1_new + crop_height

    point = [x1_new, x2_new, y1_new, y2_new]
    return point


def random_crop(data, target_size):
    """
        Randomly crops a 2D 'data' to shape (target_size[0], target_size[1]).
        Assumes data is a 2D numpy array
        Input:
            - data: 2D numpy array
            - target_size: tuple containing the target size of the data
        Output:
            - data: 2D numpy array with shape (target_size[0], target_size[1])
        Raises:
            - ValueError if target size is larger than the data dimensions    
    """
    H, W = data.shape

    if target_size[0] > H or target_size[1] > W:
        raise ValueError('Target size is larger than the data dimensions.')
    
    if H == target_size[0] and W == target_size[1]:
        return data

    y = random.randint(0, H - target_size[0])
    x = random.randint(0, W - target_size[1])
    return data[y:y + target_size[0], x:x + target_size[1]]


def make_tensor_resize(target_size):
    """
        Create a transform that resizes a tensor to the target size.
        Necessary because the Resize transform in torchvision expects a PIL image.
        Resize otherwise silently no-ops on a 2D tensor.
    """
    return transforms.Lambda(
        lambda t:
        F.interpolate(
            t.unsqueeze(0), # [1, C, H, W]
            size=target_size, # (new_height, new_width)
            mode='bilinear', # Or 'nearest' or 'bicubic' etc
            align_corners=False,
        ).squeeze(0)        # Back to [C, H, W]
    )

class SafeToTensor:
    def __call__(self, x):

        if isinstance(x, np.ndarray):
            return transforms.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"Unexpected input type: {type(x)}. Expected np.ndarray or torch.Tensor.")

class ResizeTensor:
    """
        Create a transform that resizes a tensor to the target size.
        Necessary because the Resize transform in torchvision expects a PIL image.
        Resize otherwise silently no-ops on a 2D tensor.
        
        Resize a torch.Tensor of shape [C,H,W] to [C,new_H,new_W]
        by torch.nn.functional.interpolate.  No PIL, no 8-bit quantization.
    """
    def __init__(self, size, mode='bilinear', align_corners=False):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2: # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions → [1, 1, H, W]
        elif x.ndim == 3: # [C, H, W]
            x = x.unsqueeze(0) # [1, C, H, W]
        elif x.ndim != 4:
            raise ValueError(f"ResizeTensor: Unsupported shape {x.shape}")
        
        # Align_corners is only used for mode='bilinear' or 'bicubic'
        if self.mode in ["bilinear", "bicubic"]:
            # interpolate → [1, C, new_H, new_W]
            x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        elif self.mode == "nearest":
            x = F.interpolate(x, size=self.size, mode=self.mode)
        else:
            raise ValueError(f"ResizeTensor: Unsupported mode {self.mode}")

        # remove batch dim → [C, new_H, new_W]
        return x.squeeze(0) # → [C, H, W] or [1, H, W] depending on input


# === Helper transform for dual-LR logic ===
class DualLRTransform:
    """
        Callable transform that applies two independent transform pipelines to the same LR numpy array/tensor
        and returns a 2-channel tensor [2, H, W] by stacking the results along the channel dimension.
    """

    def __init__(self, transform_a, transform_b):
        self.ta = transform_a
        self.tb = transform_b
    def __call__(self, x):
        a = self.ta(x)
        b = self.tb(x)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a)
        if isinstance(b, np.ndarray):
            b = torch.tensor(b)
        if a.ndim == 2:
            a = a.unsqueeze(0) # [1, H, W]
        if b.ndim == 2:
            b = b.unsqueeze(0) # [1, H, W]
        # Ensure dtype and device alignment
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        if a.device != b.device:
            b = b.to(a.device)
        return torch.cat([a, b], dim=0) # [2, H, W]


def list_all_keys(zgroup):
    all_keys = []
    for key in zgroup.keys():
        all_keys.append(key)
        member = zgroup[key]
        if isinstance(member, zarr.Group):
            sub_keys = list_all_keys(member)
            all_keys.extend([f"{key}/{sub}" for sub in sub_keys])
    return all_keys


# Helper for robust array extraction
def _first_hw_slice(arr: np.ndarray) -> np.ndarray:
    """
        Return a 2D (H, W) view by taking the first slice along any leading dimensions.
        Works for shapes like (H, W), (1, H, W), (N, H, W), (N, 1, H, W), etc.
    """
    arr = np.asarray(arr) # Ensure it's a numpy array
    if arr.ndim < 2:
        raise ValueError(f"Array must have at least 2 dimensions, got shape {arr.shape}")
    H, W = arr.shape[-2], arr.shape[-1]
    return arr.reshape(-1, H, W)[0] # First slice of shape (H, W)

# Helper for extracting field from zarr group entry with common dataset keys per variable
def _extract_2d_from_zarr_entry(zgroup: zarr.Group, file_key: str, var_name: str) -> np.ndarray:
    """
        Load a 2D field from a zarr group entry, trying common dataset keys per variable.
        Returns a numpy array with shape (H, W).
    """
    entry = zgroup[file_key]

    KEY_CANDIDATES = {
        'temp': ['t', 'data', 'arr_0'],
        't2m': ['t', 'data', 'arr_0'],
        'prcp': ['tp', 'data', 'arr_0'],
        'tp': ['tp', 'data', 'arr_0'],
        '_default': ['data', 'arr_0'],
    }

    candidates = KEY_CANDIDATES.get(var_name, []) + KEY_CANDIDATES['_default']
    for k in candidates:
        if k in entry:
            arr = entry[k][()] # Load the array
            return _first_hw_slice(arr) # type: ignore # Return as (H, W)
        
    # Fallback: try any array-like members under the entry
    for k in entry.keys(): # type: ignore
        try:
            arr = entry[k][()]
            return _first_hw_slice(arr) # type: ignore
        except Exception:
            continue
    raise KeyError(f"Could not find a suitable data array in zarr entry '{file_key}' for variable '{var_name}'. Tried keys: {candidates} and all members.")


class DANRA_Dataset_cutouts_ERA5_Zarr(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                # Must-have parameters
                hr_variable_dir_zarr:str,           # Path to high resolution data
                hr_data_size:tuple,                 # Size of data (2D image, tuple)
                # HR target variable and its scaling parameters
                hr_variable:str = 'temp',           # Variable to load (temp or prcp)
                hr_model:str = 'DANRA',             # Model name (e.g. 'DANRA', 'ERA5')
                hr_scaling_method:str = 'zscore',   # Scaling method for high resolution data
                # LR conditions and their scaling parameters (not including geo variables. they are handled separately)
                lr_conditions:list = ['temp'],      # Variables to load as low resolution conditions
                lr_model:str = 'ERA5',              # Model name (e.g. 'DANRA', 'ERA5')
                lr_scaling_methods:list = ['zscore'], # Scaling methods for low resolution conditions
                lr_cond_dirs_zarr:Optional[dict] = None,      # Path to directories containing conditional data (in format dict({'condition1':dir1, 'condition2':dir2}))
                # NEW: LR conditioning area size (if cropping is desired)
                lr_data_size:Optional[tuple] = None,         # Size of low resolution data (2D image, tuple), e.g. (589,789) for full LR domain
                # Optionally a separate cutout domain for LR conditions
                lr_cutout_domains:Optional[list] = None,     # Domains to use for cutouts for LR conditions
                resize_factor: int = 1,             # Resize factor for input conditions (1 for full HR size, 2 for half HR size, etc. Mainly used for testing on smaller data)
                # Geo variables (stationary) and their full domain arrays
                geo_variables:Optional[list] = ['lsm', 'topo'], # Geo variables to load
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                # Seasonality variables for conditional sampling
                conditional_seasons:bool = False,   # Whether to use seasonal conditional sampling
                use_sin_cos_embedding: bool = False, # Whether to use sin-cos embedding for seasonal conditional sampling
                use_leap_years: bool = True,      # Whether to use leap years for day-of-year conditional sampling
                # Configuration information
                cfg: dict | None = None,
                split: str = "train",
                # Other dataset parameters
                n_samples:int = 365,                # Number of samples to load
                cache_size:int = 365,               # Number of samples to cache
                shuffle:bool = False,               # Whether to shuffle data (or load sequentially)
                cutouts:bool = False,               # Whether to use cutouts 
                cutout_domains:Optional[list] = None,         # Domains to use for cutouts
                n_samples_w_cutouts:Optional[int] = None,     # Number of samples to load with cutouts (can be greater than n_samples)
                sdf_weighted_loss:bool = False,     # Whether to use weighted loss for SDF
                scale:bool = True,                  # Whether to scale data to new interval
                save_original:bool = False,         # Whether to save original data
                n_classes:Optional[int] = None,                # Number of classes for conditional sampling
                fixed_cutout_hr: bool = False,         # Whether to use a fixed cutout (no random sampling)
                fixed_cutout_lr: bool = False,         # Whether to use a fixed cutout (no random sampling)
                fixed_hr_bounds: Optional[list] = None,  # Fixed cutout bounds for HR data (if fixed_cutout=True)
                fixed_lr_bounds: Optional[list] = None,  # Fixed cutout bounds for LR data (if fixed_cutout=True)
                ):                          
        '''
        Initializes the dataset.
        '''
        
        # Basic dataset parameters
        self.hr_variable_dir_zarr = hr_variable_dir_zarr
        self.n_samples = n_samples
        self.hr_data_size = hr_data_size
        self.cache_size = cache_size

        # LR conditions and scaling parameters
        # (Remove any geo variable from conditions list, if accidentally included)
        self.geo_variables = geo_variables
        # Check that there are the same number of scaling methods as conditions
        if len(lr_conditions) != len(lr_scaling_methods):
            raise ValueError('Number of conditions and scaling methods must be the same')

        # Go through the conditions, and if condition is in geo_variables, remoce from list, and remove scaling methods and params associated with it
        # But only if any geo_variables exist
        if self.geo_variables is not None:
            for geo_var in self.geo_variables:
                if geo_var in lr_conditions:
                    idx = lr_conditions.index(geo_var)
                    lr_conditions.pop(idx)
                    lr_scaling_methods.pop(idx)
                    
        self.lr_conditions = lr_conditions
        self.lr_model = lr_model
        self.lr_scaling_methods = lr_scaling_methods

        # Check for dual_lr or predict_residual with baseline_space in cfg
        self.dual_lr = cfg['lowres'].get('dual_lr', False) if cfg is not None and 'lowres' in cfg else False
        self.predict_residual = cfg['edm'].get('predict_residual', False) if cfg is not None and 'edm' in cfg else False
        self.lr_baseline_space = cfg['edm'].get('baseline_space', 'auto') if cfg is not None and 'edm' in cfg else 'auto'

        
        # If any conditions exist, set with_conditions to True
        self.with_conditions = len(self.lr_conditions) > 0

        # Save new LR parameters
        self.lr_data_size = lr_data_size
        self.lr_cutout_domains = lr_cutout_domains
        
        self.lr_cutout_name = cfg['lowres'].get('cutout_name', 'custom') if cfg is not None and 'lowres' in cfg else 'custom'

        # Check whether lr_cutout_domains are parsed as a list or tuple - even if 'None' - and set correctly to None if not
        if isinstance(self.lr_cutout_domains, list) or isinstance(self.lr_cutout_domains, tuple):
            if len(self.lr_cutout_domains) == 0 or (len(self.lr_cutout_domains) == 1 and str(self.lr_cutout_domains[0]).lower() == 'none'):
                self.lr_cutout_domains = None
            else:
                self.lr_cutout_domains = self.lr_cutout_domains
        
        
        # Specify target LR size (if different from HR size)
        self.target_lr_size = self.lr_data_size if self.lr_data_size is not None else self.hr_data_size
        
        # Resize factor for input conditions (for running with smaller data)
        self.resize_factor = resize_factor
        if self.resize_factor > 1:
            self.hr_size_reduced = (int(self.hr_data_size[0]/self.resize_factor), int(self.hr_data_size[1]/self.resize_factor))
            self.lr_size_reduced = (int(self.target_lr_size[0]/self.resize_factor), int(self.target_lr_size[1]/self.resize_factor))
        elif self.resize_factor == 1:
            self.hr_size_reduced = self.hr_data_size
            self.lr_size_reduced = self.target_lr_size
        else:
            raise ValueError('Resize factor must be greater than 0')


        # Save LR condition directories
        # lr_cond_dirs_zarr is a dict mapping each condition to its own zarr directory path
        self.lr_cond_dirs_zarr = lr_cond_dirs_zarr
        # Open each LR condition's zarr group and list its files
        self.lr_cond_zarr_dict = {} 
        self.lr_cond_files_dict = {}
        if self.lr_cond_dirs_zarr is not None:
            for cond in self.lr_cond_dirs_zarr:
                logger.info(f'Loading zarr group for condition {cond}')
                # logger.info(f'Path to zarr group: {self.lr_cond_dirs_zarr[cond]}')
                group = zarr.open_group(self.lr_cond_dirs_zarr[cond], mode='r')
                self.lr_cond_zarr_dict[cond] = group
                self.lr_cond_files_dict[cond] = list(group.keys())
        else:
            raise ValueError('LR condition directories (lr_cond_dirs_zarr) must be provided as a dictionary.')

        # HR target variable parameters
        self.hr_variable = hr_variable
        self.hr_model = hr_model
        self.hr_scaling_method = hr_scaling_method

        # ========= Dual-LR/main LR scaling preferences =========
        self.lr_main_var_scale = (cfg['lowres'].get('lr_main_var_scale', 'HR') if cfg is not None and 'lowres' in cfg else 'HR')
        fall_back_method = 'log_zscore' if self.hr_variable in ['prcp', 'tp', 'cape'] else 'zscore'
        self.lr_main_var_scale_method = (cfg['lowres'].get('lr_main_var_scale_method', fall_back_method) if cfg is not None and 'lowres' in cfg else fall_back_method)

        # Name used in stats files for combined HR+LR statistics (fallback to HR if missing)
        self.combined_stats_model_name = (cfg['lowres'].get('combined_stats_model_name', 'DANRA_ERA5') if cfg is not None and 'lowres' in cfg else 'DANRA_ERA5')

        # Identify the "main" LR condition (matching the HR target variable, if existing)
        self.main_lr_cond = None
        if self.hr_variable in self.lr_conditions:
            self.main_lr_cond = self.hr_variable
        else:
            # Common alias pairs for mapping HR var to LR names
            alias_pairs = [('prcp', 'tp'), ('tp', 'prcp'), ('temp', 't2m'), ('t2m', 'temp')]
            for hr_name, lr_alias in alias_pairs:
                if self.hr_variable == hr_name and lr_alias in self.lr_conditions:
                    self.main_lr_cond = lr_alias
                    logger.info(f"Main LR condition for HR variable '{self.hr_variable}' set to '{self.main_lr_cond}' using alias mapping.")
                    break
        if self.main_lr_cond is None:
            logger.warning(f"Could not identify a main LR condition matching HR variable '{self.hr_variable}'. Dual-LR logic will be skipped; using standard per-condition scaling.")


        # Global epsilon for log scaling to avoid log(0)
        self.glob_prcp_epsilon = cfg['transforms'].get('prcp_eps', 0.01) if cfg is not None else 0.01
        
        # Save geo variables full-domain arrays
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain

        # Cache seasonality parameters
        self.conditional_seasons = conditional_seasons
        self.use_sin_cos_embedding = use_sin_cos_embedding
        self.use_leap_years = use_leap_years

        # Save classifier-free guidance parameters
        self.cfg = cfg
        self.split = split

        # Save what split statistics to use for scaling (train, valid, test or all). ALMOST ALWAYS USE 'train' TO AVOID DATA LEAKAGE 
        self.scaling_split = cfg['transforms']['scaling_split'] if cfg is not None else 'train'

        # Save other parameters
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.hr_cutout_domains = cutout_domains
        self.hr_cutout_name = cfg['highres'].get('cutout_name', 'custom') if cfg is not None and 'highres' in cfg else 'custom'
        self.sdf_weighted_loss = sdf_weighted_loss
        self.scale = scale
        self.save_original = save_original
        self.n_classes = n_classes
        self.n_samples_w_cutouts = self.n_samples if n_samples_w_cutouts is None else n_samples_w_cutouts

        # ========= Fixed crop (stationary) knob =========
        self.fixed_cutout_hr = fixed_cutout_hr
        self.fixed_cutout_lr = fixed_cutout_lr

        self.fixed_hr_bounds = fixed_hr_bounds
        self.fixed_lr_bounds = fixed_lr_bounds

        # If fixed cutout is True, fixed bounds must be provided and set up
        if self.fixed_cutout_hr:
            # Make sure that fixed_hr_bounds is provided and valid - if not, set fixed_cutout_hr to False
            if self.fixed_hr_bounds is None or len(self.fixed_hr_bounds) != 4:
                self.fixed_hr_bounds = None
                self.fixed_cutout_hr = False
                logger.warning('Fixed cutout for HR is set to True, but fixed_hr_bounds is not provided or invalid. Setting fixed_cutout_hr to False.')
            else:
                # Check that the bounds are valid
                if (self.fixed_hr_bounds[1] - self.fixed_hr_bounds[0] != self.hr_data_size[1]) or (self.fixed_hr_bounds[3] - self.fixed_hr_bounds[2] != self.hr_data_size[0]):
                    raise ValueError('Fixed HR cutout bounds are not valid. They must match the HR data size.')
                else:
                    logger.info(f'Using fixed cutout for HR with bounds: {self.fixed_hr_bounds}')
        else:
            self.fixed_hr_bounds = None
            logger.info('Not using fixed cutout for HR.')

        if self.fixed_cutout_lr:
            # Make sure that fixed_lr_bounds is provided and valid - if not, set fixed_cutout_lr to False
            if self.fixed_lr_bounds is None or len(self.fixed_lr_bounds) != 4:
                self.fixed_lr_bounds = None
                self.fixed_cutout_lr = False
                logger.warning('Fixed cutout for LR is set to True, but fixed_lr_bounds is not provided or invalid. Setting fixed_cutout_lr to False.')
            else:
                # Check that the bounds are valid
                if (self.fixed_lr_bounds[1] - self.fixed_lr_bounds[0] != self.target_lr_size[1]) or (self.fixed_lr_bounds[3] - self.fixed_lr_bounds[2] != self.target_lr_size[0]):
                    raise ValueError('Fixed LR cutout bounds are not valid. They must match the target LR data size.')
                else:
                    logger.info(f'Using fixed cutout for LR with bounds: {self.fixed_lr_bounds}')
        else:
            self.fixed_lr_bounds = None
            logger.info('Not using fixed cutout for LR.')

        # Build file maps based on the date in the file name      
        # Open main (HR) zarr group, and get HR file keys (pure filenames)
        self.zarr_group_img = zarr.open_group(hr_variable_dir_zarr, mode='r')
        hr_files_all = list(self.zarr_group_img.keys())
        self.hr_file_map = {}
        for file in hr_files_all:
            try:
                date = FileDate(file)
                self.hr_file_map[date] = file
            except Exception as e:
                logger.warning(f"Could not extract date from file {file}. Skipping file. Error: {e}")
                

        # For each LR condition, build a file map: date -> file key
        self.lr_file_map = {}
        for cond in self.lr_conditions:
            self.lr_file_map[cond] = {}
            for file in self.lr_cond_files_dict[cond]:
                try:
                    date = FileDate(file)
                    self.lr_file_map[cond][date] = file
                except Exception as e:
                    logger.warning(f"Could not extract date from file {file} for condition {cond}. Skipping file. Error: {e}")

        # Compute common dates across HR and all LR conditions
        common_dates = set(self.hr_file_map.keys())
        for cond in self.lr_conditions:
            common_dates = common_dates.intersection(set(self.lr_file_map[cond].keys()))
        self.common_dates = sorted(list(common_dates))
        if len(self.common_dates) < self.n_samples:
            self.n_samples = len(self.common_dates)
            logger.warning(f"Not enough common dates ({len(self.common_dates)}) to sample {self.n_samples} samples. Reducing n_samples to {self.n_samples}.")

        # if self.shuffle:
        #     self.common_dates = random.sample(self.common_dates, self.n_samples)
        # === Selection policy for date list ===
        # Default behaviour (train/val): follow self.shuffle as before. For generation split, allow fixed dates that repeat every epohc, with either sequential (subsequent) or random selection dates 
        is_gen_split = str(self.split).lower() in ('gen', 'generation', 'test', 'inference')
        if is_gen_split:
            gen_cfg_root = (self.cfg or {}).get('generation', None)
            if gen_cfg_root is not None:
                fixed_dates = bool((gen_cfg_root or {}).get('fixed_dates', True))
                random_dates = bool((gen_cfg_root or {}).get('random_dates', False))
                seed = int((gen_cfg_root or {}).get('seed', 509))
                start_index = int((gen_cfg_root or {}).get('start_index', 0))

                if fixed_dates:
                    # Build a deterministic selection once per Dataset construction
                    if random_dates:
                        rng = random.Random(seed)
                        # Deterministic sample w/out replacement
                        if self.n_samples > len(self.common_dates):
                            logger.warning(f"Not enough common dates ({len(self.common_dates)}) to sample {self.n_samples} fixed random samples. Reducing n_samples to {len(self.common_dates)}.")
                            self.n_samples = len(self.common_dates)
                        self.common_dates = rng.sample(self.common_dates, self.n_samples)
                        logger.info(f"Using {self.n_samples} fixed random dates for generation split with seed {seed}.")
                    else:
                        # Sequential (subsequent) block starting at start_index
                        if start_index < 0 or start_index >= len(self.common_dates):
                            start_index = 0
                        end_index = start_index + self.n_samples
                        if end_index > len(self.common_dates):
                            logger.warning(f"[gen] start_index + n_samples exceeds available common dates. Wrapping around.")
                            start_index = 0
                            end_index = min(self.n_samples, len(self.common_dates))
                        self.common_dates = self.common_dates[start_index:end_index]
                        logger.info(f"Using N={self.n_samples} fixed sequential dates for generation split starting at index {start_index}.")
                    # Ensure stable ordering and no per-epoch reshuffle
                    self.shuffle = False
                else:
                    # Not fixed: retain legacy behaviour controlled by self.shuffle
                    if self.shuffle:
                        self.common_dates = random.sample(self.common_dates, self.n_samples)
            else:
                logger.warning(f"Generation split selected but no generation config found in cfg. Retaining legacy behaviour controlled by self.shuffle.")
                if self.shuffle:
                    self.common_dates = random.sample(self.common_dates, self.n_samples)
        else:
            # Non-generation splits keep legacy shuffle behaviour
            if self.shuffle:
                self.common_dates = random.sample(self.common_dates, self.n_samples)

        
        # Set cache for data loading - if cache_size is 0, no caching is used
        # If num_workers > 0 each worker has its own Dataset instance
        # self.cache = multiprocessing.Manager().dict()
        self.cache = {}  # Use a simple dict for caching, if cache_size > 0

        if self.scale:
            # 1. Set condition transforms
            self.lr_transforms_dict = {}
            domain_str_hr = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg is not None else f"{self.hr_data_size[0]}x{self.hr_data_size[1]}"
            domain_str_lr = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg is not None else f"{self.target_lr_size[0]}x{self.target_lr_size[1]}"
            crop_region_hr = cfg['highres']['cutout_domains'] if (cfg is not None and self.cutouts and self.hr_cutout_domains is not None) else "full"
            crop_region_hr_str = '_'.join(map(str, crop_region_hr)) # if (cfg is not None and self.cutouts and self.hr_cutout_domains is not None) else "full"
            crop_region_lr = cfg['lowres']['cutout_domains'] if (cfg is not None and self.cutouts and self.lr_cutout_domains is not None) else "full"
            crop_region_lr_str = '_'.join(map(str, crop_region_lr)) # if (cfg is not None and self.cutouts and self.lr_cutout_domains is not None) else "full"
            scaling_split = self.scaling_split
            stats_load_dir = cfg['paths']['stats_load_dir'] if cfg is not None else './stats'
            self.hr_buffer_frac = cfg['highres'].get('buffer_frac', 0.05) if cfg is not None and 'highres' in cfg else 0.05
            self.lr_buffer_frac = cfg['lowres'].get('buffer_frac', 0.05) if cfg is not None and 'lowres' in cfg else 0.05

            for cond_var, trans_type in zip(self.lr_conditions, self.lr_scaling_methods):
                logger.info(f"LR condition: {cond_var}, scaling method: {trans_type}")
                # Common prefix for all LR transforms
                prefix = [
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced)
                ]
                eps_val = self.glob_prcp_epsilon if cond_var in ['prcp', 'tp', 'cape'] else 0.0

                # Is this the main LR condition matching the HR variable?
                is_main = (self.main_lr_cond is not None and cond_var == self.main_lr_cond)

                if is_main and self.dual_lr:
                    # Build TWO channels: (A) combined HR+LR stats (fallback to HR if missing), (B) standard LR-only stats

                    # (A) Combined HR+LR stats
                    try: 
                        t_combined = transforms.Compose(prefix + [
                            get_transforms_from_stats(
                                variable=cond_var,
                                model=self.combined_stats_model_name, # TODO: Should be recalculated in stats module and named DANRA_ERA5
                                domain_str=domain_str_lr,
                                crop_region_str=crop_region_lr_str,
                                scaling_split=scaling_split,
                                transform_type=trans_type,
                                buffer_frac=self.lr_buffer_frac,
                                stats_file_path=stats_load_dir,
                                eps=eps_val,
                            )
                        ])
                        logger.info(f"Using combined HR+LR stats '{self.combined_stats_model_name}' for dual LR main channel A of condition '{cond_var}'")
                    except Exception as e:
                        logger.warning(f"Failed to load combined HR+LR '{self.combined_stats_model_name}' for condition '{cond_var}'. Falling back to HR stats for channel A. Error: {e}")
                        t_combined = transforms.Compose(prefix + [
                            get_transforms_from_stats(
                                variable=cond_var,
                                model=self.hr_model,
                                domain_str=domain_str_lr,
                                crop_region_str=crop_region_lr_str,
                                scaling_split=scaling_split,
                                transform_type=trans_type,
                                buffer_frac=self.hr_buffer_frac,
                                stats_file_path=stats_load_dir,
                                eps=eps_val,
                            )
                        ])
                    # (B) Standard LR-only stats
                    t_lr_only = transforms.Compose(prefix + [
                        get_transforms_from_stats(
                            variable=cond_var,
                            model=self.lr_model,
                            domain_str=domain_str_lr,
                            crop_region_str=crop_region_lr_str,
                            scaling_split=scaling_split,
                            transform_type=trans_type,
                            buffer_frac=self.lr_buffer_frac,
                            stats_file_path=stats_load_dir,
                            eps=eps_val,
                        )
                    ])

                    # Wrap into a callable that returns a stacked 2-channel tensor
                    self.lr_transforms_dict[cond_var] = DualLRTransform(t_combined, t_lr_only)
                    logger.info(f"Dual-LR transform set for main condition '{cond_var}': returning 2-channels [combined, LR-only].")
                elif is_main and not self.dual_lr:
                    # Single channel, but choose statistics source as per lr_main_var_scale
                    scale_mode = str(self.lr_main_var_scale).upper()
                    if scale_mode == 'HR':
                        stats_model = self.hr_model
                        ds, cr = domain_str_hr, crop_region_hr_str
                    elif scale_mode == 'LR':
                        stats_model = self.lr_model
                        ds, cr = domain_str_lr, crop_region_lr_str
                    else: # Combined 'HR_LR' by default
                        stats_model = self.combined_stats_model_name
                        ds, cr = domain_str_lr, crop_region_lr_str # NOTE: Need to consider with different domain sizes
                    try:
                        transform_list = prefix + [
                            get_transforms_from_stats(
                                variable=cond_var,
                                model=stats_model,
                                domain_str=ds,
                                crop_region_str=cr,
                                scaling_split=scaling_split,
                                transform_type=trans_type,
                                buffer_frac=self.lr_buffer_frac,
                                stats_file_path=stats_load_dir,
                                eps=eps_val,
                            )
                        ]
                        logger.info(f"Main LR condition '{cond_var}' scaled using '{scale_mode}' stats from model '{stats_model}'.")
                    except Exception as e:
                        # Fallbacks: combined -> HR, otherwise LR
                        if scale_mode == 'HR_LR':
                            logger.warning(f"Combined stats '{stats_model}' not found for main condition '{cond_var}'. Falling back to HR stats. Error: {e}")
                            transform_list = prefix + [
                                get_transforms_from_stats(
                                    variable=cond_var,
                                    model=self.hr_model,
                                    domain_str=domain_str_hr,
                                    crop_region_str=crop_region_hr_str,
                                    scaling_split=scaling_split,
                                    transform_type=trans_type,
                                    buffer_frac=self.hr_buffer_frac,
                                    stats_file_path=stats_load_dir,
                                    eps=eps_val,
                                )
                            ]
                        else:
                            logger.warning(f"Requested stats '{stats_model}' not found for main condition '{cond_var}'. Falling back to LR stats. Error: {e}")
                            transform_list = prefix + [
                                get_transforms_from_stats(
                                    variable=cond_var,
                                    model=self.lr_model,
                                    domain_str=domain_str_lr,
                                    crop_region_str=crop_region_lr_str,
                                    scaling_split=scaling_split,
                                    transform_type=trans_type,
                                    buffer_frac=self.lr_buffer_frac,
                                    stats_file_path=stats_load_dir,
                                    eps=eps_val,
                                )
                            ]
                    self.lr_transforms_dict[cond_var] = transforms.Compose(transform_list)
                else:
                    # Not main condition - standard single-channel LR-only stats
                    transform_list = prefix + [
                        get_transforms_from_stats(
                            variable=cond_var,
                            model=self.lr_model,
                            domain_str=domain_str_lr,
                            crop_region_str=crop_region_lr_str,
                            scaling_split=scaling_split,
                            transform_type=trans_type,
                            buffer_frac=self.lr_buffer_frac,
                            stats_file_path=stats_load_dir,
                            eps=eps_val,
                        )
                    ]
                    self.lr_transforms_dict[cond_var] = transforms.Compose(transform_list)

                    
            # 2. Set HR target transform
            hr_transform_list = [
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced)
            ]
            hr_transform_list.append(get_transforms_from_stats(
                variable=self.hr_variable,
                model=self.hr_model,
                domain_str=domain_str_hr,
                crop_region_str=crop_region_hr_str,
                scaling_split=scaling_split,
                transform_type=self.hr_scaling_method,
                buffer_frac=self.hr_buffer_frac,
                stats_file_path=stats_load_dir,
                eps=self.glob_prcp_epsilon if self.hr_variable in ['prcp', 'tp'] else 0.0,
            ))
            self.hr_transform = transforms.Compose(hr_transform_list)
        
            # 3. Set geo variable transforms (if any)
            if self.geo_variables is not None:
                if self.topo_full_domain is None:
                    raise ValueError("topo_full_domain must be provided if 'topo' is in geo_variables")
                topo_scale_min = cfg['stationary_conditions']['geographic_conditions']['norm_min'] if (cfg is not None and 'stationary_conditions' in cfg and 'geographic_conditions' in cfg['stationary_conditions']) else -1.0
                topo_scale_max = cfg['stationary_conditions']['geographic_conditions']['norm_max'] if (cfg is not None and 'stationary_conditions' in cfg and 'geographic_conditions' in cfg['stationary_conditions']) else 1.0
                logger.info(f"Topography will be scaled to [{topo_scale_min}, {topo_scale_max}] interval.")
                # Update topo scaling transform to use cfg values if provided
                self.geo_transform_topo = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced, mode='bilinear', align_corners=False),
                    Scale(topo_scale_min, topo_scale_max, self.topo_full_domain.min(), self.topo_full_domain.max())
                ])
                self.geo_transform_lsm = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced, mode='nearest'), # Nearest for categorical data
                ])
        else:
            # 1. Set condition transforms
            self.lr_transforms_dict = {cond: transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.lr_size_reduced)
            ]) for cond in self.lr_conditions}

            # 2. Set HR target transform
            self.hr_transform = transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced)
            ])

            # 3. Set geo variable transforms (if any)
            if self.geo_variables is not None:
                self.geo_transform_topo = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced)
                ])
                self.geo_transform_lsm = self.geo_transform_topo

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.common_dates)

    def _domains_equivalent(self, a, b, name_a, name_b):
        """ Return Tru if two cutout-domain specs are semantically equivalent.
        """
        if a is None or b is None:
            return False
        
        try:
            if list(a) == list(b):
                if name_a != name_b:
                    logger.warning(f"{name_a} and {name_b} are different but have the same cutout domains: {a}. Assuming not equivalent.")
                    return False
                return True
            return False
        except Exception as e:
            if a == b:
                if name_a != name_b:
                    logger.warning(f"{name_a} and {name_b} are different but have the same cutout domains: {a}. Assuming not equivalent.")
                    return False
                return True
            return False
        
    @staticmethod
    def _map_point_to_size(point, src_size, dst_size):
        """
            Map a crop point [x1, x2, y1, y2] from src_size=(H,W) by scale factors.
            Always returns integers, preserving order [x1_new, x2_new, y1_new, y2_new].
        """
        if src_size == dst_size:
            return point
        sx = dst_size[1] / float(src_size[1]) # Width scale factor
        sy = dst_size[0] / float(src_size[0]) # Height scale factor
        x1 = int(round(point[0] * sx))
        x2 = int(round(point[1] * sx))
        y1 = int(round(point[2] * sy))
        y2 = int(round(point[3] * sy))
        return [x1, x2, y1, y2]
    
    @staticmethod
    def _valid_bounds(b):
        return (b is not None) and hasattr(b, '__len__') and (len(b) == 4) 

    def _compute_crop_points(self):
        """
            Decide (hr_point, lr_point) with independent fixed-cutout knobs. Based on HR and LR ROI domains. (ROI = region of interest)

            Conventions:
            - Points are [x1, x2, y1, y2] in pixel indices on their *native* grids.
            - Slicing is [y1:y2, x1:x2] elsewhere in the code.
            - self.hr_data_size = (H_hr, W_hr); self.target_lr_size = (H_lr, W_lr).

            Priority:
            1) If fixed_cutout_hr==True and fixed_hr_bounds valid:
                hr_point = fixed_hr_bounds
                lr_point = (fixed_lr_bounds if fixed_cutout_lr and valid) else map HR→LR (or same if no lr_data_size)
                return
            2) Else if fixed_cutout_lr==True and fixed_lr_bounds valid:
                lr_point = fixed_lr_bounds
                hr_point = map LR→HR  (co-locate if domains align)  # preferred
                            (else fall back to HR random if HR domain truly unrelated)
                return
            3) Else if cutouts==False:
                return (None, None)
            4) Else (random HR crop):
                hr_point = random from HR cutout domain
                lr_point = map HR→LR if LR domain equivalent/unspecified, else random from LR domain
                return
        """
        # ----- Case 1: HR is fixed (authoritative), LR follows
        if self.fixed_cutout_hr and self._valid_bounds(self.fixed_hr_bounds):
            hr_point = [int(v) for v in self.fixed_hr_bounds]  # authoritative HR ROI # type: ignore

            # LR decision
            if self.lr_data_size is None:
                lr_point = hr_point  # same indices if LR uses HR grid/size
            else:
                if self.fixed_cutout_lr and self._valid_bounds(self.fixed_lr_bounds):
                    lr_point = [int(v) for v in self.fixed_lr_bounds] # authoritative LR ROI # type: ignore
                else:
                    # co-locate LR by mapping HR→LR
                    lr_point = self._map_point_to_size(hr_point, self.hr_data_size, self.target_lr_size)
            return hr_point, lr_point

        # Warn once if HR was requested fixed but bounds invalid
        if self.fixed_cutout_hr and not self._valid_bounds(self.fixed_hr_bounds):
            logger.warning(f"fixed_cutout_hr=True but fixed_hr_bounds invalid: {self.fixed_hr_bounds}. "
                        "HR will not be fixed; proceeding with LR/normal policy.")

        # ----- Case 2: LR is fixed (authoritative), HR follows
        if self.fixed_cutout_lr and self._valid_bounds(self.fixed_lr_bounds):
            lr_point = [int(v) for v in self.fixed_lr_bounds]  # authoritative LR ROI # type: ignore

            # Prefer to co-locate HR by mapping LR→HR when domains are equivalent/unspecified.
            # If LR domain is clearly unrelated to HR domain, fall back to HR random (to avoid nonsense mapping).
            domains_same = self._domains_equivalent(self.lr_cutout_domains, self.hr_cutout_domains,
                                                    self.lr_cutout_name, self.hr_cutout_name)
            if domains_same or (self.lr_cutout_domains is None):
                hr_point = self._map_point_to_size(lr_point, self.target_lr_size, self.hr_data_size)
            else:
                # HR domain differs materially; choose a valid HR crop instead of blind mapping
                if not self.cutouts:
                    hr_point = None
                else:
                    hr_point = find_rand_points(self.hr_cutout_domains, self.hr_data_size)
                    # NOTE: If you *want* forced co-location even for different named domains,
                    # replace the line above with the mapping call and accept possible mismatch.
            return hr_point, lr_point

        # Warn once if LR was requested fixed but bounds invalid
        if self.fixed_cutout_lr and not self._valid_bounds(self.fixed_lr_bounds):
            logger.warning(f"fixed_cutout_lr=True but fixed_lr_bounds invalid: {self.fixed_lr_bounds}. "
                        "LR will not be fixed; proceeding with HR/normal policy.")

        # ----- Case 3: No cutouts → full domain (points unused)
        if not self.cutouts:
            return None, None

        # ----- Case 4: Random HR crop, LR follows policy
        hr_point = find_rand_points(self.hr_cutout_domains, self.hr_data_size)

        if self.lr_data_size is None:
            lr_point = hr_point
        else:
            domains_same = self._domains_equivalent(self.lr_cutout_domains, self.hr_cutout_domains,
                                                    self.lr_cutout_name, self.hr_cutout_name)
            if (self.lr_cutout_domains is None) or domains_same:
                lr_point = self._map_point_to_size(hr_point, self.hr_data_size, self.target_lr_size)
            else:
                lr_point = find_rand_points(self.lr_cutout_domains, self.target_lr_size)

        return hr_point, lr_point

    def _addToCache(self, idx:int, data):
        '''
            Add item to cache. 
            If cache is full, remove random item from cache.
            Input:
                - idx: index of item to add to cache
                - data: data to add to cache
        '''
        # If cache_size is 0, no caching is used
        if self.cache_size > 0:
            # If cache is full, remove random item from cache
            if len(self.cache) >= self.cache_size:
                # Get keys from cache
                keys = list(self.cache.keys())
                # Select random key to remove
                key_to_remove = random.choice(keys)
                # Remove key from cache
                self.cache.pop(key_to_remove, None) # Safe removal, in case key is not found
            # Add data to cache
            self.cache[idx] = data

    @staticmethod
    def _validate_season_y(y: torch.Tensor, use_sincos: bool, where: str = ""):
        """
            Strict runtime validation of seasonal label encoding. Raises on mismatch.
            'where' is a short tag to identify the call site in error messages.
        """
        if y is None:
            return
        if use_sincos:
            if not torch.is_floating_point(y):
                raise TypeError(f"[DOY-check{where}] Expected float sin/cos tensor, got {y.dtype}.")
            # Tolerate [1,2] from some collates: squeeze batch if present
            if y.ndim == 2 and y.shape[0] == 1 and y.shape[1] == 2:
                y = y.squeeze(0)
            if y.ndim != 1 or y.shape[0] != 2:
                raise ValueError(f"[DOY-check{where}] Expected shape [2] for sin/cos tensor, got {y.shape}.")
            m = float(torch.min(y)); M = float(torch.max(y))
            if m < -1.05 or M > 1.05:
                raise ValueError(f"[DOY-check{where}] Sin/cos values out of range [-1,1]: min {m}, max {M}.")
        else:
            # Categorical season expected as Long scalar (keep 1..4 convention)
            if y.dtype not in (torch.long, torch.int64):
                raise TypeError(f"[DOY-check{where}] Expected Long season index, got dype {y.dtype}.")
            # Accept scalar or 1-element tensor (including [1] from some collates)
            if y.ndim > 1 or (y.ndim == 1 and y.numel() != 1):
                raise ValueError(f"[DOY-check{where}] Expected scalar or 1-element tensor for season index, got shape {y.shape}.")

    def _build_seasonal_label(self, hr_file_name: str) -> torch.Tensor:
        """
            Single source of truth for seasonal label construction.
            Returns:
                y: 
                  - Float[2] in [-1,1] if self.use_sin_cos_embedding is True
                  - Long scalar for categorical, depending on self.n_classes (4, 12, 365, 366)
        """
        # 1) Parse date once
        dateObj = DateFromFile(hr_file_name)

        # 2) Sin/cos embedding
        if self.use_sin_cos_embedding:
            doy = dateObj.determine_day()
            if self.use_leap_years:
                N = 366 if DateFromFile.is_leap_year(dateObj.year) else 365
                theta = 2.0 * math.pi * float(doy - 1) / float(N) 
            else:
                # Avoid 2*pi wrap: fold Feb 29 onto Feb 28 in leap years
                if doy == 60 and DateFromFile.is_leap_year(dateObj.year):
                    doy = 59
                theta = 2.0 * math.pi * float(doy - 1) / 365.0
            y = torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float32)
            self._validate_season_y(y, use_sincos=True, where=" /dataset-build")
            return y
        
        # 3) Categorical encoding - zero-based
        n_classes = 12 if (self.n_classes is None) else int(self.n_classes)

        if n_classes == 4:
            cls = dateObj.determine_season()   # 1..4
        elif n_classes == 12:
            cls = dateObj.determine_month()   # 1..12
        elif n_classes in (365, 366):
            doy = dateObj.determine_day()
            if n_classes == 365:
                # Merge Feb 29 into Feb 28 for non-leap-year encoding
                if DateFromFile.is_leap_year(dateObj.year) and doy == 60: # Feb 29
                    doy = 59
            cls = doy   # 0..365 or 0..366
        else:
            raise ValueError(f"Unsupported n_classes={n_classes}. Supported: 4, 12, 365, 366.")
        
        y = torch.tensor(cls, dtype=torch.long)
        self._validate_season_y(y, use_sincos=False, where=" /dataset-build")
        return y

    def __getitem__(self, idx:int):
        '''
            For each sample:
            - Loads LR conditions from the main zarr group (and, if applicable, from additional condition directories)
            - Loads HR target variable from the main zarr group
            - Loads the stationary geo variables (lsm and topo) from provided full-domain arrays
            - Applies cutouts and the appropriate transforms
        '''

        if self.cache_size > 0 and (self.split != 'train' or not self.cutouts):
            cached = self.cache.get(idx, None)
            if cached is not None:
                return cached

        sample_dict = {}
        # Get the common date corresponding to the index
        date = self.common_dates[idx]
        sample_dict['date'] = date


        # Determine crop region, centralized policy
        hr_point, lr_point = self._compute_crop_points()
        # Add a bit of logging
        logger.debug(f"Index {idx}, date {date}, hr_point {hr_point}, lr_point {lr_point}")
        

        # Look up HR file using the common date
        hr_file_name = self.hr_file_map[date]

        # Look up LR files for each condition using the common date
        for cond in self.lr_conditions:
            lr_file_name = self.lr_file_map[cond][date]
            # Load LR condition data from its own zarr group
            try:
                data = _extract_2d_from_zarr_entry(self.lr_cond_zarr_dict[cond], lr_file_name, cond)
                # Apply unit corrections consistently
                data = correct_variable_units(cond, self.lr_model, data)
            except Exception as e:
                logger.error(f'Error loading {cond} data for {lr_file_name}. Error: {e}')
                data = None
            
            # Crop LR data using lr_point if cutouts are enabled and lr_point is not None
            if self.cutouts and data is not None and lr_point is not None:
                # lr_point is in format [x1, x2, y1, y2]
                data = data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
                logger.debug(f"Cropped {cond} data to shape {data.shape} using lr_point {lr_point}")
            # logger.debug(f"Data shape for {cond}: {data.shape if data is not None else None}")
                
            # If save_original is True, save original conditional data
            if self.save_original:
                sample_dict[f"{cond}_lr_original"] = data.copy() if data is not None else None

            # Apply specified transform (specific to various conditions)
            if data is not None and self.lr_transforms_dict.get(cond, None) is not None:
                data = self.lr_transforms_dict[cond](data)
            sample_dict[cond + "_lr"] = data
        

        # Load HR target variable data
        try:
            hr_np = _extract_2d_from_zarr_entry(self.zarr_group_img, hr_file_name, self.hr_variable)
            # Apply unit corrections consistently
            hr_np = correct_variable_units(self.hr_variable, self.hr_model, hr_np)
            hr = torch.tensor(hr_np, dtype=torch.float32)
        except Exception as e:
            logger.error(f'Error loading HR {self.hr_variable} data for {hr_file_name}. Error: {e}')
            hr = None

        if self.cutouts and (hr is not None) and (hr_point is not None):
            hr = hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
            logger.debug(f"Cropped HR data to shape {hr.shape} using hr_point {hr_point}")
        if self.save_original and (hr is not None):
            sample_dict[f"{self.hr_variable}_hr_original"] = hr.clone()
        if hr is not None:
            hr = self.hr_transform(hr)
        sample_dict[self.hr_variable + "_hr"] = hr

        # Process a separate HR mask for geo variables (if 'lsm' is needed for HR SDF and masking HR images)
        if self.geo_variables is not None and 'lsm' in self.geo_variables and self.lsm_full_domain is not None:
            lsm_hr = self.lsm_full_domain
            if self.cutouts and lsm_hr is not None and hr_point is not None:
                lsm_hr = lsm_hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
            # Ensure the mask is contiguous and transform
            lsm_hr = np.ascontiguousarray(lsm_hr)
            # Separate geo transform, with resize to HR size
            geo_transform_lsm_hr = transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced, mode='nearest')  # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
            ])
            lsm_hr = geo_transform_lsm_hr(lsm_hr)
            # Re-binarize after resizing, just in case
            lsm_hr = (lsm_hr > 0.5).to(lsm_hr.dtype)  # type: ignore # Ensure binary mask (0 and 1)
            sample_dict['lsm_hr'] = lsm_hr


        # Load geo variables (stationary) from full-domain arrays (may be cropped using lr_data_size and lr_cutout_domains)
        if self.geo_variables is not None:
            for geo in self.geo_variables:
                if geo == 'lsm':
                    if self.lsm_full_domain is None:
                        raise ValueError("lsm_full_domain must be provided if 'lsm' is in geo_variables")
                    geo_data = self.lsm_full_domain
                    # logger.info('lsm_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_lsm
                elif geo == 'topo':
                    if self.topo_full_domain is None:
                        raise ValueError("topo_full_domain must be provided if 'topo' is in geo_variables")
                    geo_data = self.topo_full_domain
                    # logger.info('topo_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_topo
                else:
                    # Add custom logic for other geo variables when needed
                    geo_data = None
                    geo_transform = None
                if geo_data is not None and self.cutouts:
                    # For geo data, if an LR-specific size and domain are provided, use lr_point
                    if self.lr_data_size is not None and self.lr_cutout_domains is not None and lr_point is not None:
                        geo_data = geo_data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
                    else:
                        if hr_point is not None:
                            geo_data = geo_data[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
                
                if geo_data is not None and geo_transform is not None:
                    geo_data = geo_transform(geo_data)
                    # For lsm, re-binarize after resizing, just in case
                    if geo == 'lsm':
                        geo_data = (geo_data > 0.5).to(geo_data.dtype)  # Ensure binary mask (0 and 1)

                sample_dict[geo] = geo_data
        
        # Seasonal/DOY sin-cos label construction 
        if self.conditional_seasons:
            y = self._build_seasonal_label(hr_file_name)
            sample_dict['y'] = y
        else:
            if 'y' in sample_dict:
                del sample_dict['y']
            

        # For SDF, ensure that it is computed for the HR mask (lsm_hr) to get it in same shape as HR
        if self.sdf_weighted_loss:
            if 'lsm_hr' in sample_dict and sample_dict['lsm_hr'] is not None:
                sdf = generate_sdf(sample_dict['lsm_hr'])
                sdf = normalize_sdf(sdf)
                sample_dict['sdf'] = sdf
            else:
                raise ValueError("lsm_hr must be provided for SDF computation if sdf_weighted_loss is True")

        # Attach cutout points for reference
        if self.cutouts:
            sample_dict['hr_points'] = hr_point
            sample_dict['lr_points'] = lr_point

        # Add item to cache
        self._addToCache(idx, sample_dict)

        return sample_dict #sample

    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        date = self.common_dates[idx]
        return date #self.hr_file_map[date]

