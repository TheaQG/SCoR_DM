import os 
import re
import zarr 
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from glob import glob
from datetime import datetime

from data_analysis_pipeline.stats_analysis.path_utils import build_data_path
from scor_dm.variable_utils import crop_to_region, get_var_name_short, correct_variable_units
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_date_from_filename(file_path, data_type, var_name_short):
    """
        Extract the date from the filename.
    """

    # Get filename from file path (basename means removing the directory)
    filename = os.path.basename(file_path)

    if data_type == 'npz':
        # Extract date from filename like varname_YYYYMMDD.npz
         match = re.search(r"(\d{8})\.npz$", filename) # match = re.search(rf"{re.escape(var_name_short)}_(\d{8})\.npz$", filename)        
    elif data_type == 'zarr':
        # For zarr, filepath is like {split}.zarr/varname_YYYYMMDD
        match = re.search(r"(\d{8})$", filename)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    if match:
        date_str = match.group(1)
        timestamp = datetime.strptime(date_str, "%Y%m%d")
    else:
        raise ValueError(f"Date not found in filename: {filename}")

    return timestamp

def process_single_file(file_path, variable, model, var_name_short, crop_region):
    """
        Reads file, applies variable correction, crops region etc.
    """
    if file_path.endswith(".npz"):
        with np.load(file_path) as npz:
            # Take the first available key if neither 'data' nor 'arr_0' exists
            key = 'data' if 'data' in npz else ('arr_0' if 'arr_0' in npz else list(npz.keys())[0])
            data = npz[key]
            data = np.array(data)  # Ensure it's a numpy array 

            # Extract date from filename like 'prcp_20200101.npz'
            timestamp = get_date_from_filename(file_path, 'npz', var_name_short)

    elif re.match(r".*\.zarr/" + var_name_short + r"_\d{8}", file_path):
        # Load from Zarr archive subpath
        zarr_root, key = os.path.split(file_path)
        zarr_group = zarr.open_group(zarr_root, mode='r')
        try:
            data = zarr_group[key][...]
        except KeyError:
            raise ValueError(f"Key {key} not found in Zarr group at {zarr_root}")

        # Extract date from filename like '{...}.zarr/prcp_20200101'
        timestamp = get_date_from_filename(file_path, 'zarr', var_name_short)

    else:
        raise ValueError(f"Unsupported file or path: {file_path}")
    
    data = correct_variable_units(variable, model, data)

    if crop_region is not None:
        data = crop_to_region(data, crop_region)

    if data is None:
        raise ValueError(f"Data is None after processing: {file_path}")

    # logger.info(f"[process_single_file] Loaded {file_path} | shape: {data.shape} | date: {timestamp.date()}")

    return data, timestamp

class DataLoader:
    def __init__(self,
                 base_dir: str,
                 n_workers: int,
                 variable: str,
                 model: str,
                 domain_size: list,
                 split: str,
                 crop_region: list,
                 verbose: bool = False):

        self.variable = variable
        self.var_name_short = get_var_name_short(variable, model)
        
        self.domain_size = domain_size
        self.crop_region = crop_region
        self.split = split
        self.zarr = self.split in ["train", "valid", "test"]

        self.model_type = model
        self.data_dir = build_data_path(base_dir, self.model_type, self.variable, self.domain_size, self.split)#, zarr=self.zarr) 
        
        self.n_workers = n_workers
        self.verbose = verbose

    def _get_file_list(self):
        if self.zarr:
            # Access the internal keys for each day from Zarr
            root = self.data_dir
            zarr_keys = sorted(glob(os.path.join(root, f"{self.var_name_short}_*")))
            return zarr_keys
        else:
            return sorted(glob(os.path.join(self.data_dir, f"{self.var_name_short}_*.npz")))

    def _process_wrapper(self, file_path):
        cutout, timestamp = process_single_file(file_path, self.variable, self.model_type, self.var_name_short, self.crop_region)
        return cutout, timestamp

    def load(self):
        file_list = self._get_file_list()
        logger.info(f"[DataLoader] Found {len(file_list)} files for variable '{self.variable}' (model:{self.model_type}, split: {self.split})")

        if not file_list:
            raise FileNotFoundError(f"No input files found in {self.data_dir} for variable {self.variable} with short name {self.var_name_short}")
        


        results = []
        if self.n_workers > 1:
            # Verbose logging of progress
            if self.verbose:
                # Set up parallel processing
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:    
                    # Create a dictionary to map futures to file names
                    futures = {executor.submit(self._process_wrapper, f): f for f in file_list}
                    
                    # Use as_completed to get results as they finish 
                    for i, future in enumerate(as_completed(futures), 1):
                        try: 
                            results.append(future.result())
                        except Exception as e:
                            logger.error(f"Error processing file {futures[future]}: {e}")
                        if i % 500 == 0 or i == len(file_list):
                            logger.info(f"[DataLoader] Processed {i}/{len(file_list)} files for {self.variable}...")
            # Non-verbose (speed optimized)
            else:
                # Set up parallel processing
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:    
                    results = list(executor.map(self._process_wrapper, file_list))
        
        else:
            for i, f in enumerate(file_list, 1):
                results.append(self._process_wrapper(f))
                if i % 500 == 0 or i == len(file_list):
                    logger.info(f"[DataLoader] Processed {i}/{len(file_list)} files for {self.variable}...")


        # Get the data sorted
        cutouts, timestamps = zip(*results)
        sorted_pairs = sorted(zip(timestamps, cutouts))
        timestamps, cutouts = zip(*sorted_pairs)

        return {
            "cutouts": list(cutouts),
            "timestamps": list(timestamps)
        }
    def load_single_day(self, date_str: str):
        """
            Load data for a single specified date (YYYYMMDD)
        """
        file_list = self._get_file_list()
        match_file = None
        for f in file_list:
            if date_str in f:
                match_file = f
                break
        if not match_file:
            raise FileNotFoundError(f"No file found for date {date_str} in {self.data_dir}")

        data, timestamp = self._process_wrapper(match_file)
        return {
            "cutouts": data,
            "timestamps": timestamp
        }

    def load_multi(self, dates_or_n):
        file_list = self._get_file_list()
        if isinstance(dates_or_n, int):
            selected_files = np.random.choice(file_list, size=dates_or_n, replace=False)
        elif isinstance(dates_or_n, (list, tuple)):
            selected_files = [f for f in file_list if any(d in f for d in dates_or_n)]
        else:
            raise ValueError(f"dates_or_n must be an int or a list/tuple of date strings. Got {dates_or_n}")

        results = [self._process_wrapper(f) for f in selected_files]
        cutouts, timestamps = zip(*results)

        return {
            "cutouts": list(cutouts),
            "timestamps": list(timestamps)
        }





