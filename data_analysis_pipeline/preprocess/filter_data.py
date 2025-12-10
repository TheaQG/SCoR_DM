import os
import numpy as np
import logging
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def filter_files(data_dir, expected_shape=None, required_keys=None, delete_invalid=False):
    """
        Scan through .npz files in the given directory, check for corruption based on expected shape and required keys.
        Corruption based on shape mismatch or missing keys.

        Parameters:
            data_dir (str): Directory containing .npz files to check.
            expected_shape (tuple, optional): Expected shape of the data arrays. If None, shape is not checked.
            required_keys (list, optional): List of keys that must be present in the .npz file. If None, keys are not checked.
            delete_invalid (bool): If True, delete files that are found to be corrupted.

        Returns:
            valid_files (list): List of valid file paths.
            corrupted_files (list): List of corrupted file paths.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    valid_files = []
    corrupted_files = []

    for fname in tqdm(files, desc="Filtering files"):
        fpath = os.path.join(data_dir, fname)
        try:
            with np.load(fpath) as data:
                # Check for required keys
                if required_keys:
                    if not all(key in data for key in required_keys):
                        raise ValueError(f"Missing required keys in {fname}")

                # Check for expected shape
                if expected_shape:
                    for v in data.values():
                        if v.shape != expected_shape:
                            raise ValueError(f"Shape mismatch in {fname}: expected {expected_shape}, got {v.shape}")

        except Exception as e:
            logger.warning(f"Corrupted file detected: {fname} | Reason: {e}")
            corrupted_files.append(fpath)
            if delete_invalid:
                os.remove(fpath)
                logger.info(f"Deleted corrupted file: {fname}")
            continue

        valid_files.append(fpath)
    logger.info(f"Total files scanned: {len(files)}")
    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Corrupted files: {len(corrupted_files)}")
    return valid_files, corrupted_files



import os
import numpy as np
from shutil import copyfile
from collections import Counter
from datetime import datetime

def is_corrupted(filepath, threshold=1.0):
    """" Returns True if the min of the data in the file is below the threshold, indicating corruption. """
    try:
        data = np.load(filepath)['arr_0']
        print(f"Checking {filepath}: min = {np.min(data)}")
        return np.min(data) < threshold
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return True  # treat unreadable files as corrupted

def scan_temp_files(temp_dir, threshold=1.0):
    corrupted_dates = set()
    valid_dates = set()

    for fname in sorted(os.listdir(temp_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(temp_dir, fname)
        date_str = fname.split('_')[-1].split('.')[0]
        if is_corrupted(path, threshold):
            print(f"Corrupted file detected: {fname}")
            corrupted_dates.add(date_str)
        else:
            valid_dates.add(date_str)
    print(f"Number of files scanned: {len(corrupted_dates) + len(valid_dates)}")
    print(f"Number of corrupted files: {len(corrupted_dates)}")
    return corrupted_dates, valid_dates

def print_month_distribution(dates, label):
    months = [datetime.strptime(d, "%Y%m%d").month for d in dates]
    dist = Counter(months)
    print(f"\nMonth distribution for {label}:")
    for month, count in sorted(dist.items()):
        print(f"  Month {month:02d}: {count} files")

def clear_directory(path):
    if os.path.exists(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

def filter_files_by_date(dataset_name, dataset_path, var_list, valid_dates):
    print(f"\nFiltering files for dataset '{dataset_name}'...")

    for var in var_list:
        var_path = os.path.join(dataset_path, f"{var}_589x789")
        input_dir = os.path.join(var_path, "all")
        output_dir = os.path.join(var_path, "all_filtered")
        os.makedirs(output_dir, exist_ok=True)

        # Clear the output directory
        if len(os.listdir(output_dir)) > 0:
            print(f"Clearing existing files in {output_dir}...")
            clear_directory(output_dir)

        print(f"Copying files from {input_dir} to {output_dir}...")

        print(f"\nCopying valid files for variable '{var}'...")

        for fname in os.listdir(input_dir):
            if not fname.endswith(".npz"):
                continue
            date_str = fname.split('_')[-1].split('.')[0]
            if date_str in valid_dates:
                src = os.path.join(input_dir, fname)
                dst = os.path.join(output_dir, fname)
                copyfile(src, dst)

        print(f"Filtered files copied to: {output_dir}")

def main():
    # User config
    base_path = "/scratch/project_xxxxxxxxx/user/Data/Data_DiffMod"
    era5_path = os.path.join(base_path, "data_ERA5/size_589x789")
    danra_path = os.path.join(base_path, "data_DANRA/size_589x789")

    temp_var = "temp"
    var_list = ["temp", "prcp"]
    threshold = 1.0  # mean threshold for detecting corruption

    temp_dir_era5 = os.path.join(era5_path, f"{temp_var}_589x789/all")
    # Step 1: Find corrupted dates
    print(f"Scanning files in {temp_dir_era5} for corrupted data...")
    corrupted_dates, valid_dates = scan_temp_files(temp_dir_era5, threshold)
    print(f"\nDetected {len(corrupted_dates)} corrupted files.")
    print(f"Remaining {len(valid_dates)} valid files.")

    # Step 2: Month-wise distribution
    print_month_distribution(corrupted_dates, "corrupted")
    print_month_distribution(valid_dates, "valid")

    # Step 3: Copy valid files
    filter_files_by_date("ERA5", era5_path, var_list, valid_dates)
    filter_files_by_date("DANRA", danra_path, var_list, valid_dates)

if __name__ == "__main__":
    main()