import os
import shutil
import argparse

import numpy as np

from utils import convert_npz_to_zarr


def create_train_valid_test_data_zarr(args):
    '''
        Function to make zarr data from all data in the data directory.
        Saves the zarr data to the zarr_files directory.
    '''
    data_dir = args.path_data
    hr_var = args.hr_var
    lr_var = args.lr_var
    data_split_type = args.data_split_type
    if data_split_type == 'Time':
        train_years = args.train_years
        valid_years = args.valid_years
        test_years = args.test_years
        data_split_params = {'train_years': train_years,
                             'valid_years': valid_years,
                             'test_years': test_years
                             }
        print(f'\nSplitting data in time with the following years:')
        print(f'Train years: {train_years}')
        print(f'Valid years: {valid_years}')
        print(f'Test years: {test_years}\n')

    elif data_split_type == 'Random':
        train_frac = args.train_frac
        valid_frac = args.valid_frac
        test_frac = args.test_frac
        data_split_params = {'train_frac': train_frac,
                             'valid_frac': valid_frac,
                             'test_frac': test_frac
                             }
        print(f'\nSplitting data randomly with the following fractions:')
        print(f'Train fraction: {train_frac}')
        print(f'Valid fraction: {valid_frac}')
        print(f'Test fraction: {test_frac}\n')
    else:
        raise ValueError('Data split type not recognized')

    # Set the paths to all data
    LR_PATH = data_dir + 'data_ERA5/size_589x789/' + lr_var + '_589x789/'
    HR_PATH = data_dir + 'data_DANRA/size_589x789/' + hr_var + '_589x789/'

    LR_PATH_ALL = LR_PATH + 'all_filtered/'
    HR_PATH_ALL = HR_PATH + 'all_filtered/'

    # Check if /train, /valid, /test directories exist. If not, create them
    if not os.path.exists(LR_PATH + 'train'):
        os.makedirs(LR_PATH + 'train')
    if not os.path.exists(LR_PATH + 'valid'):
        os.makedirs(LR_PATH + 'valid')
    if not os.path.exists(LR_PATH + 'test'):
        os.makedirs(LR_PATH + 'test')

    if not os.path.exists(HR_PATH + 'train'):
        os.makedirs(HR_PATH + 'train')
    if not os.path.exists(HR_PATH + 'valid'):
        os.makedirs(HR_PATH + 'valid')
    if not os.path.exists(HR_PATH + 'test'):
        os.makedirs(HR_PATH + 'test')

    # Empty the /train, /valid, /test directories
    print(f'\nEmptying LR and HR directories')
    for file in os.listdir(LR_PATH + 'train'):
        os.remove(LR_PATH + 'train/' + file)
    for file in os.listdir(LR_PATH + 'valid'):
        os.remove(LR_PATH + 'valid/' + file)
    for file in os.listdir(LR_PATH + 'test'):
        os.remove(LR_PATH + 'test/' + file)

    for file in os.listdir(HR_PATH + 'train'):
        os.remove(HR_PATH + 'train/' + file)
    for file in os.listdir(HR_PATH + 'valid'):
        os.remove(HR_PATH + 'valid/' + file)
    for file in os.listdir(HR_PATH + 'test'):
        os.remove(HR_PATH + 'test/' + file)


    print(f'\nMoving LR files from {LR_PATH_ALL}')
    print(f'Moving HR files from {HR_PATH_ALL}\n')

    if data_split_type == 'Time':
        print('\nSplitting data in time with the following years:')
        # Define the splits
        train_years = data_split_params['train_years']
        valid_years = data_split_params['valid_years']
        test_years = data_split_params['test_years']
        print(f'Train years: {train_years}')
        print(f'Valid years: {valid_years}')
        print(f'Test years: {test_years}\n')

        # Copy all .npz data files to the correct directory. The year is the [-12:-8] part of the filename (including the .npz extension)
        print('Copying LR files to /train, /valid, /test directories')
        n_train = 0
        n_valid = 0
        n_test = 0
        for file in sorted(os.listdir(LR_PATH_ALL)):
            if file[-12:-8] in train_years:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'train/' + file)
                n_train += 1
            elif file[-12:-8] in valid_years:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'valid/' + file)
                n_valid += 1
            elif file[-12:-8] in test_years:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'test/' + file)
                n_test += 1
        print(f'\nCopied {n_train} files to /train, {n_valid} files to /valid and {n_test} files to /test\n')
        # Now make .zarr files from files in the /train, /valid, /test directories (.npz for both LR and HR)
        convert_npz_to_zarr(LR_PATH + 'train/', LR_PATH + 'zarr_files/train.zarr')
        convert_npz_to_zarr(LR_PATH + 'valid/', LR_PATH + 'zarr_files/valid.zarr')
        convert_npz_to_zarr(LR_PATH + 'test/', LR_PATH + 'zarr_files/test.zarr')

        print('Copying HR files to /train, /valid, /test directories')
        n_train = 0
        n_valid = 0
        n_test = 0
        for file in sorted(os.listdir(HR_PATH_ALL)):
            if file[-12:-8] in train_years:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'train/' + file)
                n_train += 1
            elif file[-12:-8] in valid_years:
                #W Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'valid/' + file)
                n_valid += 1
            elif file[-12:-8] in test_years:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'test/' + file)
                n_test += 1
        print(f'\nCopied {n_train} files to /train, {n_valid} files to /valid and {n_test} files to /test\n')
        # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
        convert_npz_to_zarr(HR_PATH + 'valid/', HR_PATH + 'zarr_files/valid.zarr')
        convert_npz_to_zarr(HR_PATH + 'train/', HR_PATH + 'zarr_files/train.zarr')
        convert_npz_to_zarr(HR_PATH + 'test/', HR_PATH + 'zarr_files/test.zarr')

    elif data_split_type == 'Random':
        # PROBLEM WITH RANDOM: NOT THE SAME DATA IN ERA5 AND DANRA - different years. 
        # (should not be a problem on LUMI, but locally it is (for now))
        
        # Define the random splits
        train_frac = data_split_params['train_frac']
        valid_frac = data_split_params['valid_frac']
        test_frac = data_split_params['test_frac']

        # Figure out, how many files to put in each directory
        n_files = len(os.listdir(LR_PATH_ALL))
        n_train = int(n_files * train_frac)
        n_valid = int(n_files * valid_frac)
        n_test = n_files - n_train - n_valid

        # Make a random permutation of the indices 
        indices = np.random.permutation(n_files)
        print(indices)

        # Get the random indices for the train, valid and test sets
        train_indices = indices[:n_train]
        valid_indices = indices[n_train:n_train+n_valid]
        test_indices = indices[n_train+n_valid:]

        # Make 0/1 arrays for the train, valid and test sets
        train_mask = np.zeros(n_files)
        valid_mask = np.zeros(n_files)
        test_mask = np.zeros(n_files)
        train_mask[train_indices] = 1
        valid_mask[valid_indices] = 1
        test_mask[test_indices] = 1

        # Make lists of the files in the LR and HR directories to be able to sort them (omit the .DS_Store file)
        LR_files = sorted(os.listdir(LR_PATH_ALL))
        HR_files = sorted(os.listdir(HR_PATH_ALL))
        if '.DS_Store' in LR_files:
            LR_files.remove('.DS_Store')
        if '.DS_Store' in HR_files:
            HR_files.remove('.DS_Store')

        # Copy all .npz data files to the correct directory
        print('Copying LR files to /train, /valid, /test directories')
        for i, file in enumerate(LR_files):
            if train_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'train/' + file)
            elif valid_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'valid/' + file)
            elif test_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(LR_PATH_ALL + file, LR_PATH + 'test/' + file)
        # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
        convert_npz_to_zarr(LR_PATH + 'train/', LR_PATH + 'zarr_files/train.zarr')

        print('Copying HR files to /train, /valid, /test directories')
        for i, file in enumerate(HR_files):
            if train_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'train/' + file)
            elif valid_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'valid/' + file)
            elif test_mask[i] == 1:
                # Copy the files to the correct directories (NOT RENAMING THEM)
                shutil.copy(HR_PATH_ALL + file, HR_PATH + 'test/' + file)
        # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
        convert_npz_to_zarr(LR_PATH + 'valid/', LR_PATH + 'zarr_files/valid.zarr')

    else:
        raise ValueError('Data split type not recognized')
    
def launch_split_from_args():
    parser = argparse.ArgumentParser(description='Split data in time or randomly and make zarr files')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The data directory')
    parser.add_argument('--hr_var', type=str, default='temp', help='The high resolution variable')
    parser.add_argument('--lr_var', type=str, default='temp', help='The low resolution variable')
    parser.add_argument('--data_split_type', type=str, default='Time', help='The data split type')
    parser.add_argument('--train_years', type=list, default=np.arange(1990, 2015).astype(str), help='The training years')
    parser.add_argument('--valid_years', type=list, default=np.arange(2015, 2018).astype(str), help='The validation years')
    parser.add_argument('--test_years', type=list, default=np.arange(2018, 2021).astype(str), help='The test years')
    parser.add_argument('--train_frac', type=float, default=0.7, help='The training fraction')
    parser.add_argument('--valid_frac', type=float, default=0.1, help='The validation fraction')
    parser.add_argument('--test_frac', type=float, default=0.2, help='The test fraction')

    args = parser.parse_args()

    create_train_valid_test_data_zarr(args)

    
    


if __name__ == '__main__':
    launch_split_from_args()
