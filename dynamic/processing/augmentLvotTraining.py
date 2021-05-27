#!/usr/bin/env python

import os
import sys
import pandas as pd
import argparse
import numpy as np

def main(in_path, out_path=None, random=False, sd=2.25, num_reps=6, seed=489):
    np.random.seed(seed)
    #in_path = 'FileList_lvot-one_split.csv'
    if out_path is None:
        suffix = '-augmented'
        if random:
            suffix += '-random'
        suffix += '.csv'
        out_path = in_path.replace('.csv', suffix)

    df = pd.read_csv(in_path, index_col='Subject')

    #C1_LVOTno1_px C1_LVOTno2_px C1_LVOTno3_px C2_LVOTno1_px C2_LVOTno2_px C2_LVOTno3_px LVOT_mean_px
    if random:
        new_col = 'LVOT_px_random'
    else:
        new_col = 'LVOT_px_trainoneframe' # Column to be predicted

    # For test and validation samples, just use LVOT_mean_px as before.
    new_df = df.loc[df['split_all_random'].isin(['val', 'test']), :]
    new_df.loc[df['split_all_random'].isin(['val', 'test']), new_col] = df.loc[df['split_all_random'].isin(['val', 'test']), 'LVOT_mean_px']

    ## For each training sample, create a separate row for each frame which has a corresponding LVOT value. This is up to six rows for each training sample.
    # Loop over framewise LVOT values.
    if random:
        for ii in range(1, num_reps+1):
            for index in df.loc[(df['split_all_random']=='train'), :].index:
                # Create a new row for each.
                new_index = str(index) + '_' + str(ii) # Append a number to the subject name, representing the specific LVOT measurement.
                new_df.loc[new_index, :] = df.loc[index, :]

                # Fill in the new prediction column.
                new_df.loc[new_index, new_col] = df.loc[index, 'LVOT_mean_px'] + np.random.normal(loc=0.0, scale=sd, size=1)
    else:
        framewise_cols = ['C1_LVOTno1_px', 'C1_LVOTno2_px', 'C1_LVOTno3_px', 'C2_LVOTno1_px', 'C2_LVOTno2_px', 'C2_LVOTno3_px']
        for lvot_num, col in enumerate(framewise_cols, 1):
            for index in df.loc[((df['split_all_random']=='train') & (~df[col].isna())), :].index:
                # Create a new row for each
                new_index = str(index) + '_' + str(lvot_num) # Append a number to the subject name, representing the specific LVOT measurement.
                new_df.loc[new_index, :] = df.loc[index, :]

                # Fill in the new prediction column.
                new_df.loc[new_index, new_col] = df.loc[index, col]

    # Sort by subject ID.
    def sort_id_key_vectorized(subject_series):
        def sort_id_key(subject):
            if 'DCM' in subject:
                val = 100000 + 10*int(subject.split('M')[-1].split('_')[0])
            else:
                val = 10*int(subject.split('T')[-1].split('_')[0])
            if '_' in subject:
                val += int(subject.split('_')[-1])
            return val

        sort_values = [sort_id_key(subject) for subject in subject_series] # list of (unsorted) numbers whose values will determine ordering of the rows.
        return sort_values

    new_df.sort_values('Subject', axis=0, key=sort_id_key_vectorized, inplace=True)

    # Save dataframe
    new_df.to_csv(out_path, index=True)

    return

if (__name__ == '__main__'):
    # Create argument parser.
    description = """Create new rows for each repeat LVOT measurement of the training set."""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("in_path", help="path to FileList.csv file.")
    
    # Define optional arguments.
    parser.add_argument("-o", "--out_path", help="output file path. Default: add '-augmented' to input file name.")
    parser.add_argument("-r", "--random", action='store_true', help="augmented values are just the mean plus some random noise, instead of actual repeat measurements")
    
    # Print help if no args input.
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit()

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))



