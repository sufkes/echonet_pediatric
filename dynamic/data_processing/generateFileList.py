#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from collections import OrderedDict

def splitDataset(df, split_type_name, custom_id_list=None, id_col='Subject', ratio_train=0.75, ratio_val=0.125, ratio_test=0.125, bin_on_col=None, date=False, seed=489):
    """
    df (pandas.DataFrame): Dataframe containing IDs to be split into train, validation, and test sets.
    split_type_name (str): Name of column that stores the split set assignements. Column will be added to df.
    custom_id_list (list): List of IDs to assign to split sets. If None, all rows in the dataframe will be assigned to split sets.
    id_col (str): Name of the column storing subject IDs.
    ratio_train (float): Proportion of subjects to put in the training set.
    ratio_val (float): Proportion of subjects to put in the validation set.
    ratio_test (float): Proportion of subjects to put in the test set.
    bin_on_col (str): Name of the column whose values will be "distributed evenly" across the split sets. If None, split sets are assigned at random.
    date (bool): Whether bin_on_col stores dates. Ignored if bin_on_col is None.
    seed (int): Seed used in random number generation.
    """

    # Generate list of subject IDs.
    if custom_id_list:
        id_list = custom_id_list
    else:
        id_list = df[id_col]
    
    # Set the seed for randomization in NumPy.
    np.random.seed(seed)

    # Calculate number of subjects to be included in each group.
    ratio_train = 0.75
    ratio_val = 0.125
    ratio_test = 0.125
    
    num_subjects = len(id_list)
    num_val = int(np.round(ratio_val*num_subjects))
    num_test = int(np.round(ratio_test*num_subjects))
    num_train = int(num_subjects - num_val - num_test)
    
    split_nums = OrderedDict([('train',num_train),
                              ('val',num_val),
                              ('test',num_test)
                              ])
    split_ids = OrderedDict([('train',[]),
                             ('val',[]),
                             ('test',[])
                             ])

    # Create splits in order of increasing size.
    splits_ordered = sorted(list(split_nums.keys()), key=lambda x: split_nums[x]) # will usually be ['test', 'val', 'train'] or ['val', 'test', 'train'].
    
    if bin_on_col is not None:
        # Convert bin_on_values to float so that it can always be sorted in the same way.
        bin_on_values = df.loc[df[id_col].isin(id_list), bin_on_col]
        if date:
            bin_on_values = pd.Series(bin_on_values).dt.strftime("%Y%m%d").astype(np.float64) # convert, e.g. 2020-12-25 to 20201225 for the purpose of sorting.
        else:
            bin_on_values = [np.float64(val) for val in bin_on_values]
                    
        # Create temporary dataframe to use for sorting.
        id_value_df = pd.DataFrame({id_col:id_list, 'value':bin_on_values, 'split':''})

        # Add a tiny random value to each entry so that the are all unique and can be put into equally sized bins using pandas.qcut.
        while id_value_df['value'].duplicated().any():
            # Perturb each duplicated value by a tiny amount.
            # Find a value to perturb the data by such that the ordering of unique values is unaffected. Make it smaller than the smallest nonzero difference between any pair of values.
            values_sorted = sorted(id_value_df['value'].unique().tolist())
            differences = [(val2 - val1) for val1, val2 in zip(values_sorted, values_sorted[1:])]
            smallest_difference = min(differences)
#            print('smallest nonzero absolute difference between values:', smallest_difference)
            
            perturbation_size = smallest_difference * 1e-3

            for index in id_value_df[id_value_df['value'].duplicated()].index:
                perturbation = np.random.uniform(low=0.0, high=perturbation_size)
                
                id_value_df.loc[index, 'value'] += perturbation

        for split_name in splits_ordered:
            # If this is the last split to be done, just take all remaining IDs and quit this loop.
            if split_name == splits_ordered[-1]:
                split_ids[split_name] = list(id_value_df.loc[(id_value_df['split']==''), id_col])
                break

            num_split = split_nums[split_name] # Number of subjects in the current set.

            ## Split the list of IDs into num_split bins
            # Calculate the boundaries of all the bins.
            _, bin_edges = pd.qcut(id_value_df['value'], num_split, retbins=True) # split into as many bins as there are subjects in the current set.

            for left, right in zip(bin_edges, bin_edges[1:]):
                # Assign one subject in the current bin to the current set.
                #bin_df = id_value_df[(id_value_df['value']>left) & (id_value_df['value']<=right)]
                picked_id = np.random.choice(id_value_df.loc[(id_value_df['value']>left) & (id_value_df['value']<=right) & (id_value_df['split']==''), id_col])
                id_value_df.loc[id_value_df[id_col]==picked_id, 'split'] = split_name # Mark the split set in the value dataframe so that it doesn't get assigned twice.
                
                split_ids[split_name].append(picked_id) # Add the ID to the list of IDs for the current set.

    else:
        id_value_df = pd.DataFrame({id_col:id_list, 'split':''})

        for split_name in splits_ordered:
            # If this is the last split to be done, just take all remaining IDs and quit this loop.
            if split_name == splits_ordered[-1]:
                split_ids[split_name] = list(id_value_df.loc[(id_value_df['split']==''), id_col])
                break

            num_split = split_nums[split_name] # Number of subjects in the current set.        
            picked_ids = np.random.choice(id_value_df.loc[id_value_df['split']=='', id_col], size=num_split, replace=False)
            
            id_value_df.loc[id_value_df[id_col].isin(picked_ids), 'split'] = split_name

            split_ids[split_name] = picked_ids

    # Fill in the split column in the real dataframe.
    for split_name, split_ids in split_ids.items():
        df.loc[df[id_col].isin(split_ids), split_type_name] = split_name
            
    return df


def plotSplitHistogram(df, split_col, out_dir, split_val=None, split_list=['train', 'val', 'test']):
    if split_val == None:
        split_val = split_col.split('_')[-1] # should be random, EF, or StudyDate

    if split_val == 'random':
        split_val = 'EF' # if this is a random split, plot against EF just for curiosity's sake.
        
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes
    x_multi = [df.loc[(df[split_col]==split_name), split_val] for split_name in split_list]

    ax0.hist(x_multi, 8, histtype='bar', label=split_list, density=False)
    ax0.legend()
    ax0.set_ylabel('Number of subjects')
    ax0.set_xlabel(split_val)
    
    ax1.hist(x_multi, 8, histtype='bar', label=split_list, density=True)
    ax1.legend()
    ax1.set_ylabel('Proportion of subset in bin')
    ax1.set_xlabel(split_val)

    fig.suptitle(split_col)
    
    fig.tight_layout()
    out_name = split_col+'.png'
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path)
    plt.close()
    return

def main(patient_data_path, dicom_data_path, avi_data_path, out_dir=os.getcwd(), make_plots=False):
    if (patient_data_path.split('.')[-1] == 'xlsx'):
        patient_df = pd.read_excel(patient_data_path)
    else:
        patient_df = pd.read_csv(patient_data_path)
    dicom_df = pd.read_csv(dicom_data_path)
    avi_df = pd.read_csv(avi_data_path)
    
    # Combine data from the patient data, DICOM data, and AVI data spreadsheets into a single dataframe.
    df = dicom_df.merge(patient_df, left_on='Subject', right_on='SUBJECT', how='inner')
#    print('Note discrepancies in scan dates between datasheet and DICOM headers:')
#    print(df[(df['StudyDate'] != df['DOS']) & (~df['DOS'].isna())])

#    print(len(df))
    df = df.merge(avi_df, on='Subject', how='inner')
#    print([s for s in df['Subject'].tolist() if not s in avi_df['Subject'].tolist()])
#    print([s for s in avi_df['Subject'].tolist() if not s in df['Subject'].tolist()])
#    print(len(df))

    # Drop columns
    drop_cols = ['SUBJECT', 'DOS']
    df.drop(columns=drop_cols, inplace=True)
    
    # Rename columns
    rename_cols = {'SEX':'sex',
                   'AGE':'age',
                   'LVEDVA4':'EDV',
                   'LVESVA4':'ESV',
                   'LVEF 4C':'LVEF_4C',
                   'LVEF  4Cand2C':'LVEF_4Cand2C',
                   }
    df.rename(columns=rename_cols, inplace=True)

    # Clean up NaN values
    float_cols = ['age', 'EDV', 'ESV', 'LVEF_4C', 'LVEF_4Cand2C']
    for col in float_cols:
        df.loc[df[col].isin(['.', '#VALUE!']), col] = np.nan
        df[col] = df[col].astype(np.float64)

    # Manually calculate LVEF 4C to be used as the ground truth.
    df['EF'] = [(edv - esv)/edv*100 for edv, esv in zip(df['EDV'],df['ESV'])]
    df['LVEF_discrepancy'] = [(ef_calc - ef_data) for ef_calc, ef_data in zip(df['EF'], df['LVEF_4C'])]
    
    # Drop rows lacking an EF value.
    df = df.loc[~df['EF'].isna(), :]
    
    # Add frame dimensions which are always the same (enforced by DCM -> AVI preprocessing script).
    df['FrameHeight'] = 112
    df['FrameWidth'] = 112
    
    # Sort subjects based on IDs.
    def sort_id_key_vectorized(subject_series):
        def sort_id_key(subject):
            if 'DCM' in subject:
                val = 10000 + int(subject.split('M')[-1])
            else:
                val = int(subject.split('C')[-1])
            return val
        
        sort_values = [sort_id_key(subject) for subject in subject_series] # list of (unsorted) numbers whose values will determine ordering of the rows.
        return sort_values
    
    df.sort_values('Subject', axis=0, key=sort_id_key_vectorized, inplace=True)

    # Convert dates to date types.
    df['StudyDate'] = pd.to_datetime(df['StudyDate'])
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Sort columns.
    first_cols = ['Subject', 'FilePath', 'EF', 'ESV', 'EDV', 'FrameHeight', 'FrameWidth', 'FPS', 'NumberOfFrames']
    sorted_cols = first_cols + [col for col in df.columns if not col in first_cols]
    df = df[sorted_cols]

    ### Assign training, validation, and test splits.
    ids_nondcm = [ii for ii in df['Subject'] if not 'DCM' in ii]
    ids_dcm = [ii for ii in df['Subject'] if 'DCM' in ii]

    # random split; all subjects
#    df = splitDataset(df, 'split_all_studydate', df['Subject'])
    df = splitDataset(df, 'split_all_random')
    # random split; non-DCM subjects
    df = splitDataset(df, 'split_nondcm_random', custom_id_list=ids_nondcm)
    # random split; DCM subjects
    df = splitDataset(df, 'split_dcm_random', custom_id_list=ids_dcm)

    # StudyDate split; all subjects
    df = splitDataset(df, 'split_all_StudyDate', bin_on_col='StudyDate', date=True)
    # time split; non-DCM subjects
    df = splitDataset(df, 'split_nondcm_StudyDate', custom_id_list=ids_nondcm, bin_on_col='StudyDate', date=True)
    # time split; DCM subjects
    df = splitDataset(df, 'split_dcm_StudyDate', custom_id_list=ids_dcm, bin_on_col='StudyDate', date=True)

    # EF split; all subjects
    df = splitDataset(df, 'split_all_EF', bin_on_col='EF')
    # value split; non-DCM subjects
    df = splitDataset(df, 'split_nondcm_EF', custom_id_list=ids_nondcm, bin_on_col='EF')
    # value split; DCM subjects
    df = splitDataset(df, 'split_dcm_EF', custom_id_list=ids_dcm, bin_on_col='EF')

    out_name = 'FileList.csv'
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)

    # Generate histograms to check the distributions.
    if make_plots:
        split_cols = [col for col in df.columns if 'split' in col]
        split_list = ['train', 'val', 'test']
        for split_col in split_cols:
            if 'StudyDate' in split_col:
                plotSplitHistogram(df, split_col, out_dir)
            elif 'EF' in split_col:
                plotSplitHistogram(df, split_col, out_dir)
            else:
                plotSplitHistogram(df, split_col, out_dir)
    return

if __name__ == '__main__':
    # Create argument parser.
    description = """Combine data from DICOM header data spreadsheet and the patient information spreadsheet."""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("patient_data_path", help="Path to the spreadsheet containing EF and other information for each patient, taken from COspreadsheet.")
    parser.add_argument("dicom_data_path", help="Path to the spreadsheet containing information from the DICOM headers and DICOM filepaths.")
    parser.add_argument("avi_data_path", help="Path to the spreadsheet containing subject names and file paths for the processed AVI files.")
    
    # Define optional arguments.
    parser.add_argument("-o", "--out_dir", type=str, help="Directory to save output to.", default=os.getcwd())
    parser.add_argument("-p", "--make_plots", action='store_true', help="Plot histograms of train/validation/test splits.")

    # Print help if no args input.
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit()

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
