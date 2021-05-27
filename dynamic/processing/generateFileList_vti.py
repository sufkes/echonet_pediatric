#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from collections import OrderedDict


def main(patient_data_path, png_data_path, split_path, out_dir=os.getcwd(), subject_prefix='VTI'):
    patient_df = pd.read_csv(patient_data_path)
    png_df = pd.read_csv(png_data_path)
    split_df = pd.read_csv(split_path)
    split_df = split_df[['id_VTI', 'split_VTI']]
    
    # Combine data from the patient data, DICOM data, and AVI data spreadsheets into a single dataframe.
    df = patient_df.merge(png_df, left_on='Subject', right_on='Subject', how='inner')
    df = df.merge(split_df, left_on='Subject', right_on='id_VTI', how='inner')

    # Drop columns.
    df.drop(columns='id_VTI', inplace=True)
    
    # Rename the split column.
    df.rename(columns={'split_VTI':'split_all_random'}, inplace=True)
    
    # Drop rows lacking a split value or pixel_scale_factor
    df = df.loc[(~df['split_all_random'].isna() | ~df['pixel_scale_factor'].isna()), :]

    # Calculate the VTI values in terms of number of pixels in the processed image.
    df['AOVTI_px'] = [vti/r for vti, r in zip(df['AOVTI'], df['pixel_scale_factor'])]
    
    # Sort subjects based on IDs.
    def sort_id_key_vectorized(subject_series):
        def sort_id_key(subject):
            if 'DCM' in subject:
                val = 10000 + int(subject.split('M')[-1])
            else:
                val = int(subject.split(subject_prefix[-1])[-1])
            return val
        
        sort_values = [sort_id_key(subject) for subject in subject_series] # list of (unsorted) numbers whose values will determine ordering of the rows.
        return sort_values
    
    df.sort_values('Subject', axis=0, key=sort_id_key_vectorized, inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Sort columns.
    
    # Save file.
    out_name = 'FileList_vti.csv'
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    
    return

if __name__ == '__main__':
    # Create argument parser.
    description = """Combine data from DICOM header data spreadsheet and the patient information spreadsheet."""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("patient_data_path", help="Path to the spreadsheet containing VTI and other information for each patient, taken from COspreadsheet.")
    parser.add_argument("png_data_path", help="Path to the spreadsheet containing information from the DICOM headers and DICOM filepaths.")
    parser.add_argument('split_path', help='Path to split file containing a column "split_VTI"')
    
    # Define optional arguments.
    parser.add_argument("-o", "--out_dir", type=str, help="Directory to save output to.", default=os.getcwd())

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
