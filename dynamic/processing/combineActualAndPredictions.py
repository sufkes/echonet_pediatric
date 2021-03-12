#!/usr/bin/env python

import os
import sys
import pandas as pd
from matplotlib import pyplot as plt

import argparse

def main(actual_path, prediction_path, out_dir=None):
    df_prediction = pd.read_csv(prediction_path, header=None)
    
    df_prediction.rename(columns={k:v for k,v in enumerate(['Subject', 'clip_num', 'EF'])}, inplace=True)
    df_prediction['Subject'] = [filename.split('.avi')[0] for filename in df_prediction['Subject']]
#    print(df_prediction)
    
    df_actual = pd.read_csv(actual_path)
    df_actual.rename(columns={'FileName':'Subject'}, inplace=True)
#    print(df_actual)
    
    df_all = pd.DataFrame(columns=['Subject', 'EF_actual', 'EF_prediction'])
    df_all['Subject'] = df_prediction['Subject'].unique()

    df_all['EF_actual'] = [df_actual.loc[(df_actual['Subject']==subject), 'EF'].values[0] for subject in df_all['Subject']]
    df_all['EF_prediction'] = [df_prediction.loc[df_prediction['Subject']==subject, 'EF'].mean() for subject in df_all['Subject']]

    ## Save plot and spreadsheet.
    if out_dir is None:
        out_dir = os.path.dirname(prediction_path)

    # Save spreadsheet.
    # Guess the split (train, val, or test) based on the predictions CSV file.
    split = os.path.basename(prediction_path).split('_')[0]
    df_all_out_name = split+'_EF_pred_and_actual.csv'
    df_all_out_path = os.path.join(out_dir, df_all_out_name)
    df_all.to_csv(df_all_out_path, index=False)
    print('Saved:',df_all_out_path)
        
    # Plot
#    out_name = 'EF_pred_vs_actual.png'
#    out_path = os.path.join(out_dir, out_name)
#    plt.figure()
#    plt.plot(df_all['EF_actual'], df_all['EF_prediction'], 'o', color='black')
#    plt.xlim(0, 100)
#    plt.ylim(0, 100)
#    plt.gca().set_aspect('equal', adjustable='box')
#    plt.savefig(out_path)
#    print('Saved:', out_path)
    return

if (__name__ == '__main__'):
    # Create argument parser
    description = """Description of function"""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("actual_path", help="path to actual EF values.")
    parser.add_argument("prediction_path", help="path to EF predictions file output by EchoNet.")
    
    # Define optional arguments.
    parser.add_argument("-o", "--out_dir", help="output directory; default: save to same directory as the predictions file.")

    # Print help if no args input.
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit()

    # Parse arguments.
    args = parser.parse_args()

    # Plot predictions against
    main(**vars(args))
