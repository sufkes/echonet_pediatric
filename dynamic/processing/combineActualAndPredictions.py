#!/usr/bin/env python3

import os
import sys
import json
import argparse
import glob

import pandas as pd
#from matplotlib import pyplot as plt


def main(config_path=None, actual_path=None, prediction_paths=None, out_dir=None, tasks='EF'):
    if (config_path is None) and ((actual_path is None) or (prediction_paths is None)):
        raise Exception('Must specify either config_path or both actual_path and prediction_path')
    if not config_path is None:
        if (actual_path or prediction_paths):
            print('Warning: ignoring actual_path and prediction_path since config_path was specified.')
        with open(config_path) as f:
            config = json.load(f)
        actual_path = config['file_list_path']
        run_out_dir = config['output']
        if 'tasks' in config:
            tasks = config['tasks']            
        prediction_paths = glob.glob(run_out_dir+'/*_predictions.csv')
    

    for prediction_path in prediction_paths:

        print('')
        print('Combining:')
        print('Actual:', actual_path)
        print('Predictions:', prediction_path)
        
        df_prediction = pd.read_csv(prediction_path, header=None)
    
        df_prediction.rename(columns={k:v for k,v in enumerate(['Subject', 'clip_num', tasks])}, inplace=True)
        df_prediction['Subject'] = [filename.split('.avi')[0] for filename in df_prediction['Subject']]
    
        df_actual = pd.read_csv(actual_path)
        df_actual.rename(columns={'FileName':'Subject'}, inplace=True)

        actual_col = tasks+'_actual'
        prediction_col = tasks+'_prediction'
        
        df_all = pd.DataFrame(columns=['Subject', actual_col, prediction_col])
        df_all['Subject'] = df_prediction['Subject'].unique()

        df_all[actual_col] = [df_actual.loc[(df_actual['Subject']==subject), tasks].values[0] for subject in df_all['Subject']]
        df_all[prediction_col] = [df_prediction.loc[df_prediction['Subject']==subject, tasks].mean() for subject in df_all['Subject']] # take the mean of the subclip predictions output from the test-time augmentation method.

        ## For LVOT, convert the actual and predicted values back to mm using the resolution for each subject.
        if 'LVOT' in tasks:
            print('Converting LVOT from pixels to cm for run', "'"+config_path+"'")
            resolution_col = 'PhysicalDeltaX'
            df_all = df_all.merge(df_actual[['Subject', resolution_col]], on='Subject', how='left')
            for lvot_col in [actual_col, prediction_col]:
                df_all[lvot_col] = [10*lvot*resolution for lvot, resolution in zip(df_all[lvot_col], df_all[resolution_col])] # Also change from cm to mm.
            df_all.drop(columns=resolution_col, inplace=True)
            
        ## Save plot and spreadsheet.
        if out_dir is None:
            out_dir = os.path.dirname(prediction_path)

        # Save spreadsheet.
        # Guess the split (train, val, or test) based on the predictions CSV file.
        split = os.path.basename(prediction_path).split('_')[0]
        df_all_out_name = split+'-'+tasks+'-pred_and_actual.csv'
        df_all_out_path = os.path.join(out_dir, df_all_out_name)
        df_all.to_csv(df_all_out_path, index=False)
        print('Saved:',df_all_out_path)
        
        # Plot
        #out_name = 'EF_pred_vs_actual.png'
        #out_path = os.path.join(out_dir, out_name)
        #plt.figure()
        #plt.plot(df_all['EF_actual'], df_all['EF_prediction'], 'o', color='black')
        #plt.xlim(0, 100)
        #plt.ylim(0, 100)
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.savefig(out_path)
        #print('Saved:', out_path)
    return

if (__name__ == '__main__'):
    # Create argument parser
    description = """Description of function"""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("config_path", help="path to JSON configuration for run; used to find the predicted and actual value CSV files. Can optionally use the -a and -p flags instead.", nargs='?') # Defaults to None if not specified, as expected.
    
    # Define optional arguments.
    parser.add_argument("-a", "--actual_path", help="path to actual values.")
    parser.add_argument("-p", "--prediction_paths", help="paths to predictions files output by EchoNet.", nargs='+') # can list 1 or more predictions files.
    parser.add_argument("-o", "--out_dir", help="output directory; default: save to same directory as the predictions file.")
    
    # Parse arguments.
    args = parser.parse_args()

    # Plot predictions against
    main(**vars(args))
