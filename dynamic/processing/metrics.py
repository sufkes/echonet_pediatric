#!/usr/bin/env python3

import os
import sys
import argparse
import glob

import pandas as pd
import numpy as np
import statsmodels.api as sm

def sortSplits(s):
    if 'val' in s:
        return 1
    elif 'test' in s:
        return 2
    elif 'train' in s:
        return 3
    else:
        return 4

def main(**kwargs):
    out_path = kwargs['out_path']
    in_dirs = kwargs['in_dirs']
    if in_dirs is None:
        in_dirs = glob.glob(os.path.dirname(__file__)+'/../../runs/*')
        in_dirs = [run_dir for run_dir in in_dirs if os.path.isdir(run_dir)]
        in_dirs.sort()

    csv_suffix = '-pred_and_actual.csv'

    # Build dataframe of metrics for each run.
    df = pd.DataFrame()
    for in_dir in in_dirs:
        run_name = os.path.basename(in_dir)
        csv_paths = glob.glob(in_dir + '/*' + csv_suffix)
        csv_paths.sort(key=sortSplits)
        for csv_path in csv_paths:
            split_prefix = os.path.basename(csv_path).split(csv_suffix)[0] # e.g. 'train-EF'
            split_prefix = os.path.basename(csv_path).split(csv_suffix)[0].split('-')[0] # e.g. 'train'

            ## Load predictions CSV to a dataframe.
            pred_df = pd.read_csv(csv_path)
            actual_col = [col for col in pred_df.columns if col.endswith('actual')][0]
            prediction_col = [col for col in pred_df.columns if col.endswith('prediction')][0] # First column ending in 'prediction'

            actual = np.array(pred_df[actual_col])
            prediction = np.array(pred_df[prediction_col])

            mean_actual = actual.mean()
            
            n = len(actual)
            assert n == len(prediction)

            ## Compute performance metrics.
            # MSE
            mse = 1/n*((actual-prediction)**2).sum()

            # RMSE
            rmse = np.sqrt(mse)

            # rRMSE # maybe a bit goofy
            #rrmse = rmse/mean_actual
            
            # MAE
            mae = 1/n*(np.abs(actual-prediction)).sum()

            # rMAE # maybe a bit goofy
            #rmae = mae/mean_actual

            # MAPE
            mape = 100/n*np.abs((actual-prediction)/actual).sum()
            
            # R
            r = np.corrcoef(actual, prediction)[1,0]

            # R squared
            r2 = r**2

            # p-value - doesn't seem to work when p < 1e-12 or so; gives crazy numbers like 1e-80
            y = pred_df[prediction_col]
            x = pred_df[actual_col]
            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            fit = model.fit()
            p = fit.pvalues[1]
            #print(fit.summary())
            #print(fit.pvalues)

#            print('Run:', run_name, split_prefix)
#            print('MSE:', mse)
#            print('RMSE:', rmse)
#            print('MAE:', mae)
#            print('R:', r)
#            print('R^2:', r2)
#            print('p:', p)
#            print()

            df.loc[run_name, split_prefix+'-MSE'] = mse
            df.loc[run_name, split_prefix+'-RMSE'] = rmse
            df.loc[run_name, split_prefix+'-MAE'] = mae
            df.loc[run_name, split_prefix+'-MAPE'] = mape
            #df.loc[run_name, split_prefix+'-rRMSE'] = rrmse
            #df.loc[run_name, split_prefix+'-rMAE'] = rmae
            df.loc[run_name, split_prefix+'-R'] = r
            df.loc[run_name, split_prefix+'-R2'] = r2
            df.loc[run_name, split_prefix+'-p'] = p

    df.to_csv(out_path, index=True)
    print('Saved:', out_path)
    return

if (__name__ == '__main__'):
    # Create argument parser.
    description = """Generate CSV of regression metrics for runs. This task is now performed in the main training and evaluation script."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Define positional arguments.


    # Define optional arguments.
    parser.add_argument('-i', '--in_dirs', help="paths to run directores contain files called 'train_predictions.csv' and 'val_predictions.csv'. Default: all directories in ../../runs", nargs='+')
    parser.add_argument("-o", "--out_path", help="output path", default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/metrics.csv')

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
