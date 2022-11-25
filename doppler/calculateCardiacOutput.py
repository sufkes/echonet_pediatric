#!/usr/bin/env python

import os
import sys
import argparse

import pandas as pd
import numpy as np

def main(vti_path, lvot_path, heart_rate_path, out_dir, prefix, vti_units, lvot_units, out_units):
    df_vti = pd.read_csv(vti_path)
    df_lvot = pd.read_csv(lvot_path)
    df_hr = pd.read_csv(heart_rate_path)

    ## Convert the subject IDs to their base names.
    df_vti['Subject'] = [s.replace('VTI','') for s in df_vti['Subject_base']]
    df_vti.drop(columns='Subject_base', inplace=True)
    df_lvot['Subject'] = [s.replace('LVOT','') for s in df_lvot['Subject']]

    ## Rename columns to nice things.
    vti_actual_col = [col for col in df_vti.columns if 'actual' in col][0]
    vti_pred_col = [col for col in df_vti.columns if 'pred' in col][0]
    df_vti.rename(columns={vti_actual_col:'vti_actual', vti_pred_col:'vti_prediction'}, inplace=True)

    lvot_actual_col = [col for col in df_lvot.columns if 'actual' in col][0]
    lvot_pred_col = [col for col in df_lvot.columns if 'pred' in col][0]
    df_lvot.rename(columns={lvot_actual_col:'lvot_actual', lvot_pred_col:'lvot_prediction'}, inplace=True)

    ## Merge dataframes
    print('Subjects with VTI but no LVOT diameter:')
    print([s for s in df_vti['Subject'] if not s in df_lvot['Subject'].tolist()])
    print('Subjects with LVOT diameter but no VTI:')
    print([s for s in df_lvot['Subject'] if not s in df_vti['Subject'].tolist()])
    df = df_vti.merge(df_lvot, on='Subject', how='outer')
    df = df.merge(df_hr, on='Subject', how='left')


    ## Calculate cardiac output (CO)
    # Calculate the factor to mutliply CO by to get the requested units. THIS IS PLACEHOLDER CODE WHICH CAN ONLY HANDLE ONE CASE.
    unit_factor = 1.0
    if out_units == 'L/min':
        if vti_units == 'cm':
            unit_factor *= 0.1 # A litre is 100 mm^3
        else:
            raise Exception('Unhandled VTI units')
        if lvot_units == 'mm':
            unit_factor *= 0.01**2
        else:
            raise Exception('Unhandled LVOT units')
    else:
        raise Exception('Unhandled cardiac output units')
    
    df['co_actual'] = [np.pi/4 * vti * lvot**2 * heart_rate * unit_factor for (vti, lvot, heart_rate) in zip(df['vti_actual'], df['lvot_actual'], df['heart_rate'])]
    df['co_prediction'] = [np.pi/4 * vti * lvot**2 * heart_rate * unit_factor for (vti, lvot, heart_rate) in zip(df['vti_prediction'], df['lvot_prediction'], df['heart_rate'])]

    ## Make dataframe with only the CO values.
    df_co = df.loc[(~df['co_actual'].isna()) & (~df['co_prediction'].isna())]
    df_co = df_co[['Subject', 'co_actual', 'co_prediction']]

    print(df_co)
    
    ## Save dataframes.
    df = df[['Subject'] + [col for col in df.columns if not col == 'Subject']] # reorder the columns
    out_name_all = (prefix + '-' if not prefix is None else '') + 'co_lvot_vti.csv'
    out_path_all = os.path.join(out_dir, out_name_all)
    df.to_csv(out_path_all, index=False)

    out_name_co = (prefix + '-' if not prefix is None else '') + 'co-pred_and_actual.csv'
    out_path_co = os.path.join(out_dir, out_name_co)
    df_co.to_csv(out_path_co, index=False)
    return

if (__name__ == '__main__'):
    # Create argument parser.
    description = '''Calculate actual and predicted cardiac output from input heart rate, velocity-time integrals, and left-ventricular outflow tract diameters.'''
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Define positional arguments.
    parser.add_argument('vti_path', help='path to CSV file with actual and predicted VTI values')
    parser.add_argument('lvot_path', help='path to CSV file with actual and predicted LVOT diameter values')
    
    # Define optional arguments.
    parser.add_argument('-r', '--heart_rate_path', help='path to CSV file with heart rates', default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/vti_heart_rates_from_dicom.csv')

    parser.add_argument('-o', '--out_dir', type=str, help='path to output directory', default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/cardiac_output')

    parser.add_argument('-p', '--prefix', type=str, help='prefix for output file (e.g. "train")', default=None)
    parser.add_argument('--vti_units', type=str, help='units of input VTI', default='cm')
    parser.add_argument('--lvot_units', type=str, help='units of input LVOT diameter', default='mm')
    parser.add_argument('--out_units', type=str, help='units for cardiac output', default='L/min')

    # Print help if no arguments input.
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit()

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
