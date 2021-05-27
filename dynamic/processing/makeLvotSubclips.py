#!/usr/bin/env python3

import os, sys
import argparse
import glob
import pandas as pd
import cv2
import numpy as np

def main(in_path, out_dir, train_only=False, multiple_measurements=False):
    df = pd.read_csv(in_path)
    new_df = pd.DataFrame(columns=df.columns)
    
    #for index in df[~df[cycle+'_midframe'].isna()].index:
    for index in df.index:
        if train_only:
            if df.loc[index, 'split_all_random'] != 'train': # use the full video for the test and val sets.
                new_df.loc[len(new_df)+1, :] = df.loc[index, :]
                if multiple_measurements:
                    new_df.loc[len(new_df), 'cycle_LVOT_px'] = new_df.loc[len(new_df), 'LVOT_mean_px']
                else:
                    new_df.loc[len(new_df), 'cycle_LVOT_mean_px'] = new_df.loc[len(new_df), 'LVOT_mean_px']
                continue
        for cycle in ['C1', 'C2']:
            if np.isnan(df.loc[index, cycle+'_midframe']):
                continue
            lower = int(df.loc[index, cycle+'_lower'])
            upper = int(df.loc[index, cycle+'_upper'])

            num_frames = int(df.loc[index, 'NumberOfFrames'])

            fps = df.loc[index, 'FPS']
            
            avi_in_path = df.loc[index, 'FilePath']
            avi_out_path = os.path.join(out_dir, os.path.basename(avi_in_path).replace('.avi', '_'+cycle+'.avi'))

            #print(avi_in_path, avi_out_path, lower, upper)

            # Read in video
            cap = cv2.VideoCapture(avi_in_path)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            video_shape = (frame_width, frame_height)

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

            out = cv2.VideoWriter(avi_out_path, fourcc, fps, video_shape)
            for ii in range(num_frames):
                ret, frame = cap.read()
                if ii in range(lower - 1, upper):
                    out.write(frame)

            out.release()
            cap.release()

            # Add row to new dataframe.
            if multiple_measurements:
                for ii in range(1, 3+1): # add three values for current cycle (e.g. C2_LVOTno3_px)
                    new_df.loc[len(new_df)+1, :] = df.loc[index, :]
                    new_df.loc[len(new_df), 'Subject'] = new_df.loc[len(new_df), 'Subject'] + '_' + cycle + '_' + str(ii)
                    new_df.loc[len(new_df), 'FilePath'] = avi_out_path
                    new_df.loc[len(new_df), 'cycle_LVOT_px'] = new_df.loc[len(new_df), cycle+'_LVOTno'+str(ii)+'_px']
            else:
                # Add row to new dataframe.
                new_df.loc[len(new_df)+1, :] = df.loc[index, :]
                new_df.loc[len(new_df), 'Subject'] = new_df.loc[len(new_df), 'Subject'] + '_' + cycle
                new_df.loc[len(new_df), 'FilePath'] = avi_out_path
                new_df.loc[len(new_df), 'cycle_LVOT_mean_px'] = new_df.loc[len(new_df), cycle+'_LVOT_mean_px']
                    

    suffix = '-subclips'
    if train_only:
        suffix += '-train_only'
    if multiple_measurements:
        suffix += '-multiple_measurements'
    suffix += '.csv'
                

    new_df_out_path = in_path.replace('.csv', suffix)
    new_df.to_csv(new_df_out_path, index=False)
    print ('Saved spreadsheet to:', new_df_out_path)
    
    return

if (__name__ == '__main__'):
    # Create argument parser.
    description = """"""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("in_path", help='path to CSV containing columns: FilePath, C1_lower, C1_upper, C2_lower, C2_upper')
    
    # Define optional arguments.
    #parser.add_argument("-i", "--in_dir", help="/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/lvot_n261")
    parser.add_argument("-o", "--out_dir", type=str, default="/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/lvot_n261-subclips")
    parser.add_argument("-t", "--train_only", action='store_true', help='only subsample the training data.')
    parser.add_argument('-m', '--multiple_measurements', action='store_true', help='use up to three LVOT values for each cycle, creating a separate row for each.')
    
    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
