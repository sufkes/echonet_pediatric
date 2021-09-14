#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd
from glob import glob
import numpy as np
#import cv2
from PIL import Image
#from skimage.color import rgb2gray
import re
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def getPeakCurve(img_path):
    '''Read the vertical axis values of the modal-velocity curve. Horizontal positions with no peak marker will be assumed to not lie within the peak region, and will be assigned vertical positions 0. 
Parameters: 
    img_path (str): path to input image with peak marked in red
Returns: 
    peak (numpy.array): vector of length equal to horizontal length of input image, containing the vertical position of the modal velocity curve; down is treated as the positive direction.'''

    img = Image.open(img_path)
    pixel_data = np.array(img) # image is RGB-encoded, but all pixels will have R==G==B, except the line marking the peak.
            
    r = pixel_data[..., 0] # red component
    g = pixel_data[..., 1] # green
    b = pixel_data[..., 2] # blue
        
    line = (r>g) & (r>b) # booleans

    peak = np.argmax(line, axis=0).astype(np.float32) # vertical axis values of peak
    
    return peak
    
def main(file_list_in_path_rescale, file_list_in_path_pad, out_dir, in_color_mode, make_plots):
    ## Read data sheets.
    df_rescale = pd.read_csv(file_list_in_path_rescale)
    df_pad = pd.read_csv(file_list_in_path_pad)
    
    # Find the annotated images.
    for index in df_rescale.index:
        img_path_rescale = df_rescale.loc[index, 'FilePath_annotated']

        # Record the subject corresponding to this image.
        subject_base = os.path.basename(img_path_rescale).split('_')[0]
        df_rescale.loc[index, 'Subject_base'] = subject_base
        df_pad.loc[index, 'Subject_base'] = subject_base
        
        ## Open image and get the peak line in rescaled space.
        peak_rescale = getPeakCurve(img_path_rescale)

        ## Scale the annotated peak back to its original size and shape. 
        old_width = df_rescale.loc[index, 'old_width']
        old_height = df_rescale.loc[index, 'old_height']
        new_width = df_rescale.loc[index, 'new_width']
        new_height = df_rescale.loc[index, 'new_height']

        x_range = np.arange(new_width, dtype=float) # placeholder x values in space of rescaled image (0, 1, 2, ..., new_width)
        x_range_compressed = np.arange(new_width, dtype=float) * old_width / new_width # array of VT plot x values for the padded image in the space of the rescaled image; used for interpolation

        interpolater = interpolate.interp1d(x_range_compressed, peak_rescale * old_height / new_height, fill_value=0, bounds_error=False) # returns a function used to map the peak shape defined in the rescaled image to the peak shape in the padded image, with zeros in the zero-padded region, per the fill_value and bounds_error arguments.
        peak_pad = interpolater(x_range) # values for the peak in the padded image, in the space of the padded image
        peak_pad = peak_pad.astype(np.float32)
        
        ## Statistics
        vti_px_img_rescale = peak_rescale.sum() # VTI in units of pixels as measured by the annotation on the rescaled image.
        if vti_px_img_rescale == 0: # if this image was not annotated
             raise Exception('No peak annotation for file: '+img_path_rescale)
        vti_img_rescale = vti_px_img_rescale*df_rescale.loc[index, 'pixel_scale_factor'] # VTI in units of cm/s as measured by the annotation on the rescaled image.
        #vti_px_real_rescale = df_rescale.loc[index, 'AOVTI_px'] # ground truth VTI in units of pixels in the rescaled image
        vti_real = df_rescale.loc[index, 'AOVTI'] # ground truth VTI in units of cm/s
        percent_difference = (vti_img_rescale - vti_real)/vti_real*100

        df_rescale.loc[index, 'AOVTI_px_annotation'] = vti_px_img_rescale
        df_rescale.loc[index, 'AOVTI_annotation'] = vti_img_rescale
        df_rescale.loc[index, 'percent_difference'] = percent_difference

        vti_px_img_pad = peak_pad.sum() # VTI in units of pixels as measured by the annotation on the padded image.
        vti_img_pad = vti_px_img_pad*df_pad.loc[index, 'pixel_scale_factor'] # VTI in units of cm/s as measured by the annotation on the padded image.
        #vti_px_real_pad = df_pad.loc[index, 'AOVTI_px'] # ground truth VTI in units of pixels in the padded image
        vti_real = df_pad.loc[index, 'AOVTI'] # ground truth VTI in units of cm/s
        percent_difference = (vti_img_pad - vti_real)/vti_real*100

        df_pad.loc[index, 'AOVTI_px_annotation'] = vti_px_img_pad
        df_pad.loc[index, 'AOVTI_annotation'] = vti_img_pad
        df_pad.loc[index, 'percent_difference'] = percent_difference
        
        
        ## Save the peak as a np.array, and update the file list spreadsheet.
        img_path_basename = os.path.basename(img_path_rescale)

        out_dir_rescale = out_dir+'-rescale'
        os.makedirs(out_dir_rescale, exist_ok=True)
        out_name_rescale = img_path_basename.replace('.png', '.npy')
        out_path_rescale = os.path.join(out_dir_rescale, out_name_rescale)
        np.save(out_path_rescale, peak_rescale)
        df_rescale.loc[index, 'peak_array_path'] = out_path_rescale
        file_list_out_path_rescale = file_list_in_path_rescale.replace('.csv', '-with_peak_arrays.csv')
        df_rescale.to_csv(file_list_out_path_rescale, index=False)
        if make_plots:
            plt.figure()
            plt.plot(peak_rescale)
            plt.savefig(os.path.join(out_dir_rescale, img_path_basename))
            plt.close()
            
        out_dir_pad = out_dir+'-pad'
        os.makedirs(out_dir_pad, exist_ok=True)
        out_name_pad = img_path_basename.replace('.png', '.npy')
        out_path_pad = os.path.join(out_dir_pad, out_name_pad)
        np.save(out_path_pad, peak_pad)
        df_pad.loc[index, 'peak_array_path'] = out_path_pad
        file_list_out_path_pad = file_list_in_path_pad.replace('.csv', '-with_peak_arrays.csv')
        df_pad.to_csv(file_list_out_path_pad, index=False)
        if make_plots:
            plt.figure()
            plt.plot(peak_pad)
            plt.savefig(os.path.join(out_dir_pad, img_path_basename))
            plt.close()

    # Compare VTI obtained from annotations to ground truth.
    print('Individual peaks:')
    print(df_rescale.loc[~df_rescale['AOVTI_annotation'].isna(), ['Subject', 'AOVTI', 'AOVTI_annotation', 'percent_difference']].to_string(index=False))
    std_df_rescale = df_rescale.groupby('Subject_base').std()
    std_df_rescale = std_df_rescale.loc[~std_df_rescale['AOVTI_annotation'].isna(), ['AOVTI_annotation']]
    mean_df_rescale = df_rescale.groupby('Subject_base').mean()
    mean_df_rescale['percent_difference'] = (mean_df_rescale['AOVTI_annotation'] - mean_df_rescale['AOVTI'])/mean_df_rescale['AOVTI']*100
    mean_df_rescale = mean_df_rescale.loc[~mean_df_rescale['AOVTI_annotation'].isna(), ['AOVTI', 'AOVTI_annotation', 'percent_difference']]
    mean_df_rescale['AOVTI_std'] = std_df_rescale['AOVTI_annotation']
    mean_df_rescale['subject_num'] = [int(re.sub("[^0-9]", "", s)) + (10000 if 'DCM' in s else 0) for s in mean_df_rescale.index]
    mean_df_rescale.sort_values(by='subject_num', inplace=True)
    
    print('\nSubject averages:')
    print(mean_df_rescale.loc[~mean_df_rescale['AOVTI_annotation'].isna(), ['AOVTI', 'AOVTI_annotation', 'percent_difference', 'AOVTI_std']].to_string())

    return
    

if __name__ == '__main__':
    # Create argument parser.
    description = """Extract velocity-time plot from VTI DICOM images, post-process, save to PNG, and generate a CSV containing data for each processed PNG to be used in the model.

This script assumes that annotations were made to peak images which were all rescaled to the same size. For each peak image, it generates a numpy.array of peak values in the rescaled image space, and another numpy.array in the native space of the image, but with zero-padding to match the size of the rescaled input images."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Define positional arguments.
    #parser.add_argument("png_data_path", help="Path to the spreadsheet containing information from the DICOM headers and DICOM filepaths.") # NEW: THIS DATAFRAME WILL BE GENERATED IN THIS SCRIPT
    
    ## Define optional arguments.
    parser.add_argument('--file_list_in_path_rescale',
                        type=str,
                        help='path to rescaled file list to read information from and update with paths to peak arrays',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_peaks-417x286-rescale.csv')
    parser.add_argument('--file_list_in_path_pad',
                        type=str,
                        help='path to padded file list to read information from and update with paths to peak arrays',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_peaks-417x286-pad.csv')
    parser.add_argument('-o', '--out_dir',
                        type=str,
                        help='base path of directory to which to save np.arrays of peak values',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225-peak_values')
    
    # image processing options
    parser.add_argument('-m', '--in_color_mode',
                        type=str,
                        help='input color mode; expect a greyscale image with RGB encoding, and with a red line bordering the modal velocity peak',
                        default='RGB')    
    parser.add_argument('-p', '--make_plots',
                        action='store_true',
                        help='save plots of peak curves')
    
    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
