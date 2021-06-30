#!/usr/bin/env python3

import numpy as np
import os
import sys
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, closing, opening, area_closing, area_opening
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image
import cv2
import pandas as pd

def main():
    #in_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225'
    in_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225-no_green'

    in_paths = glob(in_dir + '/*.png')

    split = 'all' # whether to split the inputs into separate images for each peak and generate a new dataframe with a separate row for each peak.

    x_max_group = 420 # found by checking all the images
    band_max_group = 300 # found by checking all the images
    
    for in_path in in_paths:
        img_pil = Image.open(in_path)
        image = np.array(img_pil)

        # Convert to greyscale if image is RGB.
        if (len(image.shape) >= 2) and (image.shape[2] == 3):
            image_gray = rgb2gray(image)
        else:
            image_gray = image[:, :, 0] # For some reason the greyscale image has two color channels. Get just the greyscale value.

        ## Find peaks to measure VTI from.
        # Select a thick horizontal strip in the middle of the image.
        mid_top = int(0.25*image.shape[0])
        mid_bottom = int(0.66*image.shape[0])
        image_mid = image_gray[mid_top:mid_bottom, :]

        # Mark pixels in the strip where the vertical slice has above average intensity (1 if above; else 0)
        bands = image_mid.mean(axis=0) > image_mid.mean()
        bands = bands.astype(int)

        # Find the biggest band in the strip, hopefully corresponding to the "biggest peak".
        band_min = np.inf
        band_max = 0
        band_size = 0
        for ii in range(len(bands)):
            if bands[ii]:
                band_size += 1

            if (band_size > 0) and ((not bands[ii]) or (ii == len(bands) - 1)): # if at the end of a band, or the right edge of the image.
                band_max = max(band_max, band_size)
                band_min = min(band_min, band_size)
                band_size = 0

        # Store the largest band across all images.
        #band_max_group = max(band_max_group, band_max)
        #x_max_group = max(x_max_group, image_gray.shape[0])
                
        # Set the minimum band size, which will be used to determine which peaks will be included.
        band_cutoff = int(0.8*band_max)

        # Zero out the bands that are too small.
        band_size = 0
        for ii in range(len(bands)):
            if bands[ii]:
                band_size += 1
            if (0 < band_size < band_cutoff) and ((not bands[ii]) or (ii == len(bands) - 1)): # if at the end of a band or right edge of the strip, and the band is still below the cutoff.
                bands[ii-band_size:ii+1] = 0 # zero out the whole band.
                band_size = 0
            elif (not bands[ii]): # if at end of an included band
                band_size = 0

        # Number the bands sequentially.
        label = 1
        left_val = 0 # store value to the left; treat left eedge of image as not part of a band.
        for ii in range(len(bands)):
            if bands[ii]:
                bands[ii] = label
            elif (not bands[ii]) and left_val: # If at right edge, increment the label.
                label += 1
            left_val = bands[ii]

            
        # Extend the peaks to the left and right so that the include the whole peak region. Assume that the extended bounds will never overlap.
        bands_copy = bands.copy()
        left_extend = int(band_max*0.5) # size of extension to the left
        right_extend = int(band_max*1.0) # size of extension to the right
        left_val = 0 # store the value of the pixel to the left to detect band edges; treat left edge of image as an edge.
        bad_labels = [] # store a list of labels corresponding to bad peaks, to be deleted later.
        for ii in range(len(bands_copy)):
            # If left edge of band ...
            if bands_copy[ii] and (not left_val):
                # ... extend band to left.
                left_bound_theoretical = ii - left_extend
                left_bound = max(0, left_bound_theoretical)

                # Check that the majority of the extension will be within the plot area.
                left_ratio_threshold = 0.8
                num_out_of_bounds = np.count_nonzero(np.array(range(left_bound_theoretical, ii)) < 0) # number of indices of the extension which would be out of the plot area
                num_in_blacked_out_region = np.count_nonzero((image_gray[4:int(0.5*image_gray.shape[0]), left_bound:ii] == 0).all(axis=0)) # number of indices which would be in a blacked out area
                num_invalid = num_out_of_bounds + num_in_blacked_out_region
                percent_invalid = num_invalid/left_extend*100
                if num_invalid > left_extend*left_ratio_threshold: # if most of the extension is invalid, mark this band for deletion; it is probably an incomplete peak.
                    bad_label = bands_copy[ii]
                    bad_labels.append(bad_label)
                    #print(f'Found bad peak during leftward extension in peak #{bad_label} for image {os.path.basename(in_path)} \n\t{num_out_of_bounds}px out-of-image \n\t{num_in_blacked_out_region}px in blacked-out region \n\t{left_extend}px extension \n\t{percent_invalid:.2f}% out of bounds')
                
                # Check that extension will no overlap with another band.
                while True:
                    if not (bands[max(0, left_bound-1):ii]>0).any(): # If no overlap, proceed.
                        break
                    left_bound += int(left_extend*0.1) # If extension is too big, make it a bit smaller.
                    #print('Resolving leftward extension overlap for: '+os.path.basename(in_path))
                    
                #bands[left_bound:ii] = 1
                bands[left_bound:ii] = bands_copy[ii]

            # If right edge of band ...
            if ((not bands_copy[ii]) and left_val) or (bands_copy[ii] and (ii == len(bands_copy)-1)): # right-edge of band may also be right-edge of image.
                # ... extend band to the right.
                right_bound_theoretical = ii+right_extend
                right_bound = min(len(bands)-1, ii+right_extend)

                # Check that the majority of the extension will be within the plot area.
                right_ratio_threshold = 0.8
                num_out_of_bounds = np.count_nonzero(np.array(range(ii, right_bound_theoretical)) >= len(bands))
                num_in_blacked_out_region = np.count_nonzero((image_gray[4:int(0.5*image_gray.shape[0]), ii:right_bound] == 0).all(axis=0))
                num_invalid = num_out_of_bounds + num_in_blacked_out_region
                percent_invalid = num_invalid/right_extend*100

                if num_invalid > right_extend*right_ratio_threshold: # if most of the extension is invalid, mark this band for deletion; it is probably an incomplete peak.
                    bad_label = left_val
                    bad_labels.append(bad_label)
                    #print(f'Found bad peak during rightward extension in peak #{bad_label} for image {os.path.basename(in_path)} \n\t{num_out_of_bounds}px out-of-image \n\t{num_in_blacked_out_region}px in blacked-out region \n\t{right_extend}px extension \n\t{percent_invalid:.2f}% out of bounds')
                    
                # Extend band to the right if not already at the right edge.
                if (ii != len(bands_copy)-1):
                    # Check that extension will not overlap with another band.
                    while True:
                        if not (bands[ii:min(len(bands)-1, right_bound+1)]>0).any():
                            break
                        right_bound -= int(right_extend*0.1)
                        #print('Resolving rightward extension overlap for: '+os.path.basename(in_path))
                    
                    #bands[ii:right_bound] = 1
                    bands[ii:right_bound] = left_val
                
            # Store the previous value.
            left_val = bands_copy[ii]

        # Remove the bad peaks.
        if len(bad_labels) < bands.max():
            for label in bad_labels:
                bands[bands==label] = 0
        else:
            print(f'Warning: All peaks identified as bad. Keeping all peaks. File: {in_path}')
            
        # Mark the peaks for inspection.
        image_peaks = image_gray.copy()
        #image_peaks[int(image_peaks.shape[0]/2-2):int(image_peaks.shape[0]/2-2)+5, bands==1] = 255
        #image_peaks[-10:, bands>0] = 255
        for label in range(1, bands.max()+1):
            strip_ymin = int(image_peaks.shape[0]/2-2)
            strip_ymax = int(image_peaks.shape[0]/2-2)+5
            image_peaks[strip_ymin:strip_ymax, bands==label] = min(255, 40+20*label)
            

        # Create a binarized version of the image using an intensity cutoff.
        #image_bin = image_gray.copy()
        #image_bin[image_bin < image_bin.mean()] = 0
        #image_bin[image_bin > 0] = 255

        #image_bin_peaks = image_bin.copy()
        #image_bin_peaks[int(image_peaks.shape[0]/2-2):int(image_peaks.shape[0]/2-2)+5, bands==1] = 255

        # Create an image with Gaussian blur.
        #image_blur_bin = cv2.GaussianBlur(image_bin, (5,5), 0)
        #image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)
        
        out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225-split_peaks'
        
        
        os.makedirs(out_dir, exist_ok=True)

        # If splitting peaks into separate files, save each file and populate a new spreadsheet.
        if split == 'all':
            
            df_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti.csv'
            df = pd.read_csv(df_path)
            df_new = pd.DataFrame(columns=df.columns)
            
            for label in range(1, bands.max()+1):
                if not label in bands:
                    continue # skip if this peak was deleted
                peak_img = np.zeros(shape=(x_max_group, band_max_group), dtype=image_gray.dtype)
                x_max = image_gray.shape[0] # vertical height of image
                y_max = np.count_nonzero(bands==label) # width of band

                peak_img[:x_max, :y_max] = image_gray[:, bands==label]

                out_name = os.path.basename(in_path).replace('.png', '_'+str(label)+'.png')
                out_path = os.path.join(out_dir, out_name)

                image = Image.fromarray(peak_img)
                image.save(out_path)

                # Add a new row in the spreadsheet.
                subject_new = os.path.basename(out_path).split('.png')[0]
                subject = subject_new.split('_')[0]
                
                if not subject in df['Subject'].tolist():
                    print('Subject is missing row in FileList_vti.csv :', subject)
                    continue
            
                df_new = pd.concat([df_new, df[df['Subject']==subject]], ignore_index=True)
                df_new.loc[df_new.index[-1], 'Subject'] = subject_new
                df_new.loc[df_new.index[-1], 'FilePath'] = out_path

    if split == 'all':
        df_path_new = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_all.csv'
        df_new.to_csv(df_path_new, index=False)
                

    ## Generate a new dataframe with separate rows for each sub image if splitting performed.
    if split == 'all':
        df_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti.csv'
        df = pd.read_csv(df_path)
        
        df_new = pd.DataFrame(columns=df.columns)
        new_paths = glob(out_dir+'/*.png')
        new_paths.sort()
        
        for new_path in new_paths:
            subject_new = os.path.basename(new_path).split('.png')[0]
            subject = subject_new.split('_')[0]

            if not subject in df['Subject'].tolist():
                print('Subject is missing row in FileList_vti.csv :', subject)
                continue
            
            df_new = pd.concat([df_new, df[df['Subject']==subject]], ignore_index=True)
            df_new.loc[df_new.index[-1], 'Subject'] = subject_new
            df_new.loc[df_new.index[-1], 'FilePath'] = new_path

        df_path_new = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_all.csv'
        df_new.to_csv(df_path_new, index=False)

if __name__ == '__main__':
    main()
