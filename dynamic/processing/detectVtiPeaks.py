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
    #in_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225-no_pad'
    in_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225'

    in_paths = glob(in_dir + '/*.png')

    for in_path in in_paths:
        img_pil = Image.open(in_path)
        image = np.array(img_pil)

        image_gray = rgb2gray(image)

        mid_top = int(0.25*image.shape[0])
        mid_bottom = int(0.66*image.shape[0])
        image_mid = image_gray[mid_top:mid_bottom, :]

#        peak_bounds = image_mid.copy()
#        peak_bounds[image_mid.mean(axis=1) > image_mid.mean(), 10:15] = 255
#        peak_bounds = image_mid.mean(axis=1) > image_mid.mean()
#        print(peak_bounds)

        #image_mid[10:15, image_mid.mean(axis=0) > image_mid.mean()] = 255
        bands = image_mid.mean(axis=0) > image_mid.mean()
        bands = bands.astype(int)

        band_min = np.inf
        band_max = 0
        band_size = 0
        for ii in range(len(bands)):
            if bands[ii]:
                band_size += 1

            if (band_size > 0) and ((not bands[ii]) or (ii == len(bands) - 1)):
                band_max = max(band_max, band_size)
                band_min = min(band_min, band_size)
                band_size = 0

        band_cutoff = int(0.8*band_max) # remove all bands smaller than this length.
        
        band_size = 0
        for ii in range(len(bands)):
            if bands[ii]:
                band_size += 1
            if (0 < band_size < band_cutoff) and ((not bands[ii]) or (ii == len(bands) - 1)):
                bands[ii-band_size:ii+1] = 0
                band_size = 0
            elif (not bands[ii]): # if at end of an included band
                band_size = 0

#        for ii in range(len(bands)):
#            if bands[ii]:
#                band_size += 1
#            elif band_size > 0:
#                    band_max = max(band_max, band_size)
#                    band_size = 0
#        band_cutoff = int(0.8*band_max) # remove all bands smaller than this length.
#        print(band_max, band_cutoff)

#        band_size = 0
#        for ii in range(len(bands)):
#            if bands[ii]:
#                band_size += 1
#            elif (band_size > 0) and (band_size < band_cutoff):
#                print('miniband size:', band_size)
#                print('index:', ii)
#                print('bef:', bands.sum())
#                print(bands[ii-band_size:ii])
#                bands[ii-band_size:ii] = 0
#                band_size = 0
#                print('aft:', bands.sum())

        image_peaks = image_gray.copy()
        image_peaks[int(image_peaks.shape[0]/2-2):int(image_peaks.shape[0]/2-2)+5, bands==1] = 255
        
#        image_bin = image > image.mean()
#        image_bin[image_bin>0] = 1
        image_bin = image_gray.copy()
        image_bin[image_bin < image_bin.mean()] = 0
        image_bin[image_bin > 0] = 255

        image_bin_peaks = image_bin.copy()
        image_bin_peaks[int(image_peaks.shape[0]/2-2):int(image_peaks.shape[0]/2-2)+5, bands==1] = 255
         
        image_blur_bin = cv2.GaussianBlur(image_bin, (5,5), 0)
        image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)

        split = False
        
        out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/runs/blobs'
        #out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_n225-bin'
        
        os.makedirs(out_dir, exist_ok=True)
        if not split:
            out_image = image_peaks
            
            out_name = os.path.basename(in_path)
            out_path = os.path.join(out_dir, out_name)        
            image = Image.fromarray(out_image)
            image.save(out_path)
        else:
            i1 = image_gray.copy()
            i2 = image_gray.copy()

            i1[:, :int(i1.shape[1]/2)] = 0
            i2[:, int(i2.shape[1]/2):] = 0

            out_name = os.path.basename(in_path).replace('.png', '_1.png')
            out_path = os.path.join(out_dir, out_name)        
            image = Image.fromarray(i1)
            image.save(out_path)

            out_name = os.path.basename(in_path).replace('.png', '_2.png')
            out_path = os.path.join(out_dir, out_name)        
            image = Image.fromarray(i2)
            image.save(out_path)

    if split:
        ## Generate a new dataframe with separate rows for each sub image.
        df_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-good.csv'
        df = pd.read_csv(df_path)
        
        #df_new = pd.DataFrame(columns=df.columns)
        df_new = df.copy()
        new_paths = glob(out_dir+'/*.png')
        new_paths.sort()
        
        for new_path in new_paths:
            subject_new = os.path.basename(new_path).split('.png')[0]
            subject = subject_new.split('_')[0]

            
            
            #print(subject)
            #print(df.loc[df['Subject']==subject, 'split_all_random'])
            #print(df.loc[df['Subject']==subject, 'split_all_random'].values[0])
            #print('')
            
            if (not subject in df['Subject'].tolist()) or (df.loc[df['Subject']==subject, 'split_all_random'].values[0] != 'train'):
                continue
            
            df_new = pd.concat([df_new, df[df['Subject']==subject]], ignore_index=True)
            df_new.loc[df_new.index[-1], 'Subject'] = subject_new
            df_new.loc[df_new.index[-1], 'FilePath'] = new_path

        df_path_new = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti-split_2.csv'
        df_new.to_csv(df_path_new, index=False)

if __name__ == '__main__':
    main()
