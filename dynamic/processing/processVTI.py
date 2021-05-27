#!/usr/bin/env python

import os, sys
import numpy as np
#import matplotlib
import pydicom
import argparse
import pandas as pd
from glob import glob
from PIL import Image
import cv2


def main(in_dir, out_dir, out_type='png', out_mode='RGB', debug=False, new_height=None, new_width=None, pad=False):
    in_paths = glob(in_dir + '/*')
        
    in_paths.sort()

    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame()
    df.index.name = 'Subject'
    
    for ii, in_path in enumerate(in_paths):
        if not in_path.lower().endswith('.dcm'):
            continue

        ## Add data to dataframe.
        subject = os.path.basename(in_path).split('.')[0]
        df.loc[subject, 'DicomFilePath'] = os.path.abspath(in_path)
                
        ## Load the image.
        dicom = pydicom.dcmread(in_path)
        pixel_data = dicom.pixel_array

        ## Get information from DICOM header.
        try:
            vt_seq = dicom[(0x0018,0x6011)][1]
            for line in dicom[(0x0018,0x6011)][1]:
                df.loc[subject, line.keyword] = line.value
        except:
            print('Cannot find information about V-T sequence for:', subject)
            continue
        
        ## Crop the V-T graph. 
        # Most images have the V-T graph within bounds (50, 913, 212, 671). It looks like some are a smaller area within this box, but none appear to extend beyond it.
        # The ReferencePixelY0 DICOM tag indicates the pixel where the x axis lies. We can cut off everything above this.
        # Cropping does not affect scaling.
        left = 50
        right = 913
        top = 212 # top of the V-T plot.
        top += int(df.loc[subject, 'ReferencePixelY0'])# x-axis of V-T plot
#        top += 1 # Do not include the axis itself.
        bottom = 671
        pixel_data = pixel_data[top:bottom+1, left:right+1, :]

        df.loc[subject, 'left'] = left
        df.loc[subject, 'right'] = right
        df.loc[subject, 'top'] = top
        df.loc[subject, 'bottom'] = bottom
        
        ## Zero everything above the y-axis which is irrelevant.
        #ReferencePixelY0 = int(df.loc[subject, 'ReferencePixelY0'])
        #pixel_data[:ReferencePixelY0, :, :] = 0
        
        ## Pad
        # The largest window below the x axis is (417, 864). Pad all images to this size.
        # Padding does not affect scaling.
        if pad:
            max_height = 417
            max_width = 864
            new = np.zeros((max_height, max_width, 3), dtype=pixel_data.dtype)
            new[:pixel_data.shape[0], :pixel_data.shape[1], :] = pixel_data
            pixel_data = new

        ## Rescale
        # Rescaling not implemented, but keep track of rescaling factor for future referece.
        df.loc[subject, 'old_height'] = pixel_data.shape[0]
        df.loc[subject, 'old_width'] = pixel_data.shape[1]

        if (new_height is None) and (new_width is None):
            new_shape = (pixel_data.shape[0], pixel_data.shape[1])
        else:
            if (new_height is None):
                new_height = pixel_data.shape[0]
            elif (new_width is None):
                new_width = pixel_data.shape[1]
            new_shape = (new_height, new_width) # (y, x) = (height, width)
            pixel_data = cv2.resize(pixel_data, (new_width, new_height), interpolation=cv2.INTER_LINEAR) # not sure what interpolation would be best here; try default first.

        rescale_y = pixel_data.shape[0]/new_shape[0]
        rescale_x = pixel_data.shape[1]/new_shape[1]

        df.loc[subject, 'new_height'] = pixel_data.shape[0]
        df.loc[subject, 'new_width'] = pixel_data.shape[1]
        
        df.loc[subject, 'rescale_y'] = rescale_y
        df.loc[subject, 'rescale_x'] = rescale_x

        im = Image.fromarray(pixel_data, mode='YCbCr')
        im = im.convert(out_mode)
        
        out_name = '.'.join(os.path.basename(in_path).split('.')[:-1]) + '.' + out_type

        ## Save the preprocessed image.
        out_path = os.path.join(out_dir, out_name)
        im.save(out_path)

        # Save processed image path to dataframe.
        df.loc[subject, 'FilePath'] = os.path.abspath(out_path)


    ## Get the final scaling factor for each row.
    # True VTI (cm) = (# processed pixels) * rescale_x * rescale_y * PhysicalDeltaX * PhysicalDeltaY
    #               = (# processed pixels) * pixel_scale_factor
    df['pixel_scale_factor'] = [a*b*c*d for (a,b,c,d) in zip(df['rescale_x'], df['rescale_y'], df['PhysicalDeltaX'], df['PhysicalDeltaY'])]
        
    df_name = 'vti_png_data.csv'
    df_path = os.path.join('/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti', df_name)
    df.to_csv(df_path, index=True)
    return

if __name__ == '__main__':
    # Create argument parser.
    description = '''Crop the velocity-time curve from a Doppler echo image. Input must be a PNG, e.g. as generated using 'convert' from ImageMagick.'''
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument('in_dir', type=str, help='input directory containing DICOM VTI files')
    parser.add_argument('out_dir', type=str, help='directory to save processed PNG files to')
    
    # Define optional arguments.
    parser.add_argument('-m', '--out_mode', type=str, help='output color mode. Can be "RGB" or "LA" (greyscale). Default: "RGB"', default='RGB')
    parser.add_argument('-t', '--out_type', type=str, help='output image type. Default: "png"', default='png')
    parser.add_argument('-d', '--debug', action='store_true', help='debug; only do 10 files.')
    parser.add_argument('-y', '--new_height', type=int, help='height of new image (pixels)')
    parser.add_argument('-x', '--new_width', type=int, help='width of new image (pixels)')
    parser.add_argument('-p', '--pad', action='store_true', help='pad cropped images to size of largest cropped image before rescaling')
    
    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
