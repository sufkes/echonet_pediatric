#!/usr/bin/env python

import os, sys
import glob
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import pydicom
from PIL import Image
import numpy as np

def makeThumbs(in_path, out_path):
    dataset = pydicom.dcmread(in_path)

    # Read in pixel data assuming YCbCr in PhotometricInterpretation
    pixel_data = dataset.pixel_array

    # If image is a video, grab the first frame.
    if len(pixel_data.shape) == 4:
        pixel_data = pixel_data[0,:,:,:]
    
    # Determine PhotometricIntrepretation.
    photometric_interpretation = dataset[0x0028,0x0004].value
    
    # Convert to RGB for thumbnail.
    if photometric_interpretation == 'RGB':
        pixel_data_rgb = pixel_data
    elif photometric_interpretation == 'YBR_FULL_422':
        im_rgb = Image.fromarray(pixel_data, mode='YCbCr')
        im_rgb = im_rgb.convert('RGB')
        pixel_data_rgb = np.array(im_rgb)
    else:
        raise Exception(f'Unknown or unhandled PhotometricInterpretation: {photometric_interpretation}')
        
    plt.figure()
    plt.imshow(pixel_data_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
if (__name__ == '__main__'):
    # Create argument parser
    description = """Input path to DICOM file, generate thumbnail PNG."""
    parser = argparse.ArgumentParser(description=description)
    
    # Define positional arguments.
    parser.add_argument("in_path", help="path to input DICOM file", type=str)
    parser.add_argument("out_path", help="path to output PNG file", type=str)
    
    # Define optional arguments.
    #parser.add_argument("-o", "--out_dir", help="path to directory to which thumbnails will be saved", type=str, default=os.getcwd())

    # Parse arguments.
    args = parser.parse_args()

    # Run the function.
    makeThumbs(**vars(args))
