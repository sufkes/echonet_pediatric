#!/usr/bin/env python3

import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import pydicom
import cv2
from PIL import Image


def mask(image):
    dimension = image.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    
    mask = ((m1+m2)>int(dimension/2) + int(dimension/10)) 
    mask *=  ((m1-m2)<int(dimension/2) + int(dimension/10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image


def main(in_paths, out_dir, data_path, skip_mask, flip, retain_color_channels, skip_match_hw, border_trim_size, out_shape):
    # Set up the dataframe which will store scan information.
    if not data_path is None:
        if os.path.exists(data_path):
            raise Exception(f'data_path already exists: {data_path}')
        if not data_path.lower().endswith('.csv'):
            raise Exception(f'data_path must end in ".csv": {data_path}')
        
        data_df = pd.DataFrame()
        data_df.index.name = 'Subject'
        
    
    # Loop over scans.
    for in_path in in_paths:
        # Open the DICOM file.
        dicom = pydicom.dcmread(in_path, force=True)
        dicom_array = dicom.pixel_array
        shape_original = dicom_array.shape
        if len(shape_original) != 4:
            print(f'Warning: Skipping input file, because it has only three dimensions, but it must have 4 (frame, height, width, color): {in_path}')
            continue
        
        # Set out path.
        os.makedirs(out_dir, exist_ok=True)
        if in_path.lower().endswith('.dcm'):
            out_name = os.path.basename(in_path)[:-4] + '.avi'
        else:
            out_name = os.path.basename(in_path) + '.avi'
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            raise Exception(f'Out path already exists: {out_path}')
        print(f'In: {in_path}')
        print(f'Out: {out_path}')
        print('')
            
        # Get frame rate. Need a value to start writing the video.
        try:
            fps_in = dicom[(0x18, 0x40)].value
            fps_out = fps_in
        except:
            fps_in = None
            fps_out = 30
            print("Warning: Could not find frame rate in DICOM header, defaulting to 30.")
        
            
        # Start up the video writer.
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out_shape = tuple(out_shape)
        out = cv2.VideoWriter(out_path, fourcc, fps_out, out_shape)

        # Cut off rows and columns at the edges such that the number of rows matches the number of columns.
        if not skip_match_hw:
            bias = int(np.abs(dicom_array.shape[2] - dicom_array.shape[1])/2)
            if bias>0:
                if dicom_array.shape[1] < dicom_array.shape[2]:
                    dicom_array = dicom_array[:, :, bias:-bias, :]
                else:
                    dicom_array = dicom_array[:, bias:-bias, :, :]
        shape_match_hw = dicom_array.shape
        #print(f'shape_match_hw: {shape_match_hw}')
        
        frames, height, width, channels = dicom_array.shape
                    
        for i in range(frames):

            frame = dicom_array[i, :, :, :]
            
            if border_trim_size > 0:
                border_width = int(width*border_trim_size)
                border_height = int(height*border_trim_size)
                frame = frame[border_height:height-border_height, border_width:width-border_width, :]
            shape_border_trim = frame.shape
            
            # Resize image.
            frame = cv2.resize(frame, dsize=out_shape, interpolation=cv2.INTER_CUBIC)
            shape_resize = frame.shape
            

            # "Mask pixels outside the scan region". Applied mask is diamond-shaped and cuts off part of the scan with no clear benefit.
            if not skip_mask:
                frame = mask(frame)
                
            # Flip the image 180 degrees.
            if flip:
                frame = np.rot90(frame, k=2)

            # Pick which color channels to keep.
            if retain_color_channels:
                ## Note: It looks all of our 4C and LVOT scans have PhotometricInterpretation = YBR_FULL_422. Most of the VTI scans have YBR_FULL_422, but some have 

                ## DICOM Documentation about YBR_FULL:
                # Pixel data represent a color image described by one luminance (Y) and two chrominance planes (CB and CR). This photometric interpretation may be used only when Samples per Pixel (0028,0002) has a value of 3. May be used for pixel data in a Native (uncompressed) or Encapsulated (compressed) format; see Section 8.2 in PS3.5 . Planar Configuration (0028,0006) may be 0 or 1.

                # Black is represented by Y equal to zero. The absence of color is represented by both CB and CR values equal to half full scale.

                ## DICOM Documentation about YBR_FULL_422:
                # The same as YBR_FULL except that the CB and CR values are sampled horizontally at half the Y rate and as a result there are half as many CB and CR values as Y values

                # This explains why keeping only the first color channel results in a sensible black and white image for (as far as I know) all of our scans.

                # Convert the pixel array from YBR_FULL_422 to RGB.
                im_rgb = Image.fromarray(frame, mode='YCbCr') # I hope this handles YBR_FULL_422 correctly, which is evidently slightly different from YBR_FULL
                im_rgb = im_rgb.convert('RGB')
                frame = np.array(im_rgb)
                
                output = frame
            else:
                channel_zero = frame[:, :, 0]
                output = cv2.merge([channel_zero, channel_zero, channel_zero])

            # Add current frame to the output
            out.write(output)

        # Close the video writer.
        out.release()

        #print(f'shape_border_trim: {shape_border_trim}')
        #print(f'shape_resize: {shape_resize}')
        
        ## Record file information to a spreadsheet. Read information from the DICOM header.
        if not data_path is None:
            # Set subject name based on input file name.
            subject = '.'.join(os.path.basename(in_path).split('.')[:-1])

            ## Record information.
            # File paths.
            data_df.loc[subject, 'FilePath'] = os.path.abspath(out_path)
            data_df.loc[subject, 'DicomFilePath'] = os.path.abspath(in_path)
            
            # Size information
            data_df.loc[subject, 'height_original'] = shape_original[1] # many frames
            data_df.loc[subject, 'width_original'] = shape_original[2]
            data_df.loc[subject, 'height_match_hw'] = shape_match_hw[1] # many frames
            data_df.loc[subject, 'width_match_hw'] = shape_match_hw[2]
            data_df.loc[subject, 'height_border_trim'] = shape_border_trim[0] # 1 frame
            data_df.loc[subject, 'width_border_trim'] = shape_border_trim[1]
            data_df.loc[subject, 'height_resize'] = shape_resize[0] # 1 frame
            data_df.loc[subject, 'width_resize'] = shape_resize[1]

            data_df.loc[subject, 'rescale_height'] = shape_border_trim[0]/shape_resize[0] # the factor by which the phyiscal width of a pixel increase
            data_df.loc[subject, 'rescale_width'] = shape_border_trim[1]/shape_resize[1]
            
            data_df.loc[subject, 'FrameHeight'] = shape_resize[0] # Record this shape again for consistency with the original EchoNet and my modification of the script.
            data_df.loc[subject, 'FrameWidth'] = shape_resize[1]

            data_df.loc[subject, 'NumberOfFrames'] = shape_original[0]
            
            # FPS
            data_df.loc[subject, 'FPS_dicom'] = fps_in
            data_df.loc[subject, 'FPS_avi'] = fps_out

            # DICOM tags (which may or may not be present
            tag_dict = OrderedDict([('PhotometricInterpretation', (0x0028,0x0004)), # color type
                                    ('StudyDate', (0x0008,0x0020)),
                                    ('Manufacturer', (0x0008,0x0070)),
                                    ('StationName', (0x0008,0x1010)),
                                    ('ManufacturerModelName', (0x0008,0x1090)),
                                    ('HeartRate', (0x0018,0x1088))
            ])
            
            for column, tag in tag_dict.items():
                try:
                    data_df.loc[subject, column] = dicom[tag].value
                except IndexError:
                    data_df.loc[subject, column] = None            

    # Save the dataframe to CSV.
    if not data_path is None:
        data_df.to_csv(data_path, index=True)
    return

if (__name__ == '__main__'):
    # Create argument parser.
    description = """Convert ultrasound DICOM files to AVI. This is based on the script provided in the EchoNet-Dynamic git repository. The original script crashes for me. When our images are processed using the original script, they appear different to the processed EchoNet images in the following ways: (1) flipped 180 degrees; and (2) a diamond-shape mask is applied, which seems to remove the round edge of the scan region. Thus, this script optionally applies an additional 180 degree rotation, and optionally skips the masking step, such that the process AVI files are as similar as possible to the processed videos in the EchoNet dataset."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Define positional arguments.    
    
    # Define optional arguments.
    parser.add_argument("-i", "--in_paths", type=str, help="paths to DICOM files", nargs="+", required=True)
    parser.add_argument("-o", "--out_dir", type=str, help="path to output directory", required=True)
    parser.add_argument("-d", "--data_path", type=str, help="path to CSV file to which paths, DICOM tags, and image processing parameters will be recorded.")
    
    parser.add_argument("-m", "--skip_mask", help="In the original EchoNet processing a mask is applied to 'mask pixels outside the scanning sector'. This action seems to strip away the round edge of the scan, leaving it in a diamond shape, unlike the processed EchoNet scans. Use this option to skip applying the masking step.", action="store_true")
    parser.add_argument("-f", "--flip", help="In the original EchoNet processing, no 180 degree flip is applied, but our scans processed through that end up rotated 180 degrees compared to the EchoNet data. Use this flag to apply 180 degree rotation to the final image to match the processed EchoNet data.", action="store_true")
    parser.add_argument("-c", "--retain_color_channels", action="store_true", help="In the original EchoNet processing, the first color channel of the input DICOM is mapped to all three color channels of the output AVI. However, the processed EchoNet videos sometimes have multiple color channels, so they were clearly not processed this way. Use this flag to retain all three color channels.")

    parser.add_argument("--skip_match_hw", action="store_true", help="In the original EchoNet processing, the height and width (H, W) are matched by cropping the center of the video in the larger dimension (i.e if H>W, crop the image to the slice [H/2-W/2:H/2+W/2, :]; if H<W, crop the image to the slice [:, W/2-H/2:W/2+H/2]). Use this flag to skip this cropping step.")
    parser.add_argument("--border_trim_size", type=float, help="In the original EchoNet processing, after the height and width are matched, the image is cropped by removing a border of pixels of width 0.1 times the image width/height from all four sides. Use this flag to change the size of the border that is removed to something other than 0.1.", default=0.1)
    parser.add_argument("--out_shape", type=int, nargs=2, help="In the original EchoNet processing, after matching the height and width, and removing a border, the images are rescaled to a common size (112, 112). Use this flag to change that size.", default=(112, 112))

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
