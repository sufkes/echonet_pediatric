#!/usr/bin/env python

import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import pydicom
import cv2
from glob import glob
from PIL import Image

def mask(image):
    """In the original EchoNet-Dynamic processing script, a mask is applied to 'mask pixels outside the scanning sector'. This action seems to strip away the round edge of the scan, leaving it in a diamond shape, unlike the processed EchoNet scans. This function performs the masking step that appears in the EchoNet-Dynamic script, but is not recommended."""
    dimension = image.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    
    mask = ((m1+m2)>int(dimension/2) + int(dimension/10)) 
    mask *=  ((m1-m2)<int(dimension/2) + int(dimension/10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image

def main(dicom_in_dir, datasheet_path, image_out_dir, file_list_out_path, names_of_length_fields_to_convert_to_pixels, skip_mask, flip, retain_color_channels, skip_match_hw, border_trim_size, out_shape):
    # Load the datasheet.
    df = pd.read_csv(datasheet_path)

    # Set index to the subject column.
    df.set_index('Subject', drop=True, inplace=True) # one row per subject
    
    #### Process DICOM files.
    # Find the DICOM files.
    dicom_in_paths = glob(dicom_in_dir + '/*')
    dicom_in_paths.sort()

    ## Loop over the DICOM files.
    first_subject = True # If this is the first subject in the list.
    for ii, in_path in enumerate(dicom_in_paths):
        if not in_path.lower().endswith('.dcm'):
            msg = 'File does not have extension ".dcm". Skipping.'
            warnings.warn(msg)
            continue

        # Add DICOM path to dataframe.
        subject = os.path.basename(in_path).split('.')[0] # Assume file is named <subject>.dcm
        df.loc[subject, 'DicomFilePath'] = os.path.abspath(in_path)

        # Open the DICOM file.
        dicom = pydicom.dcmread(in_path, force=True)
        dicom_array = dicom.pixel_array
        shape_original = dicom_array.shape
        if len(shape_original) != 4:
            print(f'Warning: Skipping input file, because it has only three dimensions, but it must have 4 (frame, height, width, color): {in_path}')
            continue
        
        # Get information from the "Sequence of Ultrasound Regions Attribute" in the DICOM header. The most important things we get here are PhysicalDeltaX/PhysicalDeltaY, which give us the physical meaning of a pixel in the X and Y directions; and PhysicalUnitsXDirection/PhysicalUnitsYDirection which give the units (e.g. cm)
        try:
            ultrasound_seq = dicom[(0x0018,0x6011)][0] # DICOM sequence tag for the plot (hopefully).
            for line in ultrasound_seq:
                df.loc[subject, line.keyword] = line.value
        except:
            print('Cannot find Sequence of Ultrasound Regions Attribute for:', in_path)
            continue

        # Determine the color encoding used in the DICOM file. This is usually "YBR_FULL_422", but is sometimes "RGB". These two cases are handled, other cases are not.
        photometric_interpretation = dicom[(0x0028,0x0004)].value
        if photometric_interpretation not in ['YBR_FULL_422', 'RGB']:
            print('Warning: Image has unhandled PhotometricInterpretation. Pretending the value is "YBR_FULL_422". Color conversion may be incorrect. Inspect output images.')
        
        # Get frame rate. Need a value to start writing the video. Note that frame rate is irrelevant in the EchoNet model. 
        try:
            fps_in = dicom[(0x18, 0x40)].value
            fps_out = fps_in
        except:
            fps_in = None
            fps_out = 30
            print("Warning: Could not find frame rate in DICOM header, defaulting to 30.")
        
        # Set out path.
        os.makedirs(image_out_dir, exist_ok=True)
        if in_path.lower().endswith('.dcm'):
            out_name = os.path.basename(in_path)[:-4] + '.avi'
        else:
            out_name = os.path.basename(in_path) + '.avi'
        out_path = os.path.join(image_out_dir, out_name)
        if os.path.exists(out_path):
            raise Exception(f'Out path already exists: {out_path}')

        # Start up the video writer.
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out_shape = tuple(out_shape)
        out = cv2.VideoWriter(out_path, fourcc, fps_out, out_shape)

        # Cut off rows and columns at the edges such that the number of rows matches the number of columns. 
        if not skip_match_hw:
            bias = int(np.abs(dicom_array.shape[2] - dicom_array.shape[1])/2)
            if bias > 0:
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
                ## Note: It looks all of our 4C and LVOT scans have PhotometricInterpretation = YBR_FULL_422. Most of the VTI scans have YBR_FULL_422.

                ## DICOM Documentation about YBR_FULL:
                # Pixel data represent a color image described by one luminance (Y) and two chrominance planes (CB and CR). This photometric interpretation may be used only when Samples per Pixel (0028,0002) has a value of 3. May be used for pixel data in a Native (uncompressed) or Encapsulated (compressed) format; see Section 8.2 in PS3.5 . Planar Configuration (0028,0006) may be 0 or 1.

                # Black is represented by Y equal to zero. The absence of color is represented by both CB and CR values equal to half full scale.

                ## DICOM Documentation about YBR_FULL_422:
                # The same as YBR_FULL except that the CB and CR values are sampled horizontally at half the Y rate and as a result there are half as many CB and CR values as Y values

                # This explains why keeping only the first color channel results in a sensible black and white image for (as far as I know) all of our scans.

                # Convert the pixel array from YBR_FULL_422 to RGB.
                if photometric_interpretation == 'YBR_FULL_422':
                    im_rgb = Image.fromarray(frame, mode='YCbCr')
                    im_rgb = im_rgb.convert('RGB')
                elif photometric_interpretation == 'RGB':
                    im_rgb = Image.fromarray(frame, mode='RGB')
                    im_rgb = im_rgb.convert('RGB')
                else: # Treat the same as YBR_FULL_422
                    im_rgb = Image.fromarray(frame, mode='YCbCr')
                    im_rgb = im_rgb.convert('RGB')
                
                frame = np.array(im_rgb)
                
                output = frame
            else: # map the first color channel in the input to all three channels in the output video.
                channel_zero = frame[:, :, 0]
                output = cv2.merge([channel_zero, channel_zero, channel_zero])

            # Add current frame to the output
            out.write(output)

        # Close the video writer.
        out.release()

        ## Record file information to a spreadsheet. Read information from the DICOM header.
        if not file_list_out_path is None:
            # Set subject name based on input file name.
            subject = '.'.join(os.path.basename(in_path).split('.')[:-1])

            ## Record information.
            # File paths.
            df.loc[subject, 'FilePath'] = os.path.abspath(out_path)
            df.loc[subject, 'DicomFilePath'] = os.path.abspath(in_path)
            
            # Size information
            df.loc[subject, 'height_original'] = shape_original[1] # many frames
            df.loc[subject, 'width_original'] = shape_original[2]
            df.loc[subject, 'height_match_hw'] = shape_match_hw[1] # many frames
            df.loc[subject, 'width_match_hw'] = shape_match_hw[2]
            df.loc[subject, 'height_border_trim'] = shape_border_trim[0] # 1 frame
            df.loc[subject, 'width_border_trim'] = shape_border_trim[1]
            df.loc[subject, 'height_resize'] = shape_resize[0] # 1 frame
            df.loc[subject, 'width_resize'] = shape_resize[1]

            df.loc[subject, 'rescale_height'] = shape_border_trim[0]/shape_resize[0] # the factor by which the phyiscal width of a pixel increased
            df.loc[subject, 'rescale_width'] = shape_border_trim[1]/shape_resize[1]
            
            df.loc[subject, 'FrameHeight'] = shape_resize[0] # Record this shape again for consistency with the original EchoNet and my modification of the script.
            df.loc[subject, 'FrameWidth'] = shape_resize[1]

            df.loc[subject, 'NumberOfFrames'] = shape_original[0]

            # Convert LVOT diameter (or other length fields) to pixels in the output image, according to the following formula:
            # LVOT diameter (cm) = (# pixels) * rescale_height * PhysicalDeltaX
            #                    = (# pixels) * pixel_scale_factor
            # In other words, pixel_scale_factor is the number of cm per pixel.
            # Check that the "centimeters per pixel" is the same in the x and y directions.
            cm_per_pixel_x = df.loc[subject, 'rescale_height'] * df.loc[subject, 'PhysicalDeltaX']
            cm_per_pixel_y = df.loc[subject, 'rescale_width'] * df.loc[subject, 'PhysicalDeltaY']
            if (cm_per_pixel_x != cm_per_pixel_y):
                msg = 'The physical size of a pixel is not square. This might have been caused by anisotropic rescaling of the pixels. The physial size of a pixel must be square in order for lengths in the processed image to be well-defined in terms of pixels. In other words, the phrase "LVOT diameter = 2.2 pixels" must be meaningful regardless of the direction along which the measurement is performed.'
                raise Exception(msg)
            df.loc[subject, 'pixel_scale_factor'] = cm_per_pixel_x
            if (not names_of_length_fields_to_convert_to_pixels is None):
                for name_of_length_field_to_convert_to_pixels in names_of_length_fields_to_convert_to_pixels:
                    name_of_length_field_in_pixels = name_of_length_field_to_convert_to_pixels + '_px'
                    df.loc[subject, name_of_length_field_in_pixels] = df.loc[subject, name_of_length_field_to_convert_to_pixels] / cm_per_pixel_x

            # FPS
            df.loc[subject, 'FPS_dicom'] = fps_in
            df.loc[subject, 'FPS_avi'] = fps_out

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
                    df.loc[subject, column] = dicom[tag].value
                except IndexError:
                    df.loc[subject, column] = None            

    # Save the dataframe to CSV.
    if not file_list_out_path is None:
        df.to_csv(file_list_out_path, index=True)
    return

if (__name__ == '__main__'):
    ## Create argument parser.
    description = """Convert DICOM files to AVI. Record the output video paths, information from the DICOM headers, and information related to the applied processing operations to a CSV file. """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Define positional arguments.
    
    ## Define optional arguments.
    # Input/output paths
    parser.add_argument('--dicom_in_dir',
                        type=str,
                        help='input directory containing DICOM video files. Input files should be named <subject>.dcm')
    parser.add_argument('--datasheet_path',
                        help='path to CSV file containing data about each input VTI file. Expected columns: Subject, split. May contain additional columns.')
    parser.add_argument('--image_out_dir',
                        type=str,
                        help='directory to save processed videos to')
    parser.add_argument('--file_list_out_path',
                        type=str,
                        help='path to save final processed file list to')
    parser.add_argument('--names_of_length_fields_to_convert_to_pixels', type=str, help='If the input datasheet includes ground truth LVOT diameter values (or other lengths) in units of centimetres, specify the names of those fields here. These values will be converted to units of pixels in the processed video. The model will be trained to output LVOT diameter predictions in units of pixels, and subsequently convert them to cm. This is necessary becuase the phyiscal length of a pixel varies between echos.', nargs='*')

    # Image processing options.
    parser.add_argument("-m", "--skip_mask", help="In the original EchoNet processing a mask is applied to 'mask pixels outside the scanning sector'. This action seems to strip away the round edge of the scan, leaving it in a diamond shape, unlike the processed EchoNet scans. Use this option to skip applying the masking step.", action="store_true")
    parser.add_argument("-f", "--flip", help="In the original EchoNet processing, no 180 degree flip is applied, but our scans processed through that end up rotated 180 degrees compared to the EchoNet data. Use this flag to apply 180 degree rotation to the final image to match the processed EchoNet data.", action="store_true")
    parser.add_argument("-c", "--retain_color_channels", action="store_true", help="In the original EchoNet processing, the first color channel of the input DICOM is mapped to all three color channels of the output AVI. However, the processed EchoNet videos sometimes have multiple color channels, so they were clearly not processed this way. Use this flag to retain all three color channels.")

    parser.add_argument("--skip_match_hw", action="store_true", help="In the original EchoNet processing, the height and width (H, W) are matched by cropping the center of the video in the larger dimension (i.e if H>W, crop the image to the slice [H/2-W/2:H/2+W/2, :]; if H<W, crop the image to the slice [:, W/2-H/2:W/2+H/2]). Use this flag to skip this cropping step.")
    parser.add_argument("--border_trim_size", type=float, help="In the original EchoNet processing, after the height and width are matched, the image is cropped by removing a border of pixels of width 0.1 times the image width/height from all four sides. Use this flag to change the size of the border that is removed to something other than 0.1.", default=0.1)
    parser.add_argument("--out_shape", type=int, nargs=2, help="In the original EchoNet processing, after matching the height and width, and removing a border, the images are rescaled to a common size (112, 112). Use this flag to change that size. If predicting a length from the input image (e.g. LVOT diameter), ensure that any rescaling done to the image is isotropic, such that a 'length' in units of pixels can be unambiguously converted to a length in centimetres.", default=(112, 112))    

    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
