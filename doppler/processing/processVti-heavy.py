#!/usr/bin/env python3

import os
import sys
import argparse
import warnings
warnings.simplefilter('once')
import pandas as pd
import pydicom
import numpy as np
np.seterr(all='ignore') # Ignore division by zero warning here. It works properly given tne b > 0 condition.
import cv2
from glob import glob
from PIL import Image
from skimage.color import rgb2gray # I should remove this dependency and just convert using PIL
#from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sort_id_key_vectorized(subject_series, subject_prefix='VTI'):
    """Special subject ID sorter"""
    def sort_id_key(subject):
        if 'DCM' in subject:
            val = 10000 + int(subject.split('M')[-1])
        else:
            val = int(subject.split(subject_prefix[-1])[-1])
        return val
    try:
        sort_values = [sort_id_key(subject) for subject in subject_series] # list of (unsorted) numbers whose values will determine ordering of the rows.
    except:
        warnings.warn('Failed to sort IDs')
        sort_values = list(subject_series)
    return sort_values

def readAnnotation(annotation_path):
    '''Read the vertical axis values of the modal-velocity curve. Horizontal positions with no peak marker will be assumed to not lie within the peak region, and will be assigned vertical positions 0. 
Parameters: 
    annotation_path (str): path to input image with peak marked in red
Returns: 
    peak (numpy.array): vector of length equal to horizontal length of input image, containing the vertical position of the modal velocity curve; down is treated as the positive direction.'''

    annotation = Image.open(annotation_path)
    pixel_data = np.array(annotation) # image is RGB-encoded, but all pixels will have R==G==B, except the line marking the peak.
            
    r = pixel_data[..., 0] # red component
    g = pixel_data[..., 1] # green
    b = pixel_data[..., 2] # blue
        
    redmap = (r>g) & (r>b) # booleans

    # There must be a slick way to find, for each column, the highest index of the nonzero pixels, but I couldn't find one, and this works fine.
    peak = np.zeros(pixel_data.shape[1])
    for x in range(redmap.shape[1]):
        maximum = 0
        for y in range(redmap.shape[0]):
            if redmap[y,x]:
                maximum = y
        peak[x] = maximum
    
    return peak

def getGroundTruthVti(annotation):
    """Compute the VTI corresponding to each annotated peak, and return the average of the VTIs across the annotated peaks. The result is VTI in units of pixels."""
    vti_list = []
    vti = 0
    for ii, val in enumerate(annotation):
        vti += val
        if (vti > 0) and ((val == 0) or (ii == len(annotation)-1)): # if at the end of an annotated peak
            vti_list.append(vti)
            vti = 0
    vti_px = np.mean(np.array(vti_list))
    return vti_px

def removeGreenLine(pixel_data_ycbcr):
    """Attempt to remove the green heartbeat line from the v-t plot. Input the cropped v-t plot pixel array."""
    # Pixel_data is currently in YCbCr format. Get the mask from RGB space.
    im_rgb = Image.fromarray(pixel_data_ycbcr, mode='YCbCr')
    im_rgb = im_rgb.convert('RGB')
    pixel_data_rgb = np.array(im_rgb)
    
    r = pixel_data_rgb[:,:,0].astype(float)
    g = pixel_data_rgb[:,:,1].astype(float)
    b = pixel_data_rgb[:,:,2].astype(float)
    rb_ratio_min = 0.0
    rb_ratio_max = 0.7
    gb_ratio_min = 1.0
    gb_ratio_max = 1.4
    rgb_sum_min  = 50
    green_mask = (b > 0) & (r/b >= rb_ratio_min) & (r/b < rb_ratio_max) & (g/b > gb_ratio_min) & (g/b < gb_ratio_max) & (r+g+b > rgb_sum_min)
    pixel_data_ycbcr[green_mask] = 0 # black out masked pixels
    return pixel_data_ycbcr

def getPeaks(pixel_data_ycbcr, annotation):
    """Create separate images for each main peak in the v-t spectrum
Parameters:
    pixel_data_ycbcr: array of shape HxWxC with raw pixel data from DICOM file in YCbCr format
Returns:
    peak_list_ycbcr: list of YCbCr pixel data arrays, one for each peak
    annotation_list: list of 1D annotation vector segments, one for each peak"""

    ## Identify peaks.
    bands = getPeakBands(pixel_data_ycbcr)

    ## Generate a list of peak images.
    peak_list_ycbcr = []
    annotation_list = [] # 1D arrays of modal velocity values
    
    for label in range(1, bands.max()+1):
        if not label in bands:
            continue # skip if this peak was deleted

        indices = (bands==label)
        peak_image = pixel_data_ycbcr[:, indices]
        if not annotation is None:
            annotation_segment = annotation[indices]
        else:
            annotation_segment = None
            
        peak_list_ycbcr.append(peak_image)
        annotation_list.append(annotation_segment)

    return peak_list_ycbcr, annotation_list
    

def getPeakBands(pixel_data_ycbcr):
    """Identify the primary VTI peaks.
Parameters:
    pixel_data_ycbcr: array of shape HxWxC with raw pixel data from DICOM file in YCbCr format
Returns:
    bands: array of shape w containing integer labels corresponding to each primary peak """
    
    # Pixel_data is currently in YCbCr format. Get the mask from RGB space.
    im_rgb = Image.fromarray(pixel_data_ycbcr, mode='YCbCr')
    im_rgb = im_rgb.convert('RGB')
    pixel_data_rgb = np.array(im_rgb)
    
    # Convert to greyscale.
    image_grey = rgb2gray(pixel_data_rgb)

    ## Find peaks to measure VTI from.
    # Select a thick horizontal strip in the middle of the image.
    mid_top = int(0.25*image_grey.shape[0])
    mid_bottom = int(0.66*image_grey.shape[0])
    image_mid = image_grey[mid_top:mid_bottom, :]

    # Mark positions in the strip where the vertical slice has above average intensity (1 if above; else 0)
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

            
    # Extend the peaks to the left and right so that they include the whole peak region. Assume that the extended bounds will never overlap.
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
            num_in_blacked_out_region = np.count_nonzero((image_grey[4:int(0.5*image_grey.shape[0]), left_bound:ii] == 0).all(axis=0)) # number of indices which would be in a blacked out area
            num_invalid = num_out_of_bounds + num_in_blacked_out_region
            percent_invalid = num_invalid/left_extend*100
            if num_invalid > left_extend*left_ratio_threshold: # if most of the extension is invalid, mark this band for deletion; it is probably an incomplete peak.
                bad_label = bands_copy[ii]
                bad_labels.append(bad_label)
                #print(f'Found bad peak during leftward extension in peak #{bad_label} for image {os.path.basename(in_path)} \n\t{num_out_of_bounds}px out-of-image \n\t{num_in_blacked_out_region}px in blacked-out region \n\t{left_extend}px extension \n\t{percent_invalid:.2f}% out of bounds')
                
            # Check that extension will not overlap with another band.
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
            num_in_blacked_out_region = np.count_nonzero((image_grey[4:int(0.5*image_grey.shape[0]), ii:right_bound] == 0).all(axis=0))
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
        print(f'Warning: All peaks identified as bad. Keeping all peaks.')
    return bands


def main(dicom_in_dir, patient_data_path, split_path, image_out_dir, file_list_out_path, out_color_mode, out_image_type, new_height, new_width, pad, rescale, remove_green_line, split_peaks, save_copy_for_annotation, read_annotations, annotations_in_dir, num_annotated_peaks_path):
    #### Generate a dataframe storing data with one row for each patient.
    patient_df = pd.read_csv(patient_data_path)
    split_df = pd.read_csv(split_path)
    
    if read_annotations:
        split_col = 'split_VTI_annotation'
    else:
        split_col = 'split_VTI' 
    split_df = split_df[['id_VTI', split_col]]

    # Drop rows lacking a split value.
    split_df.drop(split_df[split_df[split_col].isna()].index, inplace=True)

    # Rename the split column.
    split_df.rename(columns={split_col:'split_all_random'}, inplace=True)
    
    # Combine dataframes into one. patient_df contains VTI values from Mael; there are subjects with annotations by Cameron which do not have corresponding VTI values from Mael.
    #base_df = patient_df.merge(split_df, left_on='Subject', right_on='id_VTI', how='right') # right merge to drop subjects without patient data or train/val/test assignment.
    base_df = split_df.merge(patient_df, right_on='Subject', left_on='id_VTI', how='left') # left merge to drop subjects without patient data or train/val/test assignment.
    
    # Drop columns.
    base_df.drop(columns='Subject', inplace=True) # Drop 'Subject' column taken from the original patient_df, which may be missing rows for some patients.

    # Rename the subject column taken from split_df, which should have a row for all included patients.
    base_df.rename(columns={'id_VTI':'Subject'}, inplace=True)

    # Add the number of peaks annotated by Cameron if reading annotations.
    if read_annotations:
        num_peaks_df = pd.read_csv(num_annotated_peaks_path)
        base_df = base_df.merge(num_peaks_df, on='Subject', how='left')
        #base_df.drop(columns='Subject', inplace=True)

    # Set index to the subject column.
    base_df.set_index('Subject', drop=True, inplace=True) # one row per subject

    #### Process DICOM files.
    # Find the DICOM files.
    dicom_in_paths = glob(dicom_in_dir + '/*')
    dicom_in_paths.sort()

    # Find the annotations files.
    if read_annotations:
        annotations_in_paths = glob(annotations_in_dir + '/*')
        annotations_in_paths.sort()
    
    # Store all pixels in the training set; use to determine mean and standard deviation of training set later.
    train_pixels_all = []
    train_set_size = 0

    # Keep track of the number of peaks extracted, and the number of peaks annotated.
    if read_annotations and split_peaks:
        total_peaks_extracted = 0
        total_peaks_annotated = 0
    
    ## Loop over the DICOM files.
    first_subject = True # If this is the first subject in the list.
    for ii, in_path in enumerate(dicom_in_paths):
        if not in_path.lower().endswith('.dcm'):
            continue
        
        # Add DICOM path to dataframe.
        subject = os.path.basename(in_path).split('.')[0]
        #print(base_df)

        # Load the corresponding annotation file.
        if read_annotations:
            annotation_name = subject + '.png'
            annotation_path = os.path.join(annotations_in_dir, annotation_name)
            if not os.path.isfile(annotation_path):
                continue
            else:
                annotation = readAnnotation(annotation_path) # 1D numpy array with length equal to the width of the VTI plot.
        else:
            annotation = None

        base_df.loc[subject, 'DicomFilePath'] = os.path.abspath(in_path)
        if read_annotations:
            base_df.loc[subject, 'AnnotationFilePath'] = os.path.abspath(annotation_path)

        ## Load the image
        dicom = pydicom.dcmread(in_path)

        # Determine the color encoding used in the DICOM file. This is usually "YBR_FULL_422", but is sometimes "RGB". These two cases are handled, other cases are not.
        photometric_interpretation = dicom[(0x0028,0x0004)].value # 2023-01-25 SU
        if photometric_interpretation == 'YBR_FULL_422':
            pixel_data_ycbcr = dicom.pixel_array
        elif photometric_interpretation == 'RGB':
            pixel_data_rgb = dicom.pixel_array
            im_rgb = Image.fromarray(pixel_data_rgb, mode='RGB')
            im_ycbcr = im_rgb.convert('YCbCr')
            pixel_data_ycbcr = np.array(im_ycbcr)
        else:
            print('Warning: Image has unhandled PhotometricInterpretation. Pretending the value is "YBR_FULL_422". Color conversion may be incorrect. Inspect output images.')
            pixel_data_ycbcr = dicom.pixel_array

        
        # Get information about the V-T plot from DICOM header, including the physical meaning of units along the x and y axes. 
        try:
            vt_seq = dicom[(0x0018,0x6011)][1] # DICOM sequence tag for the VT plot (hopefully).
            for line in dicom[(0x0018,0x6011)][1]:
                base_df.loc[subject, line.keyword] = line.value
        except:
            print('Cannot find information about V-T sequence for:', subject)
            continue

        ## Crop the V-T graph. 
        # Most images have the V-T graph within bounds (50, 913, 212, 671). It looks like some are a smaller area within this box, but none appear to extend beyond it.
        # The ReferencePixelY0 DICOM tag indicates the pixel where the horizontal axis lies. We can cut off everything above this.
        # Cropping does not affect scaling.
        # For some reason, I had the values hard-coded instead of being read from the files. I will read from the headers instead.
        left = vt_seq[(0x0018, 0x6018)].value # Region Location Min X0
        right = vt_seq[(0x0018, 0x601c)].value # Region Location Max X1
        if right != 913:
            warnings.warn('RegionLocationMaxX1 has value other than 913. This may be because part of the VT plot is blacked out. Assuming the right edge of the VT plot occurs at column 913, and ignoring the value of RegionLocationMaxX1.')
            right = 913
            pass
        top = vt_seq[(0x0018, 0x601a)].value # Region Location Min Y0
        top += vt_seq[(0x0018, 0x6022)].value # Reference Pixel Y0
        bottom = vt_seq[(0x0018, 0x601e)].value # Region Location Max Y1
        #left = 50
        #right = 913
        #top = 212 # top of the V-T plot.
        #top += int(base_df.loc[subject, 'ReferencePixelY0']) # horizontal axis of V-T plot
        #bottom = 671
        pixel_data_ycbcr = pixel_data_ycbcr[top:bottom+1, left:right+1, :]

        # 2023-01-25 SU: We presume that the script has been run previously with --save_copy_for_annotation, the resulting PNG files have been annotated, and we are now rerunning with --read_annotations. We want to check that the size of the annotated PNG file is the same as the size of the extracted v-t plot.
        if read_annotations:
            if pixel_data_ycbcr.shape[1] != len(annotation):
                raise Exception('Width of annotation file does not match width of VTI plot extracted from Doppler image.')

        # Store the positions of the v-t plot boundaries
        base_df.loc[subject, 'left'] = left
        base_df.loc[subject, 'right'] = right
        base_df.loc[subject, 'top'] = top
        base_df.loc[subject, 'bottom'] = bottom

        ## Blacken the green-blue line at the bottom of the V-T plot using color ratios.
        if remove_green_line:
            pixel_data_ycbcr = removeGreenLine(pixel_data_ycbcr)

        ## Get the ground truth VTI value from the annotated image. This is the mean of the VTI's for each annotated peak, regardless of whether the peaks are identified by the getPeaks method later.
        if read_annotations:
            vti_ground_truth_px = getGroundTruthVti(annotation)
            vti_ground_truth = base_df.loc[subject, 'PhysicalDeltaX'] * base_df.loc[subject, 'PhysicalDeltaY'] * vti_ground_truth_px # ground truth VTI in cm.
            base_df.loc[subject, 'AOVTI_annotation_px_original'] = vti_ground_truth_px
            base_df.loc[subject, 'AOVTI_annotation'] = vti_ground_truth
            
        ## Generate a list of preprocessed sub-images if splitting image into pieces.
        if split_peaks:
            image_list, annotation_list = getPeaks(pixel_data_ycbcr, annotation)
        else:
            image_list = [pixel_data_ycbcr] # if not splitting the peaks, then image_list is a list of one image: the complete v-t plot.
            annotation_list = [annotation]


        #### Continue processing and save each sub-image in the list.
        ## If this is the first subject, create a new dataframe to store one row per sub-image, derived from the dataframe storing one row per subject.
        if first_subject:
            df = pd.DataFrame(columns=base_df.columns) # one row per sub-image
            df.index.name = 'Subject'
            first_subject = False

        if read_annotations:
            num_peaks_annotated = 0 # keep track of the number of extracted peaks which were annotated by Cameron.
        for image_num, (pixel_data_ycbcr, annotation) in enumerate(zip(image_list, annotation_list), 1):
            ## Add row in new dataframe.
            if split_peaks:
                subject_new = subject+'_'+str(image_num)
            else:
                subject_new = subject

            ## Copy base row; replace subject name 
            df.loc[subject_new, :] = base_df.loc[subject]
            df.loc[subject_new, 'Subject_base'] = subject

            
            #### Resize the sub-image (rescale or pad) if requested.
            ## Store original shape of the sub-image.
            old_height = pixel_data_ycbcr.shape[0]
            old_width = pixel_data_ycbcr.shape[1]
            df.loc[subject_new, 'old_height'] = old_height
            df.loc[subject_new, 'old_width'] = old_width
            
            ## Pad
            if pad:
                ## Record the pixel rescaling.
                rescale_y = 1
                rescale_x = 1
                
                if (new_height < pixel_data_ycbcr.shape[0]) or (new_width < pixel_data_ycbcr.shape[1]):
                    raise Exception('Requested padding but (new_height, new_width) is set smaller than the current image size.')
                new = np.zeros((new_height, new_width, 3), dtype=pixel_data_ycbcr.dtype)
                new[:pixel_data_ycbcr.shape[0], :pixel_data_ycbcr.shape[1], :] = pixel_data_ycbcr
                pixel_data_ycbcr = new

                if read_annotations:
                    new_annotation = np.zeros((new_width,), dtype=annotation.dtype)
                    new_annotation[:annotation.shape[0]] = annotation
                    annotation = new_annotation
                
            ## Rescale
            elif rescale:
                ## Record the pixel rescaling factor.
                rescale_y = pixel_data_ycbcr.shape[0]/new_height
                rescale_x = pixel_data_ycbcr.shape[1]/new_width
                
                pixel_data_ycbcr = cv2.resize(pixel_data_ycbcr, (new_width, new_height), interpolation=cv2.INTER_LINEAR) # not sure what interpolation would be best here

                # Rescale the annotation.
                if read_annotations: # 2023-01-25 SU
                    x_range_old = np.arange(old_width, dtype=float)
                    x_range_new = np.arange(new_width, dtype=float) * old_width/new_width
                    annotation = np.interp(x_range_new, x_range_old, annotation)
                    annotation = new_height/old_height*annotation
                
            else:
                rescale_y = 1
                rescale_x = 1

            df.loc[subject_new, 'new_height'] = pixel_data_ycbcr.shape[0]
            df.loc[subject_new, 'new_width'] = pixel_data_ycbcr.shape[1]
        
            df.loc[subject_new, 'rescale_y'] = rescale_y
            df.loc[subject_new, 'rescale_x'] = rescale_x
            
            ## Save the preprocessed image and associated data.
            im = Image.fromarray(pixel_data_ycbcr, mode='YCbCr')
            im = im.convert(out_color_mode) # convert to greyscale or RGB
            if save_copy_for_annotation:
                im_annotated = Image.fromarray(pixel_data_ycbcr, mode='YCbCr')
                # Convert pixels to greyscale, then back to RGB, so that the output image is greyscale, but we can annotate in red.
                im_annotated = im.convert('L')
                im_annotated = im.convert('RGB')
                image_out_dir_annotated = image_out_dir + '-to_annotate'

            # Set file paths and save.
            if split_peaks:
                out_name = '.'.join(os.path.basename(in_path).split('.')[:-1]) + '_' + str(image_num) + '.' + out_image_type # <old name>.png
            else:
                out_name = '.'.join(os.path.basename(in_path).split('.')[:-1]) + '.' + out_image_type # <old name>.png
            out_path = os.path.join(image_out_dir, out_name)
            os.makedirs(image_out_dir, exist_ok=True)
            im.save(out_path)
            df.loc[subject_new, 'FilePath'] = os.path.abspath(out_path)

            if read_annotations:                
                # Save the modal velocity NumPy array
                peak_array_out_name = out_name.replace('.png', '.npy')
                peak_array_out_dir = image_out_dir+'-peak_arrays'
                peak_array_out_path = os.path.join(peak_array_out_dir, peak_array_out_name)
                os.makedirs(peak_array_out_dir, exist_ok=True)
                annotation = annotation.astype(np.float32)
                np.save(peak_array_out_path, annotation)
                df.loc[subject_new, 'peak_array_path'] = os.path.abspath(peak_array_out_path)

                # Save a copy of the image with the extracted modal velocity curve overlaid, as a sanity check.
                image_annotated_out_name = out_name
                image_annotated_out_dir = image_out_dir+'-extracted_annotations'
                image_annotated_out_path = os.path.join(image_annotated_out_dir, image_annotated_out_name)
                os.makedirs(image_annotated_out_dir, exist_ok=True)
                
                pixel_data = np.array(im)
                plt.figure()
                plt.axis('off')
                plt.imshow(pixel_data, cmap='gray')
                plt.plot(annotation, color='red', linewidth=1)
                plt.savefig(image_annotated_out_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            if save_copy_for_annotation:
                out_path_annotated = os.path.join(image_out_dir_annotated, out_name)
                os.makedirs(image_out_dir_annotated, exist_ok=True)
                im_annotated.save(out_path_annotated)
                df.loc[subject_new, 'FilePath_annotated'] = os.path.abspath(out_path_annotated)
            
            ## Get the final scaling factor for each row (could be different for each peak image if splitting peaks.
            # True VTI (cm) = (# processed pixels) * rescale_x * rescale_y * PhysicalDeltaX * PhysicalDeltaY
            #               = (# processed pixels) * pixel_scale_factor
            pixel_scale_factor = rescale_x * rescale_y * base_df.loc[subject, 'PhysicalDeltaX'] * base_df.loc[subject, 'PhysicalDeltaY']
            df.loc[subject_new, 'pixel_scale_factor'] = pixel_scale_factor
            
            if split_peaks and read_annotations:
                # Calculate VTI from tracing.
                vti_px_annotation = annotation.sum()
                df.loc[subject_new, 'AOVTI_px_annotation_extracted_peaks'] = vti_px_annotation # the number of pixels; this will be zero if the extracted peak was not annotated
                df.loc[subject_new, 'AOVTI_annotation_extracted_peaks'] = vti_px_annotation*pixel_scale_factor
                if vti_px_annotation > 0:
                    num_peaks_annotated += 1
            
            # Record pixels in the training set in order to obtain the mean and standard deviation. Should be done differently if using multiple color channels
            if (base_df.loc[subject, 'split_all_random'] == 'train'):
                if (not read_annotations) or (vti_px_annotation > 0):
                    pixels_flat = np.array(im).flatten()
                    train_pixels_all.extend(pixels_flat)
                    train_set_size += 1
            
        # Keep track of the number of peaks which were annotated by Cameron.
        if read_annotations and split_peaks:
            df.loc[df['Subject_base'] == subject, 'num_peaks_extracted'] = len(annotation_list)
            df.loc[df['Subject_base'] == subject, 'num_peaks_annotated_found'] = num_peaks_annotated
            df.loc[df['Subject_base'] == subject, 'num_peaks_annotated_missed'] = df.loc[subject_new, 'num_peaks_annotated_actual'] - num_peaks_annotated 

            total_peaks_extracted += len(annotation_list)
            total_peaks_annotated += num_peaks_annotated

    ## Calculate the VTI values in terms of number of pixels in the processed image.
    df['AOVTI_px'] = [vti/r for vti, r in zip(df['AOVTI'], df['pixel_scale_factor'])]# VTI in units of pixels, as measured by Mael. This is for the old values of VTI which were recorded directly, and not extracted from annotated images. 

    ## Sort the rows based on subject name
    df.sort_values('Subject', axis=0, key=sort_id_key_vectorized, inplace=True)

    ## Sort columns.
    columns_start = ['Subject_base']
    columns_end = ['num_peaks_annotated_found', 'num_peaks_annotated_actual', 'num_peaks_annotated_missed', 'num_peaks_extracted']
    columns_sorted = columns_start + [c for c in df.columns if (not c in columns_start) and (not c in columns_end)] + columns_end
    # Remove columns that don't actually appear in the dataframe.
    columns_sorted = [c for c in columns_sorted if c in df.columns]
    df = df[columns_sorted]

    ## Calculate the mean and standard deviation of the training set pixels.
    train_mean = np.array(train_pixels_all).mean()
    train_std = np.array(train_pixels_all).std()
    #print('Training set mean:', train_mean)
    #print('Training set std :', train_std)
    #print('Training set size :', train_set_size)
    df.loc[:, 'train_mean'] = train_mean
    df.loc[:, 'train_std'] = train_std
    
    ## Drop rows lacking a split value or pixel_scale_factor
    #df.drop(df[df['split_all_random'].isna()].index, inplace=True)
    df.drop(df[df['pixel_scale_factor'].isna()].index, inplace=True)
    
    if split_peaks and read_annotations:
        df_all_peaks = df.copy() # Create a copy of the dataframe which includes a row for every extracted peak, even those that were not annotated by the expert. 
        df.drop(df[df['AOVTI_px_annotation_extracted_peaks'] == 0].index, inplace=True) # For the "main" dataframe, remove rows corresponding to peaks that were not annotated by the expert.
    
    ## Calculate VTI as the mean across the peaks labelled by Cameron (including only those which are extracted by this script). # Not really useful; we want to also include the annotated peaks not extracted by the script, which is done earlier in the script.
#    subject_means = df.groupby('Subject_base')[['AOVTI_px_annotation_extracted_peaks', 'AOVTI_annotation_extracted_peaks']].mean()
#    for subject_base in df['Subject_base'].unique().tolist():
#        df.loc[df['Subject_base']==subject_base, 'AOVTI_px_annotation_extracted_peaks_mean'] = subject_means.loc[subject_base, 'AOVTI_px_annotation_extracted_peaks']
#        df.loc[df['Subject_base']==subject_base, 'AOVTI_annotation_extracted_peaks_mean'] = subject_means.loc[subject_base, 'AOVTI_annotation_extracted_peaks']

    ## Print number of peaks annotated.
    if split_peaks and read_annotations:
        print('Number of peaks extracted:', total_peaks_extracted)
        print('Number of peaks annotated:', total_peaks_annotated)
    
    ## Save the dataframe.
    df.to_csv(file_list_out_path, index=True)
    if split_peaks and read_annotations:
        df_all_out_path = file_list_out_path.replace('.csv', '-all_peaks.csv')
        df_all_peaks.to_csv(df_all_out_path, index=True)
    return

if __name__ == '__main__':
    # Create argument parser.
    description = """Extract velocity-time plot from VTI DICOM images, post-process, save to PNG, and generate a CSV containing data for each processed PNG to be used in the model."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Define positional arguments.
    
    ## Define optional arguments.
    # input/output paths
    parser.add_argument('--dicom_in_dir',
                        type=str,
                        help='input directory containing DICOM VTI files',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/data_from_onedrive-20210118_vti_files_only')
    parser.add_argument('--patient_data_path',
                        help='path to the spreadsheet containing VTI and other information for each patient, taken from COspreadsheet.',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/vti_master_raw_data_sheet_2021-04-21.csv')
    parser.add_argument('--split_path',
                        type=str,
                        help='path to split file containing a column "split_VTI"',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/split.csv')
    parser.add_argument('--image_out_dir',
                        type=str,
                        help='directory to save processed images to',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_camerons_annotations-split_peaks-417x286-rescale')
    parser.add_argument('--file_list_out_path',
                        type=str,
                        help='path to save final processed file list to',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList-vti_camerons_annotations-split_peaks-417x286-rescale.csv')
    
    # image processing options
    parser.add_argument('-m', '--out_color_mode',
                        type=str,
                        help='output color mode. Can be "RGB" or "L" (greyscale).',
                        default='L')
    parser.add_argument('--out_image_type',
                        type=str,
                        help='output image type',
                        default='png')
    parser.add_argument('-y', '--new_height',
                        type=int,
                        help='height of new image (pixels)',
                        default=417)
    parser.add_argument('-x', '--new_width',
                        type=int,
                        help='width of new image (pixels)',
                        default=864)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--pad',
                        action='store_true',
                        help='pad cropped images to size NEW_HEIGHT x NEW_WIDTH')
    group.add_argument('-r', '--rescale',
                        action='store_true',
                        help='rescale images to size NEW_HEIGHT x NEW_WIDTH')
    parser.add_argument('-g', '--remove_green_line',
                        action='store_true',
                        help='remove solid green-blue line at bottom of V-T plot')
    parser.add_argument('-s', '--split_peaks',
                        action='store_true',
                        help='remove solid green-blue line at bottom of V-T plot')

    # Annotation settings
    parser.add_argument('--save_copy_for_annotation',
                        help='save a second copy of the processed image to be annotated; add column in the file list to record the path of copy of the image to be annotated',
                        action='store_true')
    parser.add_argument('--read_annotations',
                        action='store_true',
                        help='read modal velocity annotations from images; annotated images must be in their original resolution, and contain only the VTI plot portion of the Doppler image as would be extracted by this script; annotations must be in red; images must be PNG; tracings are read from the bottom-most red pixel in a column of the image')
    parser.add_argument('--annotations_in_dir',
                        type=str,
                        help='directory containing VTI plots with modal velocity annotations drawn in red',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/vti_camerons_annotations/annotated')
    parser.add_argument('--num_annotated_peaks_path',
                        type=str,
                        help='path to CSV file containing the number of peaks annotated by Cameron in each VTI image',
                        default='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/num_peaks_annotated_by_cameron.csv')
                        
    # Parse arguments.
    args = parser.parse_args()

    # Run main function.
    main(**vars(args))
