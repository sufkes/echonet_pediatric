#!/usr/bin/env python
# coding: utf-8

# In[1]:


# David Ouyang 10/2/2019

# Notebook which iterates through a folder, including subfolders, 
# and convert DICOM files to AVI files of a defined size (natively 112 x 112)

import re
import os, os.path
from os.path import splitext
import pydicom as dicom
import numpy as np
from pydicom.uid import UID, generate_uid
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
#from scipy.misc import imread
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math

destinationFolder = '/hpf/largeprojects/ccmbio/sufkes/echonet/data/data_processed/4c'
#destinationFolder = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/lvot_n261'

# In[10]:


# Dependencies you might need to run code
# Commonly missing

#!pip install pydicom
#!pip install opencv-python

# In[2]:

def mask(output):
    dimension = output.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    
    mask = ((m1+m2)>int(dimension/2) + int(dimension/10)) 
    mask *=  ((m1-m2)<int(dimension/2) + int(dimension/10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)
    
    #print(maskedImage.shape)
    
    return maskedImage


# In[3]:


def makeVideo(fileToProcess, destinationFolder):
    # SU: removed try-except block.
#    try:
        #fileName = fileToProcess.split('\\')[-1] #\\ if windows, / if on mac or sherlock
                                                 #hex(abs(hash(fileToProcess.split('/')[-1]))).upper()

    fileName = os.path.basename(fileToProcess) # SU lines above seem to be looking for the filename.
    print(fileName) # SU

    if not os.path.isdir(os.path.join(destinationFolder,fileName)):
        dataset = dicom.dcmread(fileToProcess, force=True)
        testarray = dataset.pixel_array

        #print(dataset)
        print('Dimensions (original):', testarray.shape) 
            
        frame0 = testarray[0] # SU: first time point; e.g. shape = (708, 1016, 3)
        mean = np.mean(frame0, axis=1) # SU: e.g. shape = (708, 3)
        mean = np.mean(mean, axis=1) # SU: e.g. shape = (708,)

        # SU: next line causes an IndexError because mean>1 everywhere. Not sure what this is attempting to do. Try skipping the crop. Maybe it's looking for the *first* value y index that has mean<1?

        #try: # SU 
        #    yCrop = np.where(mean<1)[0][0] # original line
        #except IndexError: # SU
        #    yCrop = 0 # SU
        #testarray = testarray[:, yCrop:, :, :]
        ## SU: comment out above section since it appears possible to alter the scaling from scan to scan.


        # SU: looks like this section cuts off rows and columns at the edges such that the number of rows matches the number of columns.
        bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
        if bias>0:
            if testarray.shape[1] < testarray.shape[2]:
                testarray = testarray[:, :, bias:-bias, :]
            else:
                testarray = testarray[:, bias:-bias, :, :]


        print('Dimensions (after matching x and y sizes):', testarray.shape)
        frames,height,width,channels = testarray.shape

        fps = 30

        try:
            fps = dataset[(0x18, 0x40)].value
        except:
            print("couldn't find frame rate, default to 30")

        #print("FPS:", fps) # SU
            
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        os.makedirs(destinationFolder, exist_ok=True)
        video_filename = os.path.join(destinationFolder, fileName.rstrip('.dcm').rstrip('.DCM') + '.avi')

        print("video_filename:", video_filename)
        out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)

        for i in range(frames):
            outputA = testarray[i,:,:,0]
            smallOutput = outputA[int(height/10):(height - int(height/10)), int(height/10):(height - int(height/10))]

            if i == 0:
                print('Dimensions (after trimming 10% of each side):', smallOutput.shape)
                print('Pixel size (x) scaled up by a factor of:', smallOutput.shape[0]/cropSize[0])
                print('Pixel size (y) scaled up by a factor of:', smallOutput.shape[1]/cropSize[1])
                print('')
                print('These factors should be recorded for each image in the FileList.csv file for the LVOT (and 4C) data. Currently, I am not recording this, and the raw LVOT model technically predicts LVOT diameter in units of original image pixels (i.e. in units of the processed images multiplied by the factor just reported). , which is then converted to cm using the recorded DICOM tags PhysicalDeltaX and PhysicalDeltaY. This sacling factor has been the same value for all LVOT images (5.071428571428571), but should be recorded in case it ever changes, and the LVOT model should predict in units of pixels of the input image for the sake of clarity.')
                print('')
                
            # Resize image
            output = cv2.resize(smallOutput, cropSize, interpolation = cv2.INTER_CUBIC)

            finaloutput = mask(output)

            finaloutput = cv2.merge([finaloutput,finaloutput,finaloutput])
            out.write(finaloutput)

        out.release()

    else:
        print(fileName,"hasAlreadyBeenProcessed")
#    except:
#        print("something filed, not sure what, have to debug", fileName)
#    return 0


# In[ ]:


AllA4cNames = '/hpf/largeprojects/ccmbio/sufkes/echonet/data/4c_files_only'
#AllA4cNames = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/data_from_onedrive-20210118_lvot_files_only'

count = 0
    
cropSize = (112,112)
#subfolders = os.listdir(AllA4cNames) # Steve: commented out to avoid error if AllA4cNames directory doesn't exist.


#for folder in subfolders:
#    print(folder)

#    for content in os.listdir(os.path.join(AllA4cNames, folder)):
#        for subcontent in os.listdir(os.path.join(AllA4cNames, folder, content)):
#            count += 1
            

#            VideoPath = os.path.join(AllA4cNames, folder, content, subcontent)

#            print(count, folder, content, subcontent)

#            if not os.path.exists(os.path.join(destinationFolder,subcontent + ".avi")):
#                makeVideo(VideoPath, destinationFolder)
#            else:
#                print("Already did this file", VideoPath)


#print(len(AllA4cFilenames))


# In[ ]:




