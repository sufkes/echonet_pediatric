#!/usr/bin/env python

from ConvertDICOMToAVI import makeVideo
from glob import glob
import os

in_files = sorted(glob('data_from_onedrive_4c_files_only/*.dcm'))
for in_file in in_files:
    try:
        makeVideo(in_file, 'data_processed/4c')
    except:
        print('Failed to convert file:', in_file)
