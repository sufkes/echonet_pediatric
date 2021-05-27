#!/usr/bin/env python

from ConvertDICOMToAVI import makeVideo
from glob import glob
import os


#in_file_glob_string = 'data_from_onedrive_4c_files_only/*.dcm'
#out_dir = 'data_processed/4c'

in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/data_from_onedrive-20210118_lvot_files_only/*.dcm'
out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/lvot_n261_redo'

in_files = sorted(glob(in_file_glob_string))
for in_file in in_files:
    try:
        makeVideo(in_file, out_dir)
    except:
        print('Failed to convert file:', in_file)
