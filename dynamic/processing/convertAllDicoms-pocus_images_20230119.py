#!/usr/bin/env python

from ConvertDICOMToAVI_decrease_frames import makeVideo
from glob import glob
import os


#in_file_glob_string = 'data_from_onedrive_4c_files_only/*.dcm'
#out_dir = 'data_processed/4c'


## Previously used command.
#in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/data_from_onedrive-20210118_lvot_files_only/*.dcm'
#out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/lvot_n261_redo'


## Test for neonatal data.
# 4C
#in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/neonates_4c/*.dcm'
#out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/neonates_4c'
# LVOT
#in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/test_neonates/lvot/*.dcm'
#out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/neonates_lvot'

## Test for new data July 30
# 4C
#in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/test_new_data_july_30/4c/raw/*.dcm'
#out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/test_new_data_july_30/4c/processed'

# LVOT
#in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/test_new_data_july_30/lvot/raw/*.dcm'
#out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/test_new_data_july_30/lvot/processed'


## Neonates 4C
in_file_glob_string = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/neonates_20220329/4c/known/*.dcm'
out_dir = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/echo_data/pocus-all_multiframe_images-decrease_framerate'


#in_files = sorted(glob(in_file_glob_string))
dicom_file_list='/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/raw/pocus-20220916/list_of_dicoms_with_multiple_frames.txt'
with open(dicom_file_list, 'r') as handle:
    in_files = handle.readlines()
in_files = [f.rstrip('\n') for f in in_files]

for in_file in in_files:
    try:
        makeVideo(in_file, out_dir)
    except:
        print('Failed to convert file:', in_file)
