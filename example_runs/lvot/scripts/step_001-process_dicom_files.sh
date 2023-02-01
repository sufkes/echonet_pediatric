#!/bin/bash

/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic/processing/processEchoVideo.py \
    --dicom_in_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvot/dicom \
    --datasheet_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvot/datasheets/start_data.csv \
    --image_out_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvot/avi \
    --file_list_out_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvot/datasheets/file_list.csv \
    --names_of_length_fields_to_convert_to_pixels LVOT \
    --skip_mask \
    --flip \
    --retain_color_channels

