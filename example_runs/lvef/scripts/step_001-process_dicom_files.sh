#!/bin/bash

/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/dynamic/processing/processEchoVideo.py \
    --dicom_in_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvef/dicom \
    --datasheet_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvef/datasheets/start_data.csv \
    --image_out_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvef/avi \
    --file_list_out_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/lvef/datasheets/file_list.csv \
    --skip_mask \
    --flip \
    --retain_color_channels

