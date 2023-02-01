#!/bin/bash

/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/doppler/processing/processVti.py \
    --dicom_in_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/dicom \
    --datasheet_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/datasheets/start_data.csv \
    --image_out_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/png \
    --out_color_mode L \
    --remove_green_line \
    --save_copy_for_annotation

