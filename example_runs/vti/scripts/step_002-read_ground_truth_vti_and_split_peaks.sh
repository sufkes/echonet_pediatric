#!/bin/bash

/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/doppler/processing/processVti.py \
    --dicom_in_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/dicom \
    --datasheet_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/datasheets/start_data.csv \
    --image_out_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/png-split_peaks \
    --file_list_out_path /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/datasheets/file_list-split_peaks.csv \
    --out_color_mode L \
    --remove_green_line \
    --split_peaks \
    --read_annotations \
    --annotations_in_dir /hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/example_runs/vti/annotated_images_example \
    -x 286 \
    -y 417 \
    --rescale
    
