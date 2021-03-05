#!/bin/bash

echo_dir="$(readlink -f "$1")"
out_path="info_from_dicom_tags.csv"

echo 'Subject,DicomFilePath,FPS,NumberOfFrames,StudyDate' > "$out_path"
find "$echo_dir" -type f | sort | while read ff; do ss="$(basename "$ff" | cut -d . -f 1)"; fps="$(dcmdump "$ff" | grep CineRate | cut -d [ -f 2 | cut -d ] -f 1)"; nframes="$(dcmdump "$ff" | grep NumberOfFrames | cut -d [ -f 2 | cut -d ] -f 1)"; sdate_raw="$(dcmdump "$ff" | grep StudyDate | cut -d [ -f 2 | cut -d ] -f 1)"; sdate="$(echo $sdate_raw | cut -c -4)"-"$(echo $sdate_raw | cut -c 5-6)"-"$(echo $sdate_raw | cut -c 7-8)"; echo "$ss","$ff","$fps","$nframes","$sdate"; done >> "$out_path"
