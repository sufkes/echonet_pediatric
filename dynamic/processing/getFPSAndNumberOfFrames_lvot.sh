#!/bin/bash

echo_dir="$(readlink -f "$1")"
#out_path="$2"
#out_path="info_from_dicom_tags.csv"

#(0018,602c) FD 0.023809523910892252                     #   8, 1 PhysicalDeltaX
#(0018,602e) FD 0.023809523910892252                     #   8, 1 PhysicalDeltaY

echo 'Subject,DicomFilePath,FPS,NumberOfFrames,StudyDate,PhysicalDeltaX,PhysicalDeltaY' #> "$out_path"
find "$echo_dir" -type f | sort | while read ff; do ss="$(basename "$ff" | cut -d . -f 1)"; fps="$(dcmdump "$ff" | grep CineRate | cut -d [ -f 2 | cut -d ] -f 1)"; nframes="$(dcmdump "$ff" | grep NumberOfFrames | cut -d [ -f 2 | cut -d ] -f 1)"; sdate_raw="$(dcmdump "$ff" | grep StudyDate | cut -d [ -f 2 | cut -d ] -f 1)"; sdate="$(echo $sdate_raw | cut -c -4)"-"$(echo $sdate_raw | cut -c 5-6)"-"$(echo $sdate_raw | cut -c 7-8)"; deltax="$(dcmdump "$ff" | grep PhysicalDeltaX | awk '{print $3}')"; deltay="$(dcmdump "$ff" | grep PhysicalDeltaY | awk '{print $3}')"; echo "$ss","$ff","$fps","$nframes","$sdate","$deltax","$deltay"; done # >> "$out_path"
