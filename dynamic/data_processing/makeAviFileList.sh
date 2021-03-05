#!/bin/bash

# Generate a CSV file (print to std out) containing paths to the processed AVI files. Use this when joining all of the data to make FileList.csv for input to EchoNet.

in_dir="$1"

# Print CSV header
echo "Subject,FilePath"
find "$in_dir" -type f | sort | while read ff; do subject="$(basename "$ff" | cut -d . -f 1)"; file_path="$(readlink -f "$ff")"; echo "$subject","$file_path"; done
