#!/bin/bash

# Install gdown if not already installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing..."
    pip install gdown
fi

# Google Drive file IDs
FILE_ID_1="model1.pt"
FILE_ID_2="model2.pt"

# Output file names
OUTPUT_1="model1.pt"
OUTPUT_2="model2.pt"

# Download the files
gdown --id $FILE_ID_1 -O $OUTPUT_1
gdown --id $FILE_ID_2 -O $OUTPUT_2

echo "Download completed."