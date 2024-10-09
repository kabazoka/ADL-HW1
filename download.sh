#!/bin/bash

# This script downloads the zip file containing inference.py and the models from Google Drive and extracts it.

# Function to check if a command exists
command_exists () {
    command -v "$1" &> /dev/null ;
}

# Install gdown if not already installed
if ! command_exists gdown ; then
    echo "gdown could not be found. Installing gdown..."
    python3 -m pip install gdown
fi

# Replace 'YOUR_ZIP_FILE_ID' with the actual file ID from Google Drive
# https://drive.google.com/file/d/15IcrG7DWwPjqEKBCbzt3H6TLtXu9J8bL/view?usp=sharing
ZIP_FILE_ID=11c2FD6ik081G25LxvyznuBFTB1DDKsFr

# Download the zip file
echo "Downloading the zip file containing inference.py and models..."
python3 -m gdown --id $ZIP_FILE_ID -O models_and_code.zip

# Unzip the file
#echo "Extracting the zip file..."
#unzip -o models_and_code.zip
#rm models_and_code.zip

#echo "Download and extraction complete."