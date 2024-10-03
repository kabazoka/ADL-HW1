#!/bin/bash

# This script performs inference using the trained models and outputs predictions on test.json.

# Usage:
# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv

# Arguments:
# $1: path to context.json
# $2: path to test.json
# $3: path to the output prediction file named prediction.csv

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv"
    exit 1
fi

CONTEXT_FILE=$1
TEST_FILE=$2
OUTPUT_FILE=$3

# Activate the conda environment if necessary
# Uncomment the following line if you're using a conda environment named 'qa_env'
# source activate qa_env

# Run the inference script
python3 inference.py \
    --context_file "$CONTEXT_FILE" \
    --test_file "$TEST_FILE" \
    --output_file "$OUTPUT_FILE" \
    --paragraph_selection_model_path "paragraph_selection_model" \
    --span_prediction_model_path "span_prediction_model" \
    --max_len 512 \
    --doc_stride 128 \
    --batch_size 8