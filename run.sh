#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check that the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_context.json> <path_to_test.json> <path_to_output_prediction.csv>"
    exit 1
fi

# Assign arguments to variables
CONTEXT_PATH="$1"
TEST_PATH="$2"
OUTPUT_PATH="$3"

# Paths to the trained models
PARAGRAPH_MODEL_DIR="./paragraph_selection_model"
SPAN_MODEL_DIR="./span_prediction_model"

# Train and save the paragraph selection model
echo "Training paragraph selection model..."
python3 train_multi_select.py --output_dir "$PARAGRAPH_MODEL_DIR"

# Train and save the span prediction model
echo "Training span prediction model..."
python3 train_span_selection.py --output_dir "$SPAN_MODEL_DIR"

# Perform inference and save predictions to the output file
echo "Running inference..."
python3 inference.py "$CONTEXT_PATH" "$TEST_PATH" "$OUTPUT_PATH"

echo "Inference complete. Predictions saved to $OUTPUT_PATH"