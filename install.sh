#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit
fi

# Create a new conda environment
conda create -n qa_env python=3.10 -y

# Activate the environment
conda activate qa_env

# Install PyTorch with CUDA support (adjust cudatoolkit version if needed)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# Install Hugging Face Transformers, Datasets, and other dependencies
pip install transformers datasets accelerate evaluate tqdm pandas numpy

echo "Environment setup complete. Activate the environment using 'conda activate qa_env'."
