# Question Answering System

This repository contains code for training and inference of a Question Answering (QA) system, which consists of two main components:

1. **Paragraph Selection Model**: Selects the most relevant paragraph for a given question.
2. **Span Prediction Model**: Predicts the answer span within the selected paragraph.

Both models are fine-tuned using pre-trained models from Hugging Face Transformers.

## Table of Contents

- [Question Answering System](#question-answering-system)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Run the Installation Script](#2-run-the-installation-script)
    - [3. Activate the Environment](#3-activate-the-environment)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Paragraph Selection Model](#paragraph-selection-model)
    - [Span Prediction Model](#span-prediction-model)
  - [Inference](#inference)
  - [Usage](#usage)
    - [Training Scripts](#training-scripts)
    - [Inference Script](#inference-script)
    - [Command-Line Arguments](#command-line-arguments)
  - [Results](#results)
  - [License](#license)
  - [**Additional Notes**](#additional-notes)

## Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA support (optional but recommended for faster training)
- Python 3.8 or higher

## Installation

Follow these steps to set up the environment:

### 1. Clone the Repository

```bash
git clone https://github.com/kabazoka/ADL_HW1.git
cd ADL_HW1
```

### 2. Run the Installation Script

The `install.sh` script sets up a new conda environment and installs all required packages.

```bash
bash install.sh
```

### 3. Activate the Environment

```bash
conda activate qa_env
```

## Dataset Preparation

Ensure that you have the following dataset files in a folder named `dataset`:

- `train.json`: Training data
- `valid.json`: Validation data
- `test.json`: Test data
- `context.json`: Context paragraphs

The dataset structure should be as follows:

- Each item in `train.json` and `valid.json` should contain:
  - `"id"`: Unique identifier for the question
  - `"question"`: The question text
  - `"paragraphs"`: A list of paragraph IDs
  - `"relevant"`: The ID of the relevant paragraph (correct answer)
  - `"answer"`: For training and validation, an object containing:
    - `"text"`: The answer text
    - `"start"`: The starting character index of the answer in the context
- `context.json` should be a dictionary mapping paragraph IDs to their corresponding text.

## Training

### Paragraph Selection Model

Train the paragraph selection model to select the most relevant paragraph for each question.

```bash
python src/train/train_paragraph_selection.py
```

This script performs the following:

- Loads and preprocesses the training and validation data.
- Fine-tunes a pre-trained model (`hfl/chinese-roberta-wwm-ext`) on the multiple-choice task.
- Saves the best model to the `paragraph_selection_model` directory.

**Note**: Adjust hyperparameters like batch size, learning rate, and number of epochs in the script if necessary.

### Span Prediction Model

Train the span prediction model to predict the answer span within the selected paragraph.

```bash
python src/train/train_span_prediction.py
```

This script performs the following:

- Loads and preprocesses the training and validation data.
- Fine-tunes a pre-trained model (`hfl/chinese-roberta-wwm-ext`) on the QA task.
- Saves the best model to the `span_prediction_model` directory.

**Note**: Ensure that the models and tokenizers are compatible and that the dataset is correctly formatted.

## Inference

Run the inference script to generate predictions on the test dataset.

```bash
python src/inference/inference.py
```

The script performs the following:

1. **Paragraph Selection**:
   - Uses the paragraph selection model to select the most relevant paragraph for each question.
2. **Span Prediction**:
   - Uses the span prediction model to predict the answer span within the selected paragraph.
3. **Output**:
   - Generates a `submission.csv` file containing the question IDs and predicted answers.

## Usage

### Training Scripts

- `train_paragraph_selection.py`: Training script for the paragraph selection model.
- `train_span_prediction.py`: Training script for the span prediction model.

Both scripts can be customized by modifying hyperparameters and file paths within the code.

### Inference Script

- `inference.py`: Runs the inference pipeline using the trained models.

### Command-Line Arguments

Alternatively, you can modify the scripts to accept command-line arguments for flexibility.

Example:

```bash
python train_paragraph_selection.py --train_file dataset/train.json --valid_file dataset/valid.json --context_file dataset/context.json --model_name hfl/chinese-roberta-wwm-ext --output_dir paragraph_selection_model --batch_size 8 --num_epochs 3
```

**Note**: You'll need to modify the scripts to parse these arguments using `argparse`.

## Results

After running the inference script, the `submission.csv` file will contain the predicted answers for the test questions.

Example of `submission.csv`:

```csv
id,answer
q1,The predicted answer for question 1
q2,The predicted answer for question 2
...
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Additional Notes**

- **Data Privacy**: Ensure that any data you use complies with data privacy regulations and that you have the rights to use and distribute it.
- **Model Checkpoints**: The trained models (`paragraph_selection_model` and `span_prediction_model`) can be large. Consider using Git Large File Storage (LFS) if you plan to upload them to GitHub.
- **CUDA Compatibility**: If you have an NVIDIA GPU, make sure that your CUDA drivers are compatible with the CUDA version specified in the `install.sh` script.
- **Error Handling**: If you encounter any errors during installation or execution, double-check that all file paths are correct and that your datasets are properly formatted.
