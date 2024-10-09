#!/bin/bash

# This script performs inference using your trained models and outputs predictions on test.json.

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

# Set the model paths (ensure these directories contain your trained models)
PARAGRAPH_SELECTION_MODEL_PATH="paragraph_selection_model"
SPAN_PREDICTION_MODEL_PATH="span_prediction_model"

# Activate the conda environment if necessary
# Uncomment the following line if you're using a conda environment named 'qa_env'
# source activate qa_env

# Run the inference code directly
python - <<EOF
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
from tqdm.auto import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np

# Get the file paths from the command line arguments
context_file = "$CONTEXT_FILE"
test_file = "$TEST_FILE"
output_file = "$OUTPUT_FILE"
paragraph_selection_model_path = "$PARAGRAPH_SELECTION_MODEL_PATH"
span_prediction_model_path = "$SPAN_PREDICTION_MODEL_PATH"

# Function to load and prepare test data
def load_test_data(test_file, context_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(context_file, 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    return test_data, context_data

# Inference function
def run_inference(test_data, context_data, paragraph_selection_model_path, span_prediction_model_path, output_file, max_len=512, doc_stride=128, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load paragraph selection tokenizer and model
    ps_tokenizer = AutoTokenizer.from_pretrained(paragraph_selection_model_path)
    ps_model = AutoModelForMultipleChoice.from_pretrained(paragraph_selection_model_path)
    ps_model.to(device)
    ps_model.eval()
    
    # Load span prediction tokenizer and model
    sp_tokenizer = AutoTokenizer.from_pretrained(span_prediction_model_path)
    sp_model = AutoModelForQuestionAnswering.from_pretrained(span_prediction_model_path)
    sp_model.to(device)
    sp_model.eval()
    
    # Prepare data for paragraph selection
    ps_examples = []
    for item in test_data:
        question = item['question']
        paragraphs = [context_data[p].strip() for p in item['paragraphs']]
        ps_examples.append({
            'id': item['id'],
            'question': question,
            'paragraphs': paragraphs
        })
    ps_dataset = Dataset.from_list(ps_examples)
    
    # Preprocess function for paragraph selection
    def ps_preprocess_function(examples):
        first_sentences = [[question] * len(paragraphs) for question, paragraphs in zip(examples['question'], examples['paragraphs'])]
        second_sentences = examples['paragraphs']

        # Flatten
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = ps_tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

        # Un-flatten
        num_choices = [len(paragraphs) for paragraphs in examples['paragraphs']]
        tokenized_inputs = {k: [] for k in tokenized_examples.keys()}
        index = 0
        for n in num_choices:
            for k in tokenized_examples.keys():
                tokenized_inputs[k].append(tokenized_examples[k][index: index + n])
            index += n
        tokenized_inputs["id"] = examples["id"]
        return tokenized_inputs

    # Tokenize paragraph selection dataset
    ps_dataset = ps_dataset.map(ps_preprocess_function, batched=True, remove_columns=ps_dataset.column_names)

    # Data collator for paragraph selection
    def ps_data_collator(features):
        batch = {}
        batch_size = len(features)
        num_choices = [len(feature['input_ids']) for feature in features]

        # Flatten features
        flattened_input_ids = []
        flattened_attention_mask = []
        for feature in features:
            flattened_input_ids.extend(feature['input_ids'])
            flattened_attention_mask.extend(feature['attention_mask'])
        # Pad and convert to tensors
        batch['input_ids'] = torch.tensor(flattened_input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(flattened_attention_mask, dtype=torch.long)
        batch['num_choices'] = num_choices
        batch['id'] = [feature['id'] for feature in features]
        return batch

    # DataLoader for paragraph selection
    ps_dataloader = DataLoader(ps_dataset, batch_size=batch_size, shuffle=False, collate_fn=ps_data_collator)

    # Inference for paragraph selection
    question_to_selected_paragraph = {}
    for batch in tqdm(ps_dataloader, desc="Running paragraph selection"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        num_choices = batch['num_choices']
        ids = batch['id']
        # Split input_ids and attention_mask according to num_choices
        input_ids_list = torch.split(input_ids, num_choices)
        attention_mask_list = torch.split(attention_mask, num_choices)
        for i in range(len(ids)):
            input_ids_i = input_ids_list[i].unsqueeze(0)
            attention_mask_i = attention_mask_list[i].unsqueeze(0)
            with torch.no_grad():
                outputs = ps_model(
                    input_ids=input_ids_i,
                    attention_mask=attention_mask_i
                )
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                question_to_selected_paragraph[ids[i]] = prediction

    # Prepare data for span prediction
    sp_examples = {'id': [], 'question': [], 'context': []}
    for item in test_data:
        question_id = item['id']
        question = item['question']
        paragraphs = [context_data[p].strip() for p in item['paragraphs']]
        selected_index = question_to_selected_paragraph[question_id]
        selected_paragraph = paragraphs[selected_index]
        sp_examples['id'].append(question_id)
        sp_examples['question'].append(question)
        sp_examples['context'].append(selected_paragraph)
    sp_dataset = Dataset.from_dict(sp_examples)

    # Tokenize span prediction data
    def prepare_test_features(examples):
        tokenized_examples = sp_tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset mappings that are not part of the context
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    tokenized_dataset = sp_dataset.map(
        prepare_test_features,
        batched=True,
        remove_columns=sp_dataset.column_names,
    )

    # Custom collate function for span prediction
    def sp_data_collator(features):
        batch = {}
        # Fields to be converted to tensors
        tensor_keys = {'input_ids', 'attention_mask'}
        # Fields to be kept as lists
        list_keys = {'offset_mapping', 'example_id'}

        for key in features[0].keys():
            if key in tensor_keys:
                batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
            elif key in list_keys:
                batch[key] = [f[key] for f in features]
            else:
                # Handle any other keys if necessary
                pass
        return batch

    # DataLoader for span prediction
    sp_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=sp_data_collator)

    # Inference for span prediction
    all_start_logits = []
    all_end_logits = []
    all_example_ids = []
    all_offset_mappings = []

    for batch in tqdm(sp_dataloader, desc="Running span prediction"):
        # Move inputs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = sp_model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            all_example_ids.extend(batch["example_id"])
            all_offset_mappings.extend(batch["offset_mapping"])

    # Flatten logits
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-processing to get predictions
    max_answer_length = 30
    n_best_size = 20

    # Build example to features mapping
    features_per_example = {}
    for i, example_id in enumerate(all_example_ids):
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(i)

    predictions = []

    for example in tqdm(sp_examples['id'], desc="Post-processing predictions"):
        example_id = example
        context = sp_examples['context'][sp_examples['id'].index(example_id)]
        feature_indices = features_per_example.get(example_id, [])
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = all_offset_mappings[feature_index]
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size -1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size -1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid predictions
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index +1 > max_answer_length:
                        continue
                    # Get the predicted answer
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    predicted_answer = context[start_char:end_char]
                    valid_answers.append({
                        'score': start_logits[start_index] + end_logits[end_index],
                        'text': predicted_answer
                    })
        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x['score'])
            pred_text = best_answer['text']
        else:
            pred_text = ''

        # Append prediction
        predictions.append({
            'id': example_id,
            'answer': pred_text
        })

    # Convert predictions to a pandas DataFrame and save as CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f"Inference completed. Results saved to {output_file}")

# Run the inference
if __name__ == "__main__":
    # Load the test data and context
    test_data, context_data = load_test_data(test_file, context_file)
    # Run inference
    run_inference(
        test_data=test_data,
        context_data=context_data,
        paragraph_selection_model_path=paragraph_selection_model_path,
        span_prediction_model_path=span_prediction_model_path,
        output_file=output_file,
        max_len=512,
        doc_stride=128,
        batch_size=8
    )
EOF