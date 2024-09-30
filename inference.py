import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np

# Function to load and prepare test data
def load_test_data(test_file, context_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(context_file, 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    examples = {'id': [], 'question': [], 'context': []}
    for item in test_data:
        question_id = item['id']
        question = item['question']
        paragraphs = [context_data[p].strip() for p in item['paragraphs']]
        # We process each paragraph separately
        for idx, paragraph in enumerate(paragraphs):
            example_id = f"{question_id}_{idx}"
            examples['id'].append(example_id)
            examples['question'].append(question)
            examples['context'].append(paragraph)
    return Dataset.from_dict(examples)

# Normalize answers function
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_punctuation(lower(s)))

# Inference function
def run_inference(test_file, context_file, model_path, output_file, max_len=512, doc_stride=128, batch_size=8):
    # Load the test dataset and context
    test_dataset = load_test_data(test_file, context_file)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Tokenize test data
    def prepare_test_features(examples):
        tokenized_examples = tokenizer(
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

    tokenized_dataset = test_dataset.map(
        prepare_test_features,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # DataLoader
    data_collator = DataLoader(tokenized_dataset, batch_size=batch_size)
    
    # Inference
    all_start_logits = []
    all_end_logits = []
    all_example_ids = []
    
    for batch in tqdm(data_collator, desc="Running inference"):
        # Move inputs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            all_example_ids.extend(batch["example_id"])
    
    # Flatten logits
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    
    # Post-processing to get predictions
    max_answer_length = 30
    n_best_size = 20

    # Build example to features mapping
    example_id_to_index = {k: i for i, k in enumerate(tokenized_dataset['example_id'])}
    features_per_example = {}
    for i, example_id in enumerate(tokenized_dataset['example_id']):
        if example_id not in features_per_example:
            features_per_example[example_id] = []
        features_per_example[example_id].append(i)

    predictions = {}
    
    for example_id in tqdm(set(tokenized_dataset['example_id']), desc="Post-processing predictions"):
        feature_indices = features_per_example[example_id]
        context = test_dataset[int(example_id_to_index[example_id])]['context']
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = tokenized_dataset[feature_index]['offset_mapping']
            
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
            predictions[example_id] = best_answer['text']
        else:
            predictions[example_id] = ''

    # Now, consolidate predictions for each question
    question_id_to_answers = {}

    for example_id, answer in predictions.items():
        question_id = example_id.rsplit('_', 1)[0]  # Remove paragraph index
        if question_id not in question_id_to_answers:
            question_id_to_answers[question_id] = []
        question_id_to_answers[question_id].append({
            'answer': answer,
            'score': 0  # Placeholder; we can sum the scores if needed
        })

    # For each question, select the best answer among all paragraphs
    final_predictions = []
    for question_id, answers in question_id_to_answers.items():
        # Filter out empty answers and normalize
        non_empty_answers = [normalize_answer(ans['answer']) for ans in answers if ans['answer'].strip()]
        if non_empty_answers:
            # Choose the most frequent answer or the first one
            final_answer = max(set(non_empty_answers), key=non_empty_answers.count)
        else:
            final_answer = ''
        final_predictions.append({
            'id': question_id,
            'answer': final_answer
        })

    # Convert predictions to a pandas DataFrame and save as CSV
    df = pd.DataFrame(final_predictions)
    df.to_csv(output_file, index=False)
    print(f"Inference completed. Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    run_inference(
        test_file='dataset/test.json',
        context_file='dataset/context.json',
        model_path='span_prediction_model',
        output_file='submission.csv',
        max_len=512,
        doc_stride=128,
        batch_size=8
    )