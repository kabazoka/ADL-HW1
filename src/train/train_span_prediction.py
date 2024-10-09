import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering, get_scheduler, DataCollatorWithPadding
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt  # Importing Matplotlib

# Load JSON data function
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

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

# Training and evaluation function
def train_and_evaluate(
        train_file, valid_file, context_file, model_name_or_path, output_dir, 
        learning_rate=3e-5, num_epochs=3, batch_size=4, max_len=512, doc_stride=128, mixed_precision="fp16"):
    
    # Load datasets
    train_data = load_json(train_file)
    valid_data = load_json(valid_file)
    context_data = load_json(context_file)
    
    # Prepare data in Hugging Face datasets format
    def prepare_qa_dataset(data):
        new_data = {'id': [], 'question': [], 'context': [], 'answers': []}
        for idx, item in enumerate(data):
            question = item['question']
            relevant_paragraph = context_data[item['relevant']].strip()
            answer_text = item['answer']['text']
            answer_start = item['answer']['start']
            new_data['id'].append(str(idx))
            new_data['question'].append(question)
            new_data['context'].append(relevant_paragraph)
            new_data['answers'].append({'text': [answer_text], 'answer_start': [answer_start]})
        return Dataset.from_dict(new_data)
    
    train_dataset = prepare_qa_dataset(train_data)
    valid_dataset = prepare_qa_dataset(valid_data)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    
    # Preprocessing the datasets
    def prepare_train_features(examples):
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
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if offsets[token_start_index][0] > end_char or offsets[token_end_index][1] < start_char:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # For validation, we need start_positions and end_positions to compute validation loss
    def prepare_validation_features(examples):
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
        offset_mapping = tokenized_examples["offset_mapping"]  # Keep offset mapping

        tokenized_examples["example_id"] = []
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset mappings that are not part of the context
            offset_mapping[i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(offsets)
            ]

            # Compute start and end positions
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                # No answer case
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Find the start token index
                token_start_index = 0
                while token_start_index < len(offset_mapping[i]) and (
                    offset_mapping[i][token_start_index] is None or
                    offset_mapping[i][token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                token_start_index -= 1

                # Find the end token index
                token_end_index = len(offset_mapping[i]) - 1
                while token_end_index >= 0 and (
                    offset_mapping[i][token_end_index] is None or
                    offset_mapping[i][token_end_index][1] >= end_char
                ):
                    token_end_index -= 1
                token_end_index += 1

                # If the answer is not fully inside the context, label it as (0, 0)
                if token_start_index >= len(offset_mapping[i]) or \
                token_end_index >= len(offset_mapping[i]) or \
                offset_mapping[i][token_start_index] is None or \
                offset_mapping[i][token_end_index] is None or \
                offset_mapping[i][token_start_index][0] > end_char or \
                offset_mapping[i][token_end_index][1] < start_char:
                    tokenized_examples["start_positions"].append(0)
                    tokenized_examples["end_positions"].append(0)
                else:
                    # Otherwise, set the token positions
                    tokenized_examples["start_positions"].append(token_start_index)
                    tokenized_examples["end_positions"].append(token_end_index)

        return tokenized_examples

    # Tokenize datasets
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    valid_dataset = valid_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    # Data collator for training
    data_collator = DataCollatorWithPadding(tokenizer)

    # Custom collate function for evaluation
    def custom_eval_collate_fn(features):
        batch = {}
        # Fields to be converted to tensors
        tensor_keys = {'input_ids', 'attention_mask', 'start_positions', 'end_positions'}
        # Fields to be kept as lists
        list_keys = {'offset_mapping', 'example_id'}

        for key in features[0].keys():
            if key in tensor_keys:
                batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
            elif key in list_keys:
                batch[key] = [f[key] for f in features]
            else:
                # Ignore other keys or handle appropriately
                pass
        return batch

    # Data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=custom_eval_collate_fn)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # Metric
    metric = evaluate.load("squad")

    # Lists to store metrics
    train_losses = []
    val_losses = []
    exact_match_scores = []

    # Training loop
    best_f1 = 0
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
            loss = outputs.loss
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} finished with average training loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_start_logits = []
        all_end_logits = []
        all_example_ids = []
        all_offset_mappings = []

        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                loss = outputs.loss
                total_val_loss += loss.item()

                start_logits = accelerator.gather(outputs.start_logits).cpu().numpy()
                end_logits = accelerator.gather(outputs.end_logits).cpu().numpy()
                example_ids = batch["example_id"]
                offset_mappings = batch["offset_mapping"]
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                all_example_ids.extend(example_ids)
                all_offset_mappings.extend(offset_mappings)

        avg_val_loss = total_val_loss / len(valid_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} finished with average validation loss: {avg_val_loss}")

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

        # Initialize lists for predictions and references
        predictions_list = []
        references_list = []

        # Get predictions
        for example in valid_data:
            example_id = str(valid_data.index(example))
            context = context_data[example['relevant']].strip()
            answers = example['answer']['text']
            answers_start = example['answer']['start']
            gold_answers = [a.strip() for a in [answers] if a.strip()]
            references_list.append({'id': example_id, 'answers': {'text': gold_answers, 'answer_start': [answers_start]}})

            feature_indices = features_per_example.get(example_id, [])
            valid_answers = []

            for feature_index in feature_indices:
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                offset_mapping = all_offset_mappings[feature_index]

                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                            continue
                        if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                            continue
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

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

            # Append prediction to the list
            predictions_list.append({'id': example_id, 'prediction_text': pred_text})

            # Optional: Inspect predicted answers
            # print(f"Question: {example['question']}")
            # print(f"Ground Truth Answer: {gold_answers[0] if gold_answers else 'No Answer'}")
            # print(f"Predicted Answer: {pred_text}")
            # print('-' * 50)

        # Compute metrics
        final_metric = metric.compute(predictions=predictions_list, references=references_list)
        exact_match = final_metric['exact_match']
        exact_match_scores.append(exact_match)
        print(f"Epoch {epoch+1}: Exact Match Score: {exact_match:.2f}%")
        print(f"Epoch {epoch+1}: F1 Score: {final_metric['f1']:.2f}%")

        # Checkpointing if F1 improves
        if final_metric['f1'] > best_f1:
            best_f1 = final_metric['f1']
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(output_dir)
            print(f"Best model saved with F1 score: {best_f1:.2f}%")

        model.train()

    print("Training completed.")

    # Plotting the learning curves
    epochs = range(1, num_epochs + 1)

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve - Loss')
    plt.legend()

    # Plotting Exact Match Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, exact_match_scores, 'g-', label='Exact Match Score')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match (%)')
    plt.title('Learning Curve - Exact Match')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Documenting the losses and exact match scores in log.txt
    with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
        log_file.write(f"Model: {model_name_or_path}\n")
        log_file.write(f"Batch Size: {batch_size}\n")
        log_file.write(f"Epochs: {num_epochs}\n")
        log_file.write("Epoch\tTrain Loss\tValidation Loss\tExact Match Score\n")
        for epoch in range(num_epochs):
            log_file.write(f"{epoch + 1}\t{train_losses[epoch]:.4f}\t{val_losses[epoch]:.4f}\t{exact_match_scores[epoch]:.2f}\n")

# Example usage
train_and_evaluate(
    train_file='dataset/train.json',
    valid_file='dataset/valid.json',
    context_file='dataset/context.json',
    model_name_or_path="hfl/chinese-macbert-base",
    output_dir="./span_prediction_model_macbert",
    learning_rate=3e-5,
    num_epochs=5,
    batch_size=8,
    max_len=512,
    doc_stride=128,
    mixed_precision="fp16"
)