import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMultipleChoice, get_scheduler
)
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import Dataset as HFDataset
from datasets import load_metric

# Load the datasets
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_json('dataset/train.json')
valid_data = load_json('dataset/valid.json')
context_data = load_json('dataset/context.json')

# Use a more powerful pre-trained model for Chinese text
model_name_or_path = "hfl/chinese-roberta-wwm-ext"

# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path)

# Prepare data in Hugging Face datasets format
def prepare_mc_dataset(data):
    questions = []
    choices = []
    labels = []
    for item in data:
        question = item['question']
        paragraphs = [context_data[p].strip() for p in item['paragraphs']]
        label = item['paragraphs'].index(item['relevant'])
        questions.append(question)
        choices.append(paragraphs)
        labels.append(label)
    return HFDataset.from_dict({'question': questions, 'choices': choices, 'label': labels})

train_dataset = prepare_mc_dataset(train_data)
valid_dataset = prepare_mc_dataset(valid_data)

# Preprocess function
def preprocess_function(examples):
    first_sentences = [[question] * len(choices) for question, choices in zip(examples['question'], examples['choices'])]
    second_sentences = examples['choices']

    # Flatten the inputs
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Un-flatten
    num_choices = len(examples['choices'][0])
    tokenized_inputs = {k: [v[i:i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

# Tokenize datasets
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=valid_dataset.column_names)

# Data collator for Multiple Choice
def mc_data_collator(features):
    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])
    flattened_features = []
    for feature in features:
        for i in range(num_choices):
            flattened_features.append({
                "input_ids": torch.tensor(feature["input_ids"][i]),
                "attention_mask": torch.tensor(feature["attention_mask"][i]),
            })

    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in flattened_features]).view(batch_size, num_choices, -1)
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in flattened_features]).view(batch_size, num_choices, -1)
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch

# DataLoaders
batch_size = 4  # Adjust based on your GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mc_data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=mc_data_collator)

# Initialize Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Prepare the model, optimizer, and dataloaders for the correct device placement
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Scheduler with warm-up
num_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),  # 10% warm-up
    num_training_steps=num_training_steps,
)

# Prepare with Accelerator
model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
)

# Metric
metric = load_metric("accuracy")

# Training loop
progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
best_accuracy = 0.0  # To track the best validation accuracy

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch +1}: Average training loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    total_eval_loss = 0
    all_predictions = []
    all_labels = []
    for step, batch in enumerate(valid_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(accelerator.gather(predictions).cpu().numpy())
            all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())

    avg_eval_loss = total_eval_loss / len(valid_dataloader)
    eval_metric = metric.compute(predictions=all_predictions, references=all_labels)
    accuracy = eval_metric['accuracy'] * 100
    print(f"Epoch {epoch +1}: Average validation loss: {avg_eval_loss:.4f}")
    print(f"Epoch {epoch +1}: Validation accuracy: {accuracy:.2f}%")

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("./paragraph_selection_model_roberta", save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained("./paragraph_selection_model_roberta")
        print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

print("Training completed.")