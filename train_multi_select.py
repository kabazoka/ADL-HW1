import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
from itertools import chain
from dataclasses import dataclass
from typing import Union, Optional
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

# Load the datasets
with open('dataset/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('dataset/valid.json', 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

with open('dataset/context.json', 'r', encoding='utf-8') as f:
    context_data = f.readlines()  # Each line is a paragraph

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator for dynamically padding the inputs for multiple choice.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        # Ensure padding and truncation are applied here
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,  # Make sure padding is enabled
            max_length=self.max_length,  # Truncate to max_length=512
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# Dataset class for Multiple Choice tasks
class MultipleChoiceDataset(Dataset):
    def __init__(self, data, context_data, tokenizer, max_len):
        self.data = data
        self.context_data = context_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        paragraphs = [self.context_data[p].strip() for p in item['paragraphs']]  # Strip each paragraph
        relevant_paragraph = self.context_data[item['relevant']].strip()  # Strip relevant paragraph

        # First sentence: question
        # Second sentence: paragraph options (multiple choices)
        inputs = [f"{question} {paragraph}" for paragraph in paragraphs]

        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,  # Ensure truncation to max_len=512
            max_length=self.max_len,  # Set max length to 512 for BERT
            padding='max_length',  # Ensure padding is applied
            return_tensors="pt"
        )

        # Check if any tokenized input exceeds the max length of 512 tokens
        input_lengths = [len(ids) for ids in tokenized_inputs['input_ids']]
        for i, length in enumerate(input_lengths):
            if length > 512:
                print(f"Warning: Tokenized input {i} exceeds 512 tokens. Length: {length}")

        try:
            relevant_idx = paragraphs.index(relevant_paragraph)  # Get index of relevant paragraph
        except ValueError:
            print(f"Relevant paragraph not found for ID {item['id']}")
            relevant_idx = -1

        return {
            'input_ids': tokenized_inputs['input_ids'],  # Shape: (num_choices, max_len)
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': torch.tensor(relevant_idx, dtype=torch.long)  # Use the index of the relevant paragraph
        }

# Instantiate the training and validation datasets
train_dataset = MultipleChoiceDataset(train_data, context_data, tokenizer, max_len=512)
valid_dataset = MultipleChoiceDataset(valid_data, context_data, tokenizer, max_len=512)

# Data Collator
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, max_length=512)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# Load Pretrained Model
model = BertForMultipleChoice.from_pretrained("bert-base-chinese")

# Initialize Accelerator
accelerator = Accelerator()

# Prepare the model, optimizer, and dataloaders for the correct device placement
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Use a scheduler for learning rate decay
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
)

# Move model, optimizer, and data to the correct device
model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
)

# Variables to track losses for learning curve visualization
train_losses = []
valid_losses = []

# Training loop
num_epochs = 1
progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_local_main_process)

model.train()
for epoch in range(num_epochs):
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch} finished with average training loss: {avg_train_loss}")

    # Evaluate on the validation set after each epoch
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    valid_losses.append(avg_valid_loss)
    print(f"Epoch {epoch} finished with average validation loss: {avg_valid_loss}")

    model.train()  # Switch back to training mode

# Save model and tokenizer
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./paragraph_selection_model")
tokenizer.save_pretrained("./paragraph_selection_model")

print(f"Model and tokenizer saved to ./paragraph_selection_model")

# Plot the learning curves
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()

# Evaluate and output predictions for the test set
with open('dataset/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Modify dataset class for test without labels
class TestDataset(Dataset):
    def __init__(self, data, context_data, tokenizer, max_len):
        self.data = data
        self.context_data = context_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        paragraphs = [self.context_data[p].strip() for p in item['paragraphs']]

        inputs = [f"{question} {paragraph}" for paragraph in paragraphs]

        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,  # Truncate paragraphs if they are too long
            max_length=self.max_len,
            padding=True,  # Ensure padding is applied
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'id': item['id']
        }

# Prepare test dataset
test_dataset = TestDataset(test_data, context_data, tokenizer, max_len=512)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# Model evaluation
model.eval()
all_predictions = []
test_progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process)
for batch in test_progress_bar:
    with torch.no_grad():
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.extend(predictions.cpu().numpy())

# Output predictions to CSV
test_ids = [item['id'] for item in test_data]
predicted_relevant = all_predictions  # Get the best paragraph index

submission = pd.DataFrame({
    'id': test_ids,
    'relevant': predicted_relevant
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")