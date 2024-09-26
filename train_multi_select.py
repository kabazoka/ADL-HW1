import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd

# Load the datasets
with open('dataset/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('dataset/valid.json', 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

with open('dataset/context.json', 'r', encoding='utf-8') as f:
    context_data = f.readlines()  # Each line is a paragraph

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

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
        
        # Format input as question + paragraph
        inputs = [f"question: {question} context: {paragraph}" for paragraph in paragraphs]
        
        # Tokenize the inputs
        tokenized_inputs = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.max_len,    # 1024 max tokens for mT5
            padding="max_length",       # Pad up to max length
            return_tensors="pt"         # Return PyTorch tensors
        )

        try:
            relevant_idx = paragraphs.index(relevant_paragraph)  # Get index of relevant paragraph
        except ValueError:
            # If index is not found, debug by printing or logging the issue
            print(f"Relevant paragraph not found for ID {item['id']}")
            relevant_idx = -1  # Or handle this more gracefully depending on your needs

        return {
            'input_ids': tokenized_inputs['input_ids'],  # Shape: (num_choices, max_len)
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': torch.tensor(relevant_idx, dtype=torch.long)  # Use the index of the relevant paragraph
        }


# Instantiate the training and validation datasets
train_dataset = MultipleChoiceDataset(train_data, context_data, tokenizer, max_len=1024)  # mT5 allows up to 1024 tokens
valid_dataset = MultipleChoiceDataset(valid_data, context_data, tokenizer, max_len=1024)

# Load Pretrained Model
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=1,  # Effective batch size is 1*2 = 2
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./paragraph_selection_model")
tokenizer.save_pretrained("./paragraph_selection_model")

print(f"Model and tokenizer saved to ./paragraph_selection_model")

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

        inputs = [f"question: {question} context: {paragraph}" for paragraph in paragraphs]

        # Tokenize the inputs
        tokenized_inputs = self.tokenizer(
            inputs,
            truncation=True,  # Truncate paragraphs if they are too long
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'id': item['id']
        }

# Prepare test dataset
test_dataset = TestDataset(test_data, context_data, tokenizer, max_len=1024)

# Make predictions
predictions = trainer.predict(test_dataset)

# Output predictions to CSV
test_ids = [item['id'] for item in test_data]
predicted_relevant = predictions.predictions.argmax(axis=-1)  # Get the best paragraph index

submission = pd.DataFrame({
    'id': test_ids,
    'relevant': predicted_relevant
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")