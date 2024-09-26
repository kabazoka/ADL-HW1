import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
import pandas as pd

# Load datasets
with open('dataset/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('dataset/valid.json', 'r', encoding='utf-8') as f:
    valid_data = json.load(f)

with open('dataset/context.json', 'r', encoding='utf-8') as f:
    context_data = f.readlines()  # Each line is a paragraph

# Use BertTokenizerFast to support return_offset_mapping
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# Dataset class handling both paragraph and span selection
class QAWithParagraphSelectionDataset(Dataset):
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
        relevant_paragraph = self.context_data[item['relevant']].strip()
        
        answer_text = item['answer']['text']
        answer_start = item['answer']['start']
        
        # Tokenize the question + relevant paragraph for span prediction
        inputs = self.tokenizer(
            question,
            relevant_paragraph,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,  # Now supported with BertTokenizerFast
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        offset_mapping = inputs['offset_mapping'].squeeze(0)  # List of tuples with start and end character positions
        
        # Find the start and end token positions for the answer
        start_position, end_position = None, None
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_position = idx
            if start < answer_start + len(answer_text) <= end:
                end_position = idx

        # If the answer is not fully found, set them to zero or handle accordingly
        if start_position is None or end_position is None:
            start_position, end_position = 0, 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

# Prepare datasets
train_dataset = QAWithParagraphSelectionDataset(train_data, context_data, tokenizer, max_len=512)
valid_dataset = QAWithParagraphSelectionDataset(valid_data, context_data, tokenizer, max_len=512)

# Load Pretrained Model
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=1,  # Effective batch size is 1*2 = 2
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    per_device_eval_batch_size=1,
    num_train_epochs=3,  # You can adjust this between 1 and 3
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

# Save the span prediction model
model.save_pretrained("./span_prediction_model")
tokenizer.save_pretrained("./span_prediction_model")

print(f"Model and tokenizer saved to ./span_prediction_model")

# Evaluate on validation set
trainer.evaluate()
