import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import os

debug = False  # Set to True to enable debugging

# Load datasets
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_json('dataset/train.json')
valid_data = load_json('dataset/valid.json')
context_data = load_json('dataset/context.json')

# Use BertTokenizerFast to support return_offset_mapping
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# Pre-tokenized Dataset class to avoid re-tokenizing in __getitem__
class QAWithParagraphSelectionDataset(Dataset):
    def __init__(self, data, context_data, tokenizer, max_len):
        self.data = data
        self.context_data = context_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocessed_data = self._preprocess()

    def _preprocess(self):
        preprocessed = []
        for item in tqdm(self.data, desc="Processing Dataset", unit="samples"):
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
                return_offsets_mapping=True,  
                return_tensors="pt"
            )

            input_ids = inputs['input_ids'].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
            offset_mapping = inputs['offset_mapping'].squeeze(0)

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

            preprocessed.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'start_positions': torch.tensor(start_position, dtype=torch.long),
                'end_positions': torch.tensor(end_position, dtype=torch.long)
            })

        return preprocessed

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

# Initialize datasets with progress bar
train_dataset = QAWithParagraphSelectionDataset(train_data, context_data, tokenizer, max_len=512)
valid_dataset = QAWithParagraphSelectionDataset(valid_data, context_data, tokenizer, max_len=512)

# Load Pretrained Model
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")

# Initialize Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Scheduler for learning rate
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# Prepare model, optimizer, dataloaders with Accelerator
model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
)

# Training loop with logging
progress_bar = tqdm(range(num_training_steps))
best_val_loss = float("inf")  # Initialize best validation loss with infinity

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Log average training loss for this epoch
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} finished with average training loss: {avg_train_loss}")

    # Validation phase
    model.eval()
    total_val_loss = 0
    correct_start_preds = 0
    correct_end_preds = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in valid_dataloader:
            outputs = model(**batch)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            start_preds = torch.argmax(outputs.start_logits, dim=-1)
            end_preds = torch.argmax(outputs.end_logits, dim=-1)

            correct_start_preds += (start_preds == batch['start_positions']).sum().item()
            correct_end_preds += (end_preds == batch['end_positions']).sum().item()
            total_predictions += batch['start_positions'].size(0)

    avg_val_loss = total_val_loss / len(valid_dataloader)
    start_accuracy = correct_start_preds / total_predictions * 100
    end_accuracy = correct_end_preds / total_predictions * 100

    print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss}")
    print(f"Start Position Accuracy: {start_accuracy:.2f}%")
    print(f"End Position Accuracy: {end_accuracy:.2f}%")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if not os.path.exists("./best_model"):
            os.makedirs("./best_model")
        unwrapped_model.save_pretrained("./best_model")
        tokenizer.save_pretrained("./best_model")
        print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss}")

    model.train()

# Final evaluation
model.eval()
total_val_loss = 0
correct_start_preds = 0
correct_end_preds = 0
total_predictions = 0

with torch.no_grad():
    for batch in valid_dataloader:
        outputs = model(**batch)
        val_loss = outputs.loss
        total_val_loss += val_loss.item()

        start_preds = torch.argmax(outputs.start_logits, dim=-1)
        end_preds = torch.argmax(outputs.end_logits, dim=-1)

        correct_start_preds += (start_preds == batch['start_positions']).sum().item()
        correct_end_preds += (end_preds == batch['end_positions']).sum().item()
        total_predictions += batch['start_positions'].size(0)

final_val_loss = total_val_loss / len(valid_dataloader)
start_accuracy = correct_start_preds / total_predictions * 100
end_accuracy = correct_end_preds / total_predictions * 100

print(f"Final Validation Loss: {final_val_loss}")
print(f"Final Start Position Accuracy: {start_accuracy:.2f}%")
print(f"Final End Position Accuracy: {end_accuracy:.2f}%")
