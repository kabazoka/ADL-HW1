import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForMultipleChoice, BertForQuestionAnswering
import pandas as pd
from tqdm.auto import tqdm

# Set the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved models and tokenizer
paragraph_selection_model = BertForMultipleChoice.from_pretrained("./paragraph_selection_model").to(device)
span_prediction_model = BertForQuestionAnswering.from_pretrained("./span_prediction_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("./paragraph_selection_model")

# Load the context and test files
context_file = 'dataset/context.json'  # Replace with your actual context file path
test_file = 'dataset/test.json'  # Replace with your actual test file path

with open(context_file, 'r', encoding='utf-8') as f:
    context_data = json.load(f)

with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Define the Test Dataset class
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_data, tokenizer, max_len=512):
        self.data = data
        self.context_data = context_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        
        # Fetch the paragraphs corresponding to the paragraph indices from context.json
        paragraphs = [self.context_data[p].strip() for p in item['paragraphs']]

        # Tokenize the question with each of the paragraphs
        inputs = self.tokenizer(
            [question] * len(paragraphs),  # Repeat the question for each paragraph
            paragraphs,                    # Candidate paragraphs
            truncation=True,               # Truncate if needed
            max_length=self.max_len,        # Ensure max length of 512 tokens
            padding="max_length",           # Pad to max length
            return_tensors="pt"             # Return tensors
        )

        return {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),  # Shape: (1, num_paragraphs, max_len)
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),  # Shape: (1, num_paragraphs, max_len)
            'paragraphs': paragraphs,  # Keep track of the paragraphs
            'id': item['id'],
            'question': question
        }

# Prepare the test dataset
test_dataset = TestDataset(test_data, context_data, tokenizer)

# Function to remove special tokens like [CLS], [SEP], and [PAD]
def remove_special_tokens(tokens):
    return [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]

# Prediction function to extract the relevant paragraph and then predict the span
def predict_and_save_csv(paragraph_selection_model, span_prediction_model, tokenizer, test_dataset, output_file):
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    
    paragraph_selection_model.eval()  # Set paragraph selection model to evaluation mode
    span_prediction_model.eval()  # Set span prediction model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during inference
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids']  # Shape: (1, num_paragraphs, max_len)
            attention_mask = batch['attention_mask']  # Shape: (1, num_paragraphs, max_len)
            paragraphs = batch['paragraphs']
            example_id = batch['id'][0]
            question = batch['question']
            
            # Step 1: Paragraph selection using the paragraph selection model
            paragraph_selection_outputs = paragraph_selection_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Select the paragraph with the highest score across the paragraphs
            selected_paragraph_idx = torch.argmax(paragraph_selection_outputs.logits[0], dim=-1).item()

            # Get the selected paragraph
            selected_paragraph = paragraphs[selected_paragraph_idx]
            
            # Step 2: Use the span prediction model to predict the answer span within the selected paragraph
            span_inputs = tokenizer(
                question,
                selected_paragraph,
                truncation=True,               # Truncate input if it exceeds max length
                max_length=512,                # Ensure that the input doesn't exceed 512 tokens
                padding="max_length",           # Pad the sequence to max length
                return_tensors="pt"             # Return tensors
            ).to(device)

            # Separate the input for the span prediction model
            input_ids = span_inputs['input_ids']
            attention_mask = span_inputs['attention_mask']

            # Run the span prediction model
            outputs = span_prediction_model(input_ids=input_ids, attention_mask=attention_mask)

            # Get the predicted start and end indices
            start_idx = torch.argmax(outputs.start_logits, dim=-1).item()
            end_idx = torch.argmax(outputs.end_logits, dim=-1).item()

            # Extract the predicted answer from the selected paragraph using the token indices
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()[start_idx:end_idx + 1])

            # Remove special tokens like [CLS], [SEP], [PAD]
            cleaned_tokens = remove_special_tokens(tokens)

            # Join tokens without spaces, remove '##'
            predicted_answer = ''.join(cleaned_tokens).replace('##', '')

            # Append the id and the predicted answer to predictions
            predictions.append({
                'id': example_id,
                'answer': predicted_answer
            })

    # Convert predictions to a pandas DataFrame and save as CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)

# Run prediction and save results
predict_and_save_csv(paragraph_selection_model, span_prediction_model, tokenizer, test_dataset, 'submission.csv')

print("Predictions saved to submission.csv")