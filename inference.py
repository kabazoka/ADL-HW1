import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMultipleChoice, BertForQuestionAnswering
import pandas as pd

# Load saved models and tokenizer
paragraph_selection_model = BertForMultipleChoice.from_pretrained("./paragraph_selection_model")
span_prediction_model = BertForQuestionAnswering.from_pretrained("./span_prediction_model")
tokenizer = BertTokenizerFast.from_pretrained("./paragraph_selection_model")

# Load the context and test files
context_file = 'dataset/context.json'  # Replace with your actual context file path
test_file = 'dataset/test.json'  # Replace with your actual test file path

with open(context_file, 'r', encoding='utf-8') as f:
    context_data = f.readlines()

with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Define the Test Dataset class
class TestDataset(Dataset):
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
        paragraphs = [self.context_data[p].strip() for p in item['paragraphs']]  # Get all related paragraphs
        
        # Tokenize the question and all paragraphs for paragraph selection model
        inputs = self.tokenizer(
            [question] * len(paragraphs),  # Repeat the question for each paragraph
            paragraphs,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'],  # Shape: (num_paragraphs, max_len)
            'attention_mask': inputs['attention_mask'],
            'paragraphs': paragraphs,  # Keep track of the paragraphs
            'id': item['id'],
            'question': question
        }

# Prepare the test dataset
test_dataset = TestDataset(test_data, context_data, tokenizer)

# Prediction function to extract the relevant paragraph and then predict the span
def predict_and_save_csv(paragraph_selection_model, span_prediction_model, tokenizer, test_dataset, output_file):
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    
    paragraph_selection_model.eval()  # Set paragraph selection model to evaluation mode
    span_prediction_model.eval()  # Set span prediction model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during inference
        for batch in test_loader:
            input_ids = batch['input_ids'].squeeze(0)  # Multiple paragraphs
            attention_mask = batch['attention_mask'].squeeze(0)
            paragraphs = batch['paragraphs'][0]  # The list of paragraphs
            example_id = batch['id'][0]
            question = batch['question']

            # Step 1: Paragraph selection using the paragraph selection model
            paragraph_selection_outputs = paragraph_selection_model(input_ids=input_ids, attention_mask=attention_mask)
            selected_paragraph_idx = torch.argmax(paragraph_selection_outputs.logits, dim=1).item()

            # Get the selected paragraph
            selected_paragraph = paragraphs[selected_paragraph_idx]

            # Step 2: Use the span prediction model to predict the answer span within the selected paragraph
            span_inputs = tokenizer(
                question,
                selected_paragraph,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            outputs = span_prediction_model(**span_inputs)

            start_idx = torch.argmax(outputs.start_logits, dim=1).item()
            end_idx = torch.argmax(outputs.end_logits, dim=1).item()

            # Get the character position mapping
            offset_mapping = span_inputs["offset_mapping"].squeeze(0)
            start_char = offset_mapping[start_idx][0].item()
            end_char = offset_mapping[end_idx][1].item()

            # Extract the predicted answer from the selected paragraph
            predicted_answer = selected_paragraph[start_char:end_char]

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