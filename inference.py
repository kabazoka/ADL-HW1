import json
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sample_code.utils_qa import postprocess_qa_predictions  # For post-processing predictions
import pandas as pd

# Custom dataset class for test data
class TestQADataset(torch.utils.data.Dataset):
    def __init__(self, data, context_data, tokenizer, max_len):
        self.data = data
        self.context_data = context_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocessed_data = self._preprocess()

    def _preprocess(self):
        preprocessed = []
        for item in tqdm(self.data, desc="Processing Test Dataset", unit="samples"):
            question_id = item['id']
            question = item['question']
            paragraphs = [self.context_data[p].strip() for p in item['paragraphs']]

            # Tokenize the question and all paragraphs
            tokenized_paragraphs = [
                self.tokenizer(
                    question,
                    paragraph,
                    truncation=True,
                    max_length=self.max_len,
                    padding="max_length",
                    return_tensors="pt"
                )
                for paragraph in paragraphs
            ]
            preprocessed.append({
                'id': question_id,  # Add question ID here
                'question': question,
                'paragraphs': paragraphs,
                'tokenized_paragraphs': tokenized_paragraphs,
                'paragraph_ids': item['paragraphs']
            })
        return preprocessed

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

# Function to remove special tokens like [CLS], [SEP], and [PAD]
def remove_special_tokens(tokens):
    return [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]

# Inference function for paragraph selection and span prediction
def run_inference(test_file, context_file, paragraph_selection_model_path, span_prediction_model_path, output_file, max_len=512, batch_size=1):
    # Load the test dataset and context
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(context_file, 'r', encoding='utf-8') as f:
        context_data = json.load(f)

    # Load tokenizer and models for both tasks
    tokenizer = BertTokenizerFast.from_pretrained(span_prediction_model_path)
    paragraph_selection_model = BertForQuestionAnswering.from_pretrained(paragraph_selection_model_path)
    span_prediction_model = BertForQuestionAnswering.from_pretrained(span_prediction_model_path)

    # Create dataset and dataloader
    test_dataset = TestQADataset(test_data, context_data, tokenizer, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Run inference for both tasks
    results = []
    paragraph_selection_model.eval()
    span_prediction_model.eval()

    for batch in tqdm(test_dataloader, desc="Running inference"):
        question_id = batch['id']  # Get question ID
        question = batch['question']
        paragraphs = batch['paragraphs']
        tokenized_paragraphs = batch['tokenized_paragraphs']

        # Step 1: Paragraph Selection
        best_paragraph_score = float('-inf')
        best_paragraph_idx = 0
        for idx, inputs in enumerate(tokenized_paragraphs):
            input_ids = inputs['input_ids'].squeeze(0).to(paragraph_selection_model.device)
            attention_mask = inputs['attention_mask'].squeeze(0).to(paragraph_selection_model.device)

            # Get logits for paragraph selection
            with torch.no_grad():
                outputs = paragraph_selection_model(input_ids=input_ids, attention_mask=attention_mask)
                score = outputs.start_logits.mean().item()  # Using average of logits as a simple score

            # Select the best paragraph
            if score > best_paragraph_score:
                best_paragraph_score = score
                best_paragraph_idx = idx

        # Step 2: Span Prediction on the Selected Paragraph
        selected_paragraph = paragraphs[best_paragraph_idx]
        tokenized_selected_paragraph = tokenized_paragraphs[best_paragraph_idx]
        input_ids = tokenized_selected_paragraph['input_ids'].squeeze(0).to(span_prediction_model.device)
        attention_mask = tokenized_selected_paragraph['attention_mask'].squeeze(0).to(span_prediction_model.device)

        # Predict answer span
        with torch.no_grad():
            outputs = span_prediction_model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.squeeze(0)
            end_logits = outputs.end_logits.squeeze(0)

        # Post-process to extract the answer
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        # Decode answer from tokenized input
        print(input_ids[0][start_idx:end_idx + 1])
        answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx + 1].cpu().numpy())
        # Remove special tokens like [CLS], [SEP], [PAD]
        cleaned_tokens = remove_special_tokens(answer_tokens)
        answer = tokenizer.convert_tokens_to_string(cleaned_tokens)
        # Join tokens without spaces, remove '##'
        answer = ''.join(cleaned_tokens).replace('##', '')
        print(f"Question: {question}, Answer: {answer}, Start: {start_idx}, End: {end_idx}")
        results.append({
            'id': question_id,  # Include the question ID in the results
            'answer': answer,
        })

    # Convert predictions to a pandas DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"Inference completed. Results saved to {output_file}")


# Example usage of the inference script
if __name__ == "__main__":
    run_inference(
        test_file='dataset/test.json',
        context_file='dataset/context.json',
        paragraph_selection_model_path='paragraph_selection_model',
        span_prediction_model_path='span_prediction_model',
        output_file='submission.csv',
        max_len=512,
        batch_size=1
    )
