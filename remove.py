import csv

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Process each row
        for row in reader:
            cleaned_row = [cell.replace("'", "").replace("[", "").replace("]", "") for cell in row]
            writer.writerow(cleaned_row)

if __name__ == "__main__":
    input_csv = 'submission.csv'  # Replace with the path to your input CSV file
    output_csv = 'submission_t.csv'  # Replace with the path to your output CSV file
    clean_csv(input_csv, output_csv)
    print(f"Cleaned CSV file saved as {output_csv}")