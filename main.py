import os
import pandas as pd
from preprocessing.clean_text import preprocess_text
from preprocessing.create_bigrams import generate_bigrams
from models.train_fasttext import prepare_fasttext_data, train_fasttext_model
from models.test_model import evaluate_model
from models.embeddings import train_with_pretrained_embeddings

# Function to display results in a table format
def display_results(results):
    # Create a DataFrame from the results list
    df = pd.DataFrame(results, columns=["Configuration", "Precision", "Recall", "F1 Score"])
    print(df)

def preprocess_data(input_file, output_file, bigrams=False):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            label, text = line.strip().split(',', 1)
            cleaned_text = preprocess_text(text)
            if bigrams:
                cleaned_text = generate_bigrams(cleaned_text)
            outfile.write(f"{label},{cleaned_text}\n")

def train_and_evaluate(train_file, model_file, test_file, configurations):
    results = []  # List to store each configuration's results

    for config in configurations:
        print(f"Training configuration: {config['name']}")
        
        # Train the FastText model with specified configuration
        model = train_fasttext_model(
            train_file,
            model_file,
            lr=config['lr'],
            epoch=config['epoch'],
            word_ngrams=config['word_ngrams']
        )
        # Ensure Pandas displays all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)  # Expands display width for full output
        pd.set_option('display.max_colwidth', None)  # Ensures full column content is shown

        # Evaluate the model
        precision, recall, f1 = evaluate_model(model_file, test_file)
        results.append([config['name'], precision, recall, f1])

    # Display the results table
    display_results(results)

def main():
    # File paths
    input_data = "Downloads/project/project/data/Walmart_reviews_data.csv"
    processed_data = "Downloads/project/project/data/processed_walmart_reviews.txt"
    fasttext_data = "Downloads/project/project/data/fasttext_walmart_reviews.txt"
    model_file = "Downloads/project/project/data/walmart_reviews_model.bin"
    test_data = "Downloads/project/project/data/test_walmart_reviews.txt"
    
    # Step 1: Preprocess data and generate bigrams if necessary
    preprocess_data(input_data, processed_data, bigrams=True)
    
    # Step 2: Prepare data in FastText format
    prepare_fasttext_data(processed_data, fasttext_data)
    
    # Configurations to test
    configurations = [
        {"name": "Original Model", "lr": 0.1, "epoch": 25, "word_ngrams": 1, "loss": "softmax"},
        {"name": "Bigrams", "lr": 0.1, "epoch": 25, "word_ngrams": 2, "loss": "softmax"},
        {"name": "Hierarchical Softmax (HS)", "lr": 0.1, "epoch": 25, "word_ngrams": 1, "loss": "hs"},
        {"name": "Multilabel", "lr": 0.1, "epoch": 25, "word_ngrams": 1, "loss": "ova"},
        {"name": "Multilabel Bigrams", "lr": 0.1, "epoch": 25, "word_ngrams": 2, "loss": "ova"},
        {"name": "Multilabel HS", "lr": 0.1, "epoch": 25, "word_ngrams": 1, "loss": "hs"}
    ]
    
    # Step 3: Train and evaluate FastText model with different configurations
    train_and_evaluate(fasttext_data, model_file, test_data, configurations)

if __name__ == "__main__":
    main()
