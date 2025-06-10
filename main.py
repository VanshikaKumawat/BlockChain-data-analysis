from src.data_prep import load_data, preprocess_data, save_processed_data

df = load_data("data/raw/transaction_dataset.csv")
df = preprocess_data(df)
save_processed_data(df, "data/processed/cleaned_wallet_data.csv")
# Main ML pipeline will go here

if __name__ == "__main__":
    print("Welcome to the Blockchain Forensics Project.")
