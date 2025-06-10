import pandas as pd

def load_data(filepath):
    """Load raw Ethereum wallet data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Perform basic preprocessing steps."""
    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert 'FLAG' to int if needed
    if df['FLAG'].dtype != int:
        df['FLAG'] = df['FLAG'].astype(int)

    # Handle missing values (example: fill 0 or drop)
    df.fillna(0, inplace=True)

    return df

def save_processed_data(df, out_path):
    """Save processed data to CSV."""
    df.to_csv(out_path, index=False)

