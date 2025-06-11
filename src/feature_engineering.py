import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(df):
    """Add new derived features for wallet behavior."""
    df = df.copy()

    # Transaction-based features
    df['tx_rate'] = df['total transactions (including tnx to create contract)'] / (df['days_since_first_transaction'] + 1)
    df['avg_gas_per_tx'] = df['total gas used'] / (df['total transactions (including tnx to create contract)'] + 1)
    df['ether_balance_ratio'] = df['total ether balance'] / (df['total ether received'] + 1)
    
    # Value-based behavior
    df['avg_val_diff'] = df['avg val received'] - df['avg val sent']
    df['total_val_diff'] = df['total ether received'] - df['total ether sent']

    # Avoid division-by-zero errors
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def scale_features(df, feature_cols):
    """Standard scale the specified features."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler

def get_top_correlated_features(df, target_col='FLAG', top_n=15):
    """Return top N features most correlated with target."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_col].drop(target_col)
    top_features = correlations.abs().sort_values(ascending=False).head(top_n).index.tolist()
    return top_features

