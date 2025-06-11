import os
os.makedirs("plots", exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# Load cleaned feature-engineered dataset
df = pd.read_csv('/content/BlockChain-data-analysis/data/processed/cleaned_dataset.csv')

# Remove unwanted columns
drop_cols = ['Unnamed: 0', 'Index']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Separate features and label
X = df.drop(columns=['FLAG', 'Address'])
y = df['FLAG']

# Replace whitespaces with NaN and convert strings to numeric where possible
X.replace(' ', np.nan, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = f"plots/conf_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_roc(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    filename = f"plots/roc_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def train_evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n📊 Model: {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    
    plot_conf_matrix(y_test, preds, name)

 
    if proba is not None:
        plot_roc(y_test, proba, name)

train_evaluate(LogisticRegression(max_iter=1000), "Logistic Regression")
train_evaluate(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
train_evaluate(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost")

