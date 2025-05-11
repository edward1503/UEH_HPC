import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def load_data(file_path):
    """Load and preprocess the Give Me Some Credit dataset."""
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    
    # Feature engineering
    df['DebtToIncomeRatio'] = df['DebtRatio'] * df['MonthlyIncome']
    df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    features = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
        'DebtToIncomeRatio',
        'IncomePerPerson'
    ]
    
    X = df[features]
    y = df['SeriousDlqin2yrs']
    
    return X, y

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def main():
    # Create model directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Load and preprocess data
    df = load_data('./data/cs-training.csv')
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'saved_models/credit_model.joblib')
    joblib.dump(scaler, 'saved_models/scaler.joblib')
    
    # Save feature names
    joblib.dump(X.columns.tolist(), 'saved_models/feature_names.joblib')

if __name__ == "__main__":
    main() 