import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import time

def load_and_prepare_data(file_path):
    """Load data and prepare for model training"""
    data = pd.read_csv(file_path)
    X = data.drop('Label_num', axis=1)
    y = data["Label_num"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model(model, X_train, y_train, model_name):
    """Train model, measure training time, and save"""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"{model_name} Training Time: {training_time:.4f} seconds")
    
    # Save model with consistent naming
    joblib.dump(model, f'../models/saved_models/{model_name.lower().replace(" ", "_")}_model.joblib')
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))

def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_prepare_data('../data/processed/spam_processed.csv')
    
    # Define models with optimized parameters
    models = [
        (LogisticRegression(max_iter=300, solver='liblinear'), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), "Random Forest"),
        (SVC(kernel="linear", probability=True, random_state=42), "Support Vector Machine"),
        (MultinomialNB(), "Naive Bayes")
    ]
    
    # Train and evaluate each model
    for model, name in models:
        trained_model = train_and_save_model(model, X_train, y_train, name)
        evaluate_model(trained_model, X_test, y_test, name)

if __name__ == "__main__":
    main()