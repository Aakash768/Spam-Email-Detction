import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def load_models_and_data(models_path='../models/saved_models', data_path="../data/processed/spam_processed.csv"):
    """Load models and preprocessed data"""
    models = {
        'SVM': joblib.load(f'{models_path}/svm_model.joblib'),
        'Naive Bayes': joblib.load(f'{models_path}/nb_model.joblib'),
        'Random Forest': joblib.load(f'{models_path}/rf_model.joblib'),
        'Linear Regression': joblib.load(f'{models_path}/lr_model.joblib')
    }
    
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Label_num"])
    y = data["Label_num"]
    
    return models, X, y

def evaluate_models(models, X, y):
    """Comprehensive model evaluation"""
    results = []
    
    for name, model in models.items():
        predictions = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Print detailed classification report
        print(f"\n{name} Model Performance:")
        print(classification_report(y, predictions))
    
    return pd.DataFrame(results)

def visualize_model_performance(results_df):
    """Create visualization of model performance"""
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df['F1 Score'], color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Comparison')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load models and data
    models, X, y = load_models_and_data()
    
    # Evaluate models and generate results
    results_df = evaluate_models(models, X, y)
    
    # Display results
    print("\nModel Performance Summary:")
    print(results_df)
    
    # Visualize performance
    visualize_model_performance(results_df)

if __name__ == "__main__":
    main()