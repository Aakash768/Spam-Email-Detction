import os
import joblib
import re
import string
from typing import List, Tuple

class SpamClassifier:
    def __init__(self, models_dir='../models/saved_models', vectorizer_path="../data/processed/tfidf_vectorizer.joblib"):
        """Initialize spam classifier with available models"""
        self.models_dir = models_dir
        self.vectorizer = joblib.load(vectorizer_path)
        self.available_models = self._load_models()
    
    def _load_models(self) -> dict:
        """Dynamically load available models"""
        models = {}
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.split('_model.joblib')[0].upper()
            model_path = os.path.join(self.models_dir, model_file)
            models[model_name] = joblib.load(model_path)
        
        return models
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return ' '.join(text.split())
    
    def predict(self, text: str, model_name: str = 'RF') -> Tuple[str, float]:
        """Predict spam probability using specified model"""
        cleaned_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([cleaned_text]).toarray()
        
        model = self.available_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0]
        spam_prob = proba[1] if prediction == 1 else proba[0]
        
        return 'Spam' if prediction == 1 else 'Ham', spam_prob
    
    def list_available_models(self) -> List[str]:
        """List available models"""
        return list(self.available_models.keys())

def interactive_spam_classifier():
    """Interactive spam classification interface"""
    classifier = SpamClassifier()
    
    print("🚀 Spam Classification Inference Tool 🚀")
    print("\nAvailable Models:", ", ".join(classifier.list_available_models()))
    
    while True:
        try:
            # User input
            email = input("\nEnter email text (or 'quit' to exit): ").strip()
            
            if email.lower() == 'quit':
                print("Thank you for using Spam Classification Tool!")
                break
            
            # Model selection
            print("\nAvailable Models:", ", ".join(classifier.list_available_models()))
            model_choice = input("Select model (default is RF): ").upper() or 'RF'
            
            # Prediction
            result, probability = classifier.predict(email, model_choice)
            
            # Display results
            print(f"\n📧 Email Analysis:")
            print(f"Classification: {result}")
            print(f"Confidence: {probability * 100:.2f}%")
            print(f"Model Used: {model_choice}")
            
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    interactive_spam_classifier()

if __name__ == "__main__":
    main()