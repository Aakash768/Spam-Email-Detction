import os
import sys
import time
import unittest
import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Dict, Any

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main utility modules
from utility.data_processing import (
    preprocess_text, 
    load_and_preprocess_data, 
    vectorize_and_split_data, 
    save_preprocessing_artifacts
)
from utility.models_training import (
    load_and_prepare_data, 
    train_and_save_model, 
    evaluate_model
)
from utility.model_evaluation import (
    load_models_and_data, 
    evaluate_models
)
from utility.demo import SpamClassifier

class SpamClassificationTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test resources and paths."""
        cls.test_data_path = "../data/raw/spam.csv"
        cls.processed_data_path = "../data/processed/spam_processed.csv"
        cls.models_dir = "../models/saved_models/"
        
        # Ensure directories exist
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.processed_data_path), exist_ok=True)
        print(f"Setup: Test data path initialized at '{cls.test_data_path}'")
        print(f"Setup: Models directory initialized at '{cls.models_dir}'")

    def test_text_preprocessing(self):
        """Test text preprocessing function."""
        print("\nRunning test: test_text_preprocessing")
        test_texts = [
            "Hello, World! 123",
            "FREE SPAM MESSAGE!!!",
            "Normal email with some numbers 456"
        ]
        expected_results = [
            "hello world",
            "free spam message",
            "normal email with some numbers"
        ]
        
        for text, expected in zip(test_texts, expected_results):
            processed = preprocess_text(text)
            print(f"Processing: '{text}' -> '{processed}'")
            self.assertEqual(processed, expected, f"Failed preprocessing: {text}")

    def test_data_loading_and_preprocessing(self):
        """Test data loading and preprocessing pipeline."""
        print("\nRunning test: test_data_loading_and_preprocessing")
        # Ensure test data exists
        self.assertTrue(os.path.exists(self.test_data_path), "Test data file not found")
        print("Data file exists.")

        # Load and preprocess data
        data = load_and_preprocess_data(self.test_data_path)
        
        # Validate preprocessing
        self.assertTrue('Cleaned_message' in data.columns, "Missing 'Cleaned_message' column")
        self.assertTrue('Label_num' in data.columns, "Missing 'Label_num' column")
        print("Preprocessing completed successfully. Columns validated.")

        # Check data cleaning
        self.assertFalse(data.duplicated().any(), "Duplicates found in data")
        self.assertFalse(data.isnull().any().any(), "Null values found in data")
        print("Data is clean and ready.")

    def test_vectorization_and_splitting(self):
        """Test data vectorization and splitting."""
        print("\nRunning test: test_vectorization_and_splitting")
        data = load_and_preprocess_data(self.test_data_path)
        x_train, x_test, y_train, y_test, tfidf = vectorize_and_split_data(data)
        
        # Validate splitting
        self.assertEqual(len(x_train) + len(x_test), len(data), "Data split sizes mismatch")
        self.assertEqual(x_train.shape[1], x_test.shape[1], "Feature size mismatch")
        print(f"Data split validated. Train size: {len(x_train)}, Test size: {len(x_test)}")

        # Performance checks
        train_ratio = len(x_train) / len(data)
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.05, msg="Train/Test split ratio mismatch")
        print(f"Train/Test ratio validated: {train_ratio:.2f}")

    def test_model_training_performance(self):
        """Test model training performance and artifacts."""
        print("\nRunning test: test_model_training_performance")
        # Preprocess data first
        data = load_and_preprocess_data(self.test_data_path)
        x_train, x_test, y_train, y_test, tfidf = vectorize_and_split_data(data)
        
        # Test training time and artifact generation
        from sklearn.linear_model import LogisticRegression
        
        start_time = time.time()
        model = LogisticRegression(max_iter=300)
        trained_model = train_and_save_model(model, x_train, y_train, "Test Model")
        training_time = time.time() - start_time
        
        # Check training artifacts
        model_path = os.path.join(self.models_dir, "test_model_model.joblib")
        self.assertTrue(os.path.exists(model_path), "Model artifact not found")
        self.assertLess(training_time, 10, "Training took too long")
        print(f"Model training completed in {training_time:.2f} seconds. Model saved at '{model_path}'.")

    def test_model_inference(self):
        """Test spam classification inference."""
        print("\nRunning test: test_model_inference")
        classifier = SpamClassifier()
        
        # Test inference pipeline
        test_texts = [
            "FREE VIAGRA DISCOUNT!!!",  # Likely spam
            "Hey, can we meet for coffee later?"  # Likely ham
        ]
        
        for text in test_texts:
            result, probability = classifier.predict(text)
            print(f"Input: '{text}' -> Prediction: {result}, Probability: {probability:.2f}")
            
            # Validate inference
            self.assertIn(result, ['Spam', 'Ham'])
            self.assertTrue(0 <= probability <= 1, "Probability out of range")

    def test_model_evaluation(self):
        """Comprehensive model performance evaluation."""
        print("\nRunning test: test_model_evaluation")
        models, X, y = load_models_and_data()
        results_df = evaluate_models(models, X, y)
        
        # Validate performance metrics
        for _, row in results_df.iterrows():
            print(f"Model: {row['Model']} -> Accuracy: {row['Accuracy']:.2f}, F1 Score: {row['F1 Score']:.2f}")
            self.assertTrue(0 <= row['Accuracy'] <= 1, "Invalid accuracy metric")
            self.assertTrue(0 <= row['F1 Score'] <= 1, "Invalid F1 Score metric")

def run_tests():
    """Execute test suite with performance timing."""
    print("Starting test execution...\n")
    start_time = time.time()
    
    # Configure test runner
    test_suite = unittest.TestLoader().loadTestsFromTestCase(SpamClassificationTestSuite)
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = test_runner.run(test_suite)
    
    # Performance and summary reporting
    total_time = time.time() - start_time
    print(f"\n🕒 Total Test Execution Time: {total_time:.2f} seconds")
    
    # Return test result status
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
