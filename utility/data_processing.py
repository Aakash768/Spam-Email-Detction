import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

def preprocess_text(text):
    """
    Optimized text preprocessing function
    """
    # Lowercase and remove punctuation in one step
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Remove numbers and extra whitespace in one pass
    text = re.sub(r'\d+\s*', '', text)
    return ' '.join(text.split())

def load_and_preprocess_data(file_path):
    """
    Load and preprocess spam dataset with optimized processing
    """
    # Read data more efficiently
    data = pd.read_csv(file_path, usecols=['Category', 'Message'])
    
    # Efficient data cleaning
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    # Vectorize preprocessing
    data['Cleaned_message'] = data['Message'].apply(preprocess_text)
    
    # Efficient mapping
    data['Label_num'] = data['Category'].map({'ham': 0, 'spam': 1})
    
    return data

def vectorize_and_split_data(data, max_features=3000):
    """
    Vectorize data and split into train/test sets
    """
    # Vectorization with optimized parameters
    tfidf = TfidfVectorizer(
        max_features=max_features, 
        stop_words='english',
        lowercase=False  # Already lowercased
    )
    
    # Directly transform cleaned messages
    x = tfidf.fit_transform(data['Cleaned_message']).toarray()
    y = data['Label_num']
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    return x_train, x_test, y_train, y_test, tfidf

def save_preprocessing_artifacts(x, y, tfidf):
    """
    Save preprocessing artifacts efficiently
    """
    # Save vectorizer
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    
    # Save processed data
    processed_data = pd.DataFrame(
        x, 
        columns=tfidf.get_feature_names_out()
    )
    processed_data['Label_num'] = y.values
    processed_data.to_csv('spam_processed.csv', index=False)
    
    # Save vocabulary
    with open('tfidf_vocal.pkl', 'wb') as file:
        pickle.dump(tfidf.vocabulary_, file)

def main():
    # Main processing pipeline
    data = load_and_preprocess_data("spam.csv")
    x_train, x_test, y_train, y_test, tfidf = vectorize_and_split_data(data)
    
    # Print dataset information
    print(f"Training Data Shape: {x_train.shape}")
    print(f"Testing Data Shape: {x_test.shape}")
    
    # Save artifacts
    save_preprocessing_artifacts(
        np.vstack([x_train, x_test]), 
        pd.concat([y_train, y_test]), 
        tfidf
    )

if __name__ == "__main__":
    main()