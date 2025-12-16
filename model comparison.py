import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# --- 1. Data Setup and Loading ---

FILE_NAME = 'medicine_data.csv'

# Define the expected features
TEXT_FEATURES = ['product_name', 'salt_composition', 'product_manufactured',
                 'medicine_desc', 'side_effects', 'drug_interactions']
TARGET_COLUMN = 'sub_category' # Assumed classification target

try:
    print(f"Loading dataset from: {FILE_NAME}")
    data = pd.read_csv(FILE_NAME)

    # 1.1. Prepare the Target (Y)
    y = data[TARGET_COLUMN]
    
    # 1.2. Prepare the Features (X)
    # Concatenate all text features into a single column 'X_text'
    # Filling NaN with empty strings prevents errors during concatenation.
    X_text = data[TEXT_FEATURES].fillna('').agg(' '.join, axis=1)

    print(f"Shape of the loaded data: {X_text.shape}")
    
    # 1.3. Encode the Target Variable (Y) if it's a string
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Target variable '{TARGET_COLUMN}' successfully encoded into {len(le.classes_)} classes.")
    
    X = X_text # X is now the single combined text column
    y = y_encoded
    
except FileNotFoundError:
    print(f"!! WARNING: File '{FILE_NAME}' not found. Creating a dummy text dataset for demonstration.")
    
    # --- DUMMY DATA CREATION for testing the structure ---
    dummy_data = {
        'sub_category': ['Pain Relief', 'Antibiotic', 'Pain Relief', 'Antifungal', 'Antibiotic'],
        'product_name': ['Disprin', 'Azithral', 'Calpol', 'Candid', 'Augmentin'],
        'salt_composition': ['Aspirin', 'Azithromycin', 'Paracetamol', 'Clotrimazole', 'Amoxicillin'],
        'product_manufactured': ['Reckitt', 'Alkem', 'GSK', 'Glenmark', 'GSK'],
        'medicine_desc': ['Fast acting tablet for headache.', 'Treats bacterial infections in throat.', 'Reduces fever and body aches.', 'Cream for topical fungal infections.', 'Broad spectrum antibiotic.'],
        'side_effects': ['Nausea, heartburn', 'Diarrhea, vomiting', 'Liver damage in overdose', 'Skin irritation', 'Rash, allergic reaction'],
        'drug_interactions': ['Blood thinners', 'Antacids', 'Alcohol', 'None', 'Oral contraceptives']
    }
    data = pd.DataFrame(dummy_data)
    
    # Re-run preprocessing on dummy data
    y = data[TARGET_COLUMN]
    X_text = data[TEXT_FEATURES].fillna('').agg(' '.join, axis=1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X = X_text
    y = y_encoded
    print("Dummy text data created. Proceeding with pipeline setup.")
    print(f"Shape of the dummy data: {X.shape}")


# --- 2. Define the Text Classification Pipelines ---
# The pipeline first vectorizes the text using TF-IDF and then runs the classifier.

# Pipeline Module 1: Logistic Regression (Excellent baseline for classification)
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
], verbose=False)

# Pipeline Module 2: Linear Support Vector Classifier (SVC) (Very effective for text)
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)),
    ('clf', LinearSVC(random_state=42, dual=True)) # dual=True is usually better for n_samples > n_features
], verbose=False)

# Pipeline Module 3: Multinomial Naive Bayes (Standard for text classification)
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)),
    ('clf', MultinomialNB())
], verbose=False)

# List of all pipelines for iteration
pipelines = {
    'Logistic Regression (LR)': pipeline_lr,
    'Linear SVC': pipeline_svc,
    'Multinomial Naive Bayes (NB)': pipeline_nb
}

# --- 3. Train Models and Calculate Cross-Validation Accuracy ---

print("\nTraining text classification models and calculating Cross-Validation Accuracy (k=5)...")
results = {}

# Use 5-fold cross-validation for stability with text data
CV_FOLDS = 5 

for name, pipeline in pipelines.items():
    try:
        # cross_val_score uses the Pipeline to vectorize and train in one step
        scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        results[name] = np.mean(scores)
        print(f"-> {name:<30}: Average Accuracy = {results[name]:.4f}")
    except Exception as e:
        results[name] = 0.0
        print(f"!! {name:<30}: Error during training/CV. This may happen if the number of samples is too small. Error: {e}")


# --- 4. Compare Results and Select the Best Model ---
print("\n" + "="*70)
print("Final Results (Average Cross-Validation Score)")
print("="*70)
if results:
    # Sort models by accuracy in descending order
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for name, score in sorted_results:
        print(f"{name:<30}: {score:.4f}")

    # Identify the best model
    best_model_name = sorted_results[0][0]
    best_accuracy = sorted_results[0][1]

    print("\n" + "="*70)
    print(f"The Best Performing Model is: {best_model_name}")
    print(f"Highest Accuracy Achieved: {best_accuracy:.4f}")
    print("="*70)
else:
    print("No models could be trained successfully.")