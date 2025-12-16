import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from io import StringIO
import numpy as np

# --- Configuration and Data Setup ---

# Your CSV file name. Ensure your data is saved with this name.
FILE_NAME = 'medicine_data.csv'

# The features (columns) that will be combined for relevance checking
TEXT_FEATURES = ['product_name', 'salt_composition', 'product_manufactured',
                 'medicine_desc', 'side_effects', 'drug_interactions']

# Sample Query for which we will check the Ranking Score.
# You can change this to test the performance on different queries.
SAMPLE_QUERY = "severe ache in body and high fever"

# --- Dummy Data Creation (If the file is not found) ---
try:
    print(f"Loading dataset from: {FILE_NAME}")
    data = pd.read_csv(FILE_NAME)
    
except FileNotFoundError:
    print(f"!! WARNING: File '{FILE_NAME}' not found. Creating dummy data for comparison display.")
    
    dummy_data = {
        'sub_category': ['Pain Relief', 'Antibiotic', 'Pain Relief', 'Antifungal', 'Allergy'],
        'product_name': ['Disprin Tablet', 'Azithral 500', 'Calpol 650', 'Candid Cream', 'Cetirizine'],
        'salt_composition': ['Aspirin', 'Azithromycin', 'Paracetamol', 'Clotrimazole', 'Cetirizine'],
        'product_manufactured': ['Reckitt', 'Alkem', 'GSK', 'Glenmark', 'Cipla'],
        'medicine_desc': ['Fast acting tablet for severe headache and minor body aches.', 'Treats bacterial infections in throat and chest.', 'Reduces high fever and muscle pain quickly.', 'Cream for topical fungal infections like skin rash.', 'Used for hay fever and allergy symptoms, reduces itching.'],
        'side_effects': ['Nausea, heartburn', 'Diarrhea, vomiting', 'Liver damage in overdose', 'Skin irritation', 'Drowsiness, dry mouth'],
        'drug_interactions': ['Blood thinners', 'Antacids', 'Alcohol', 'None', 'Cold medicines']
    }
    data = pd.DataFrame(dummy_data)
    print("Dummy medicine data created.")

# --- Text Preprocessing ---

# 1. Combining all text features into a single 'combined_text' column
data['combined_text'] = data[TEXT_FEATURES].astype(str).agg(' '.join, axis=1)
corpus = data['combined_text'].tolist()
print(f"Total {len(corpus)} medicines are in the database.")


# --- 2. Defining the three (3) Text Processing Pipelines ---

# Pipeline Module 1: Standard TF-IDF (Unigrams)
pipeline_1 = Pipeline([
    ('tfidf_standard', TfidfVectorizer(
        stop_words='english', 
        max_df=0.8,
        ngram_range=(1, 1) # Only single words (Unigrams)
    ))
])

# Pipeline Module 2: TF-IDF with Word Bigrams
pipeline_2 = Pipeline([
    ('tfidf_bigram', TfidfVectorizer(
        stop_words='english', 
        max_df=0.8,
        ngram_range=(1, 2) # Single words and Word Pairs (Bigrams)
    ))
])

# Pipeline Module 3: TF-IDF with Character N-grams (for Robustness)
pipeline_3 = Pipeline([
    ('tfidf_char_ngram', TfidfVectorizer(
        stop_words='english', 
        max_df=0.8,
        analyzer='char_wb', # Character N-grams
        ngram_range=(3, 5) # Looks at sequences of 3 to 5 characters
    ))
])


# List of all pipelines
pipelines = {
    '1. Standard TF-IDF (Unigrams)': pipeline_1,
    '2. TF-IDF with Word Bigrams': pipeline_2,
    '3. TF-IDF with Character N-grams': pipeline_3
}

# --- 3. Model Training (Vectorization) and Comparison Function ---

def suggest_top_k(pipeline, corpus, query, k=5):
    """
    Calculates similarity between the user query and the database, 
    and returns Top K medicines with their Relevance Score.
    """
    # 1. Fit and transform the data using the Pipeline
    X_matrix = pipeline.fit_transform(corpus)
    
    # 2. Transform the Query (using the existing vocabulary)
    query_vector = pipeline.transform([query])
    
    # 3. Calculate the Cosine Similarity Score (This is our 'Accuracy' metric)
    similarity_scores = cosine_similarity(query_vector, X_matrix).flatten()
    
    # 4. Get the Top K indices
    top_k_indices = np.argsort(similarity_scores)[::-1][:k]
    
    # 5. Format the Results
    results = []
    for i in top_k_indices:
        results.append({
            'product_name': data.loc[i, 'product_name'],
            'score': similarity_scores[i]
        })
    return results

# --- 4. Comparison Results ---

print("\n" + "="*80)
print(f"TEST QUERY (User Query): '{SAMPLE_QUERY}'")
print("="*80)

final_comparison = {}

for name, pipeline in pipelines.items():
    print(f"\n--- MODULE/PIPELINE: {name} ---")
    
    # Displaying Top 3 suggestions
    results = suggest_top_k(pipeline, corpus, SAMPLE_QUERY, k=3)
    
    # Calculate the average of the Top 3 scores and store for comparison
    average_top_3_score = np.mean([res['score'] for res in results])
    final_comparison[name] = average_top_3_score
    
    for i, res in enumerate(results):
        print(f"  {i+1}. {res['product_name']:<20} | Relevance Score: {res['score']:.4f}")

# 5. Final Decision
print("\n" + "="*80)
print("FINAL COMPARISON (Average Relevance Score of Top 3 Suggestions)")
print("="*80)

# Sort the scores in descending order
sorted_comparison = sorted(final_comparison.items(), key=lambda item: item[1], reverse=True)

for name, avg_score in sorted_comparison:
    print(f"{name:<40}: Average Score = {avg_score:.4f}")

# Selecting the Best Model
best_model_name = sorted_comparison[0][0]
best_accuracy = sorted_comparison[0][1]

print("\n" + "="*80)
print(f"BEST MODULE: {best_model_name}")
print(f"HIGHEST AVERAGE SCORE: {best_accuracy:.4f} (This is your model's 'Accuracy')")
print("================================================================================")