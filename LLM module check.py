import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import numpy as np
import json
import asyncio # Required for running async functions and sleep
import requests # Required for making synchronous HTTP requests

# --- Configuration and Data Setup ---

# Your CSV file name. Ensure your data is saved with this name.
FILE_NAME = 'medicine_data.csv'

# Text features to be combined for relevance calculation
TEXT_FEATURES = ['product_name', 'salt_composition', 'product_manufactured',
                 'medicine_desc', 'side_effects', 'drug_interactions']

# Sample Query to test the ranking performance
SAMPLE_QUERY = "severe ache in body and high fever"
TOP_K = 3 # Number of top suggestions to check for comparison (must be <= 5)

# API Configuration for the LLM Module
# NOTE: The LLM model is used for semantic scoring, which is necessary when keyword matching fails.
API_KEY = "AIzaSyBfn8zWHrST7D3FvSTCrPGuBkoiwa8E4_c"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- Dummy Data Creation (For demonstration if the file is missing) ---
try:
    print(f"Loading dataset from: {FILE_NAME}")
    data = pd.read_csv(FILE_NAME)

except (FileNotFoundError, pd.errors.ParserError): # Catch ParserError too
    print(f"!! WARNING: File '{FILE_NAME}' not found or malformed. Creating dummy data for comparison.")

    dummy_data = {
        'sub_category': ['Pain Relief', 'Antibiotic', 'Pain Relief', 'Antifungal', 'Allergy', 'Pain Relief', 'Antibiotic', 'Pain Relief', 'Fever', 'Pain Relief'],
        'product_name': ['Disprin Tablet', 'Azithral 500', 'Calpol 650', 'Candid Cream', 'Cetirizine', 'Combiflam', 'Doxy-1', 'Dolo 650', 'Paracip', 'Saridon'],
        'salt_composition': ['Aspirin', 'Azithromycin', 'Paracetamol', 'Clotrimazole', 'Cetirizine', 'Ibuprofen', 'Doxycycline', 'Paracetamol', 'Paracetamol', 'Propyphenazone'],
        'product_manufactured': ['Reckitt', 'Alkem', 'GSK', 'Glenmark', 'Cipla', 'Sanofi', 'USV', 'Microlabs', 'Cipla', 'Piramal'],
        'medicine_desc': ['Fast acting tablet for severe headache and minor body aches.', 'Treats bacterial infections in throat and chest.', 'Reduces high fever and muscle pain quickly.', 'Cream for topical fungal infections like skin rash.', 'Used for hay fever and allergy symptoms, reduces itching.', 'Dual-action for severe body pain and inflammation.', 'Broad spectrum antibiotic for various infections.', 'Effective for high fever, body ache, and mild pain.', 'Fast relief from fever and common cold.', 'Quick relief from headache and tiredness.'],
        'side_effects': ['Nausea, heartburn', 'Diarrhea, vomiting', 'Liver damage in overdose', 'Skin irritation', 'Drowsiness, dry mouth', 'Stomach upset', 'Photosensitivity', 'Rare allergic reactions', 'Stomach problems', 'Dizziness'],
        'drug_interactions': ['Blood thinners', 'Antacids', 'Alcohol', 'None', 'Cold medicines', 'Anticoagulants', 'Dairy products', 'Other fever meds', 'Other fever meds', 'Caffeine']
    }
    data = pd.DataFrame(dummy_data)
    print(f"Dummy medicine data created. Total {len(data)} entries.")

# --- Text Preprocessing ---
# Combine all relevant text columns into a single string for vectorization
data['combined_text'] = data[TEXT_FEATURES].astype(str).agg(' '.join, axis=1)
corpus = data['combined_text'].tolist()

# --- 2. Define the Four Ranking Pipelines (Modules) ---

# Pipeline Module 1: Standard TF-IDF (Unigrams) - Simple keyword matching
pipeline_1 = Pipeline([
    ('tfidf_standard', TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 1)))
])

# Pipeline Module 2: TF-IDF with Word Bigrams - Better context awareness
pipeline_2 = Pipeline([
    ('tfidf_bigram', TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 2)))
])

# Pipeline Module 3: TF-IDF with Character N-grams - Robust to misspellings
pipeline_3 = Pipeline([
    ('tfidf_char_ngram', TfidfVectorizer(stop_words='english', max_df=0.8, analyzer='char_wb', ngram_range=(3, 5)))
])

# All 4 pipelines list (LLM is a placeholder as it uses a different function)
pipelines = {
    '1. Standard TF-IDF (Unigrams)': pipeline_1,
    '2. TF-IDF with Word Bigrams': pipeline_2,
    '3. TF-IDF with Character N-grams': pipeline_3,
    '4. LLM Semantic Scoring (NEW & BEST)': None
}

# --- 3. Comparison Function (for TF-IDF based pipelines) ---

def suggest_top_k_tfidf(pipeline, corpus, query, k=10):
    """
    Ranks candidates using standard TF-IDF cosine similarity. Returns candidates
    and the average score of the top-k results.
    """
    X_matrix = pipeline.fit_transform(corpus)
    query_vector = pipeline.transform([query])
    similarity_scores = cosine_similarity(query_vector, X_matrix).flatten()
    top_k_indices = np.argsort(similarity_scores)[::-1][:k]

    candidates = []
    for i in top_k_indices:
        candidates.append({
            'index': i,
            'product_name': data.loc[i, 'product_name'],
            'description': data.loc[i, 'medicine_desc'],
            'score': similarity_scores[i] # Initial TF-IDF score
        })
    return candidates, np.mean([c['score'] for c in candidates[:TOP_K]])

# --- 4. LLM SEMANTIC SCORING FUNCTION (The new, advanced module) ---

async def suggest_top_k_llm(candidates, query):
    """
    Uses the Gemini LLM to semantically score the relevance of medicine candidates
    to the user's query/symptoms on a scale of 0.0 to 1.0.
    """
    print("   [LLM Module: Calling Gemini API for Semantic Score... Please wait.]")

    # Create the list of candidates for the prompt
    candidate_list_str = "\n".join([f"ID: {i+1}, Name: {c['product_name']}, Desc: {c['description']}" for i, c in enumerate(candidates)])

    system_prompt = (
        "You are an expert medical relevance scoring engine. Your task is to calculate the relevance of each medicine to the user's symptoms/query on a scale of 0.0 to 1.0 (where 1.0 is a perfect match). "
        "Analyze the product description and name against the query's medical terms and symptoms. Output the result ONLY as a JSON array of objects, keeping the original ID and adding the 'relevance_score'."
    )
    user_query = (
        f"Query/Symptoms: '{query}'\n\n"
        f"Score the following list of medicine candidates based on relevance to the symptoms:\n"
        f"{candidate_list_str}"
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "ID": {"type": "INTEGER"},
                        "relevance_score": {"type": "NUMBER", "format": "float"}
                    },
                    "required": ["ID", "relevance_score"]
                }
            }
        }
    }

    # Exponential Backoff implementation
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We use the synchronous requests library inside the async function.
            # This is acceptable for simple single requests like this.
            response = requests.post(
                API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )

            if response.status_code == 429: # Rate limit handling
                raise Exception("Rate limit exceeded")
            
            # Raise an exception for bad status codes (4xx, 5xx)
            response.raise_for_status() 

            result = response.json()

            # Robust parsing based on the expected Gemini API JSON response structure
            if 'candidates' in result and result['candidates'] and result['candidates'][0]['content'] and result['candidates'][0]['content']['parts']:
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                # The LLM is forced to output JSON string by responseMimeType
                llm_scores = json.loads(json_string)
            else:
                raise Exception(f"LLM response structure invalid or empty. Response: {result}")

            # Combine LLM scores with original candidate data
            scored_results = []
            # Map LLM's 1-based ID to the score
            score_map = {item['ID']: item['relevance_score'] for item in llm_scores}

            for i, candidate in enumerate(candidates):
                llm_id = i + 1
                # Use .get() for safety and clamp score
                score = score_map.get(llm_id, 0.0)
                scored_results.append({
                    'product_name': candidate['product_name'],
                    'score': max(0.0, min(1.0, score)) # Clamp score between 0.0 and 1.0
                })

            # Sort by the new LLM relevance score
            final_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

            return final_results

        except Exception as e:
            print(f"!! LLM call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # Pause execution using the Python async sleep
                await asyncio.sleep(wait_time) 
            else:
                print(f"!! LLM call failed after {max_retries} attempts.")
                return [{'product_name': c['product_name'], 'score': 0.0} for c in candidates]
    return []

# --- 5. Comparison Execution ---
async def run_comparison():
    print("\n" + "="*90)
    print(f"TEST QUERY (User Query): '{SAMPLE_QUERY}'")
    print("="*90)

    final_comparison = {}

    # A. TF-IDF Pipelines Ranking Check
    for name, pipeline in list(pipelines.items())[:-1]:
        print(f"\n--- MODULE/PIPELINE: {name} ---")

        # Get candidates using the TF-IDF module and the average score
        candidates, avg_score = suggest_top_k_tfidf(pipeline, corpus, SAMPLE_QUERY, k=TOP_K)

        final_comparison[name] = avg_score

        for i, res in enumerate(candidates[:TOP_K]):
            print(f"  {i+1}. {res['product_name']:<20} | Keyword Score: {res['score']:.4f}")

    # B. LLM Semantic Scoring Module Ranking Check
    name_llm = '4. LLM Semantic Scoring (NEW & BEST)'

    # 1. Use the best performing TF-IDF (Module 2, Word Bigrams, is often best) to get initial candidates
    # We use 5 candidates here to give the LLM more options.
    initial_candidates, _ = suggest_top_k_tfidf(pipeline_2, corpus, SAMPLE_QUERY, k=5)

    # 2. Score these candidates using the LLM
    llm_ranked_results = await suggest_top_k_llm(initial_candidates, SAMPLE_QUERY)

    print(f"\n--- MODULE/PIPELINE: {name_llm} ---")

    # Display Top K LLM results and store average score
    if llm_ranked_results:
        # Check if llm_ranked_results has enough elements before taking mean
        results_to_average = llm_ranked_results[:TOP_K]
        if results_to_average:
            average_llm_score = np.mean([res['score'] for res in results_to_average])
        else:
            average_llm_score = 0.0
            
        final_comparison[name_llm] = average_llm_score

        for i, res in enumerate(llm_ranked_results[:TOP_K]):
            print(f"  {i+1}. {res['product_name']:<20} | Semantic Score: {res['score']:.4f}")
    else:
        final_comparison[name_llm] = 0.0
        print(f"  LLM Scoring failed. Score recorded as 0.0.")


    # C. Final Decision
    print("\n" + "="*90)
    print("FINAL COMPARISON (Average Relevance Score of Top 3 Suggestions)")
    print("="*90)

    sorted_comparison = sorted(final_comparison.items(), key=lambda item: item[1], reverse=True)

    for name, avg_score in sorted_comparison:
        print(f"{name:<40}: Average Score = {avg_score:.4f}")

    best_model_name = sorted_comparison[0][0]
    best_accuracy = sorted_comparison[0][1]

    print("\n" + "="*90)
    print(f"NEW BEST MODULE: {best_model_name}")
    print(f"HIGHEST AVERAGE SCORE: {best_accuracy:.4f}")
    print("This new score is based on 'Meaning Matching' (Semantic Similarity), not simple keyword matching.")
    print("=================================================================================")

# Execute the asynchronous comparison function using asyncio.run() for standard Python execution
if __name__ == "__main__":
    asyncio.run(run_comparison())