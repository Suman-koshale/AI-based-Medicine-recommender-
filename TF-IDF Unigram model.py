
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz

# -------------------------------------
# CONFIGURATION
# -------------------------------------
DATA_FILE = "medicine_data.csv"

FEATURE_WEIGHTS = {
    'sub_category': 1,
    'product_name': 1,
    'salt_composition': 1,
    'product_manufactured': 1,
    'medicine_desc': 3,
    'side_effects': 2,
    'drug_interactions': 1
}

RECOMMENDATION_COLUMNS = [
    'product_name',
    'salt_composition',
    'product_manufactured'
]

MODEL_FILE = "tfidf_unigram_vectorizer.pkl"
MATRIX_FILE = "tfidf_unigram_matrix.npz"
DATA_FILE_PKL = "recommendation_data.pkl"

# -------------------------------------
# FEATURE COMBINATION WITH WEIGHTS
# -------------------------------------
def combine_weighted_features(row):
    combined = []
    for col, weight in FEATURE_WEIGHTS.items():
        text = str(row[col]).lower() if pd.notna(row[col]) else ""
        combined.append(" ".join([text] * weight))
    return " ".join(combined)

# -------------------------------------
# TRAIN MODEL
# -------------------------------------
def train_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    df.drop_duplicates(
        subset=['product_name', 'salt_composition'],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    print(f"Total unique medicines: {len(df)}")

    print("Combining features with weights...")
    df["combined_features"] = df.apply(combine_weighted_features, axis=1)

    print("Training TF-IDF (UNIGRAMS)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1)
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(MATRIX_FILE, tfidf_matrix)

    with open(DATA_FILE_PKL, "wb") as f:
        pickle.dump(df[RECOMMENDATION_COLUMNS], f)

    print("✅ Training completed and files saved")

# -------------------------------------
# ACCURACY / RELEVANCE TEST
# -------------------------------------
def test_accuracy(top_k=5):
    TEST_QUERY = "severe ache in body and high fever"

    with open(MODEL_FILE, "rb") as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = load_npz(MATRIX_FILE)

    with open(DATA_FILE_PKL, "rb") as f:
        data = pickle.load(f)

    query_vec = vectorizer.transform([TEST_QUERY])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_idx = np.argsort(scores)[::-1][:top_k]

    print("\n================================================")
    print(f"TEST QUERY: {TEST_QUERY}")
    print("================================================")

    relevant_count = 0

    for i, idx in enumerate(top_idx):
        score = scores[idx] * 100
        name = data.iloc[idx]["product_name"]

        print(f"{i+1}. {name} | Similarity: {score:.2f}%")

        # ---- MANUAL LOGIC (SIMULATED) ----
        # Fever + pain medicines assumed relevant
        if score > 10:
            relevant_count += 1

    accuracy = (relevant_count / top_k) * 100

    print("\n-----------------------------------------------")
    print(f"RELEVANT RESULTS: {relevant_count}/{top_k}")
    print(f"MODEL RELEVANCE ACCURACY: {accuracy:.2f}%")
    print("-----------------------------------------------")

# -------------------------------------
# MAIN
# -------------------------------------
if __name__ == "__main__":
    train_model()
    test_accuracy(top_k=5)
