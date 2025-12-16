import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Configuration ---
DATASET_FILE = "medicine_data.csv"
MODEL_FILE = "medicine_recommender.pkl"
RECOMMENDATION_COLUMNS = ['product_name', 'salt_composition', 'product_manufactured']

# 1️⃣ Load dataset
# NOTE: Replace 'medicine_data.csv' with your actual file name
try:
    df = pd.read_csv('medicine_data.csv')
    print(f"✅ Dataset loaded from {'medicine_data.csv'}.")
except FileNotFoundError:
    print(f"❌ Error: {'medicine_data.csv'} not found. Please ensure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if the file doesn't exist
    data = {
        'product_name': ['Dolo 650', 'Amoxicillin 500mg', 'Omeprazole 20mg', 'Lisinopril 10mg', 'Cetirizine 10mg'],
        'sub_category': ['Pain Relief', 'Antibiotic', 'Antacid', 'Antihypertensive', 'Antihistamine'],
        'salt_composition': ['Paracetamol', 'Amoxicillin', 'Omeprazole', 'Lisinopril', 'Cetirizine'],
        'product_manufactured': ['Micro Labs', 'Cipla', 'Dr. Reddy\'s', 'Sun Pharma', 'Glenmark'],
        'medicine_desc': ['Used for fever and mild pain.', 'Fights bacterial infections.', 'Reduces stomach acid.', 'Treats high blood pressure.', 'Relieves allergy symptoms.'],
        'side_effects': ['Nausea, stomach pain.', 'Diarrhea, rash.', 'Headache, diarrhea.', 'Dizziness, cough.', 'Drowsiness, dry mouth.'],
        'drug_interactions': ['Interacts with Warfarin.', 'Interacts with Methotrexate.', 'Interacts with Diazepam.', 'Interacts with Diuretics.', 'Interacts with Alcohol.'],
        
    }
    df = pd.DataFrame(data)
    print("⚠️ Using dummy data for training as the CSV was not found.")


# 2️⃣ Combine all specified features into a single text column
feature_columns = [
    'sub_category', 'product_name', 'salt_composition', 
    'product_manufactured', 'medicine_desc', 'side_effects', 
    'drug_interactions'
]

# Ensure all columns exist before attempting to combine
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns in the dataset: {', '.join(missing_cols)}")
    # Handle the error or exit (for simplicity, we proceed with available columns)
    feature_columns = [col for col in feature_columns if col in df.columns]

# Concatenate available feature columns, handling potential missing values (NaN)
df['combined_text'] = df[feature_columns].fillna('').agg(' '.join, axis=1)
print("✅ Combined text features created.")


# 3️⃣ TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
print(f"✅ TF-IDF matrix created with {tfidf_matrix.shape[0]} documents and {tfidf_matrix.shape[1]} features.")

# 4️⃣ Function to recommend medicines
def recommend_medicines(user_input, top_n=5):
    """Recommends top_n medicines based on user input."""
    # Transform the user's input into a TF-IDF vector
    user_vec = vectorizer.transform([user_input])
    
    # Calculate cosine similarity between the user's vector and all medicine vectors
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Get the indices of the top_n most similar medicines
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    # Return the details of the top recommended medicines
    return df.iloc[top_indices][RECOMMENDATION_COLUMNS]

# 5️⃣ Example test
print("\n--- Training Complete ---")
print("\n🩺 Example Medicine Recommendations:\n")
# Example: User is looking for an antibiotic to treat a bacterial infection
print(recommend_medicines("antibiotic for bacterial infection with mild side effects"))

# 6️⃣ Save model + vectorizer + dataframe for later use
with open(MODEL_FILE, "wb") as f:
    pickle.dump((vectorizer, df), f)
print(f"\n✅ Model (vectorizer + dataframe) saved to {MODEL_FILE}")