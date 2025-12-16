import streamlit as st
import pickle 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csr import csr_matrix

# --- ⚠️ IMPORTANT: Load Model ---
# Ensure the 'medicine_recommender.pkl' file (created in the previous step) is in the same directory.
MODEL_FILE_NAME = "medicine_recommender.pkl"

try:
    with open(MODEL_FILE_NAME, "rb") as f:
        # vectorizer is the TF-IDF model object
        # df is the training dataframe containing the data and 'combined_text'
        vectorizer, df = pickle.load(f)
    
    # Pre-calculate the TF-IDF matrix for all combined texts for faster lookups
    # We must re-calculate this because the matrix was not saved in the .pkl file previously
    tfidf_matrix = vectorizer.transform(df['combined_text'])
    
    st.success(f"✅ Medicine Recommender Model loaded successfully! Dataset size: {df.shape[0]} records.")

except FileNotFoundError:
    st.error(f"❌ Error: Model file '{MODEL_FILE_NAME}' not found.")
    st.warning("Please run the training script first to generate the model file.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()


# Define the columns we want to display in the results
RECOMMENDATION_COLUMNS = ['product_name', 'salt_composition', 'product_manufactured']
    
# --- 1️⃣ Recommendation Logic Function ---
def recommend_medicines(user_input, top_n=5):
    """
    Recommends top_n medicines based on user input using TF-IDF and Cosine Similarity.
    """
    # 1. Transform user input into a TF-IDF vector
    user_vec = vectorizer.transform([user_input])
    
    # 2. Calculate cosine similarity against the pre-calculated matrix
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # 3. Get the indices of the top_n most similar medicines
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    # 4. Return the details of the top recommended medicines
    return df.iloc[top_indices][RECOMMENDATION_COLUMNS]

# -------------------------------------
# 2️⃣ STREAMLIT UI LAYOUT
# -------------------------------------

# CSS for a clean look (Modified from your original CSS)
st.markdown("""
    <style>
        .main-header {
            font-size: 42px; 
            margin-top: 20px;
            color: #1a5e37; /* Dark Green */
        }
        .subheader {
            font-size: 18px; 
            color: #666; 
            margin-bottom: 30px;
        }
        /* Custom card styling for results */
        .medicine-card {
            padding: 20px;
            margin-top: 15px;
            border-radius: 12px;
            background-color: #e8f5e9; /* Light Green background */
            border: 1px solid #c8e6c9; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-title {
            color: #388e3c; /* Green title */
            font-size: 24px;
            margin-bottom: 5px;
        }
        .card-details {
            font-size: 16px;
            margin-bottom: 10px;
        }
        .link-button {
            padding: 10px 18px; 
            background: #4CAF50; /* Primary Green button */
            color: white; 
            border-radius: 8px; 
            text-decoration: none;
            display: inline-block;
            transition: background-color 0.3s;
        }
        .link-button:hover {
            background-color: #388e3c;
        }
    </style>
""", unsafe_allow_html=True)


# --- DASHBOARD HEADER ---
st.markdown("""
    <h1 class="main-header">💊 Health AI: Medicine Recommender</h1>
    <p class="subheader">
        Just tell me your symptoms or the type of medicine you need, and I'll find the best options for you.
    </p>
""", unsafe_allow_html=True)


# --- SEARCH INPUT ---
user_query = st.text_input(
    "💬 What kind of medicine are you looking for?",
    placeholder="e.g., pain relief for headache, antibiotic for bacterial infection, medicine with Paracetamol..."
)


# --- RECOMMENDATION TRIGGER ---
if st.button("Find Medicines 🔍"):
    if user_query.strip() == "":
        st.warning("⚠️ Please enter a query about the medicine or condition!")
    else:
        st.info(f"Searching for: **{user_query}**")
        
        # Get Recommendations
        try:
            results = recommend_medicines(user_query, top_n=5)
            
            st.success("✨ Top 5 Recommended Medicines:")
            
            # --- BEAUTIFUL CARD LAYOUT ---
            for index, row in results.iterrows():
                # Handling potential NaN values for display
                product_name = str(row.get('product_name', 'N/A'))
                salt_composition = str(row.get('salt_composition', 'N/A'))
                manufacturer = str(row.get('product_manufactured', 'N/A'))
                link = str(row.get('link', '#'))
                
                st.markdown(f"""
                    <div class="medicine-card">
                        <h3 class="card-title">📦 {product_name}</h3>
                        <p class="card-details">
                            <b>Salt Composition:</b> {salt_composition} <br>
                            <b>Manufactured By:</b> {manufacturer}
                        </p>
                        <a href="{link}" target="_blank" class="link-button">
                            View Details 🔗
                        </a>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")