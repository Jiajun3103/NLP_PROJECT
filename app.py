import streamlit as st
import joblib
import re
import nltk
import os
import google.generativeai as genai
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Cinematch AI | Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Environment Setup ---
@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords')
    return set(stopwords.words('english')), PorterStemmer()

stop_words, ps = load_nltk_data()

# --- Gemini API Configuration ---
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🔑 API Key Missing: Please configure GEMINI_API_KEY in your secrets.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- Core Function: NLP Text Cleaning ---
def clean_text(text):
    if not text: return ""
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- Step 1: Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('TF-IDF and SVM/movie_genre_model.pkl')
        tfidf = joblib.load('TF-IDF and SVM/tfidf_vectorizer.pkl')
        return model, tfidf
    except:
        return None, None

model, tfidf = load_models()

# --- Step 2: Professional Sidebar with Chat ---
with st.sidebar:
    st.title("🤖 Cinematch Assistant")
    st.caption("Your AI guide for movie insights")
    
    # Chat System
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history container with fixed height
    chat_history_container = st.container(height=400)

    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Sticky chat input at the bottom of sidebar
    if chat_input := st.chat_input("Ask about the project..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with chat_history_container.chat_message("user"):
            st.markdown(chat_input)

        with chat_history_container.chat_message("assistant"):
            system_context = "You are a professional assistant for Cinematch AI. Techniques: TF-IDF and SVM. Guide users to input plots for genre prediction."
            response = gemini_model.generate_content(f"{system_context}\nUser question: {chat_input}")
            bot_response = response.text
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

    if any(keyword in (chat_input or "").lower() for keyword in ["use", "how", "help"]):
        st.info("💡 **Tip:** Paste a movie plot in the main area and hit 'Analyze'!")

# --- Step 3: Main Website Portal Interface ---
st.title("🎬 Cinematch AI")
st.subheader("Professional Movie Genre Classification")
st.write("Leveraging Support Vector Machines (SVM) and Natural Language Processing to categorize cinema with precision.")

# Main Input Area in a clean Card-like layout
with st.container(border=True):
    user_input = st.text_area("Enter Movie Synopsis", placeholder="Paste the movie plot or overview here...", height=200)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("✨ Analyze Genre", type="primary")

if predict_btn:
    if user_input:
        with st.spinner('Analyzing text patterns...'):
            if model and tfidf:
                cleaned_input = clean_text(user_input)
                input_vector = tfidf.transform([cleaned_input])
                prediction = model.predict(input_vector)[0]
                
                st.balloons()
                st.success(f"### 🎯 Predicted Genre: **{prediction.upper()}**")
                st.caption("Classification based on trained SVM Model (Accuracy: 40%)")
            else:
                st.error("System Error: Model files not found.")
    else:
        st.warning("⚠️ Please provide a movie plot to proceed.")

# --- Step 4: Project Information Section ---
st.divider()
st.header("Project Insights")
tab1, tab2, tab3 = st.tabs(["🛠 Technical Stack", "🎯 Objectives", "💼 Commercial Potential"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Core Algorithms**")
        st.write("- Feature Extraction: TF-IDF Vectorization")
        st.write("- Classifier: Linear Support Vector Machine (SVM)")
    with col_b:
        st.markdown("**NLP Libraries**")
        st.write("- NLTK (Stemming & Stopword removal)")
        st.write("- Scikit-Learn (Model pipeline)")

with tab2:
    st.write("1. Prepare an intelligent system using NLP techniques for document classification.")
    st.write("2. Automate the metadata tagging process for digital cinema libraries.")

with tab3:
    st.success("Cinematch AI is designed as a scalable API service for streaming platforms to auto-categorize large volumes of user-generated content or historical archives.")