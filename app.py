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

# --- Environment Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# --- Gemini API Configuration ---

# Load environment variables
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # 本地开发时才加载 .env
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

st.write("API Key 已加载:", bool(api_key))
GEMINI_API_KEY = api_key

# Check if the API Key is loaded successfully
if not GEMINI_API_KEY:
    st.error(f"Error: GEMINI_API_KEY not found. Attempted path: {env_path}")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using stable model version
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Core Function: NLP Text Cleaning ---
def clean_text(text):
    if not text: return ""
    # 1. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    # 2. Convert to lowercase
    text = text.lower()
    # 3. Tokenization and Cleaning
    words = text.split()
    # 4. Remove stopwords and apply Stemming
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- Step 1: Model Loading ---
try:
    # Ensure the model files are located within the correct subdirectory
    model = joblib.load('TF-IDF and SVM/movie_genre_model.pkl')
    tfidf = joblib.load('TF-IDF and SVM/tfidf_vectorizer.pkl')
except:
    st.error("Error: Model files not found. Please check the 'TF-IDF and SVM' folder.")

# --- Step 2: Integrated Gemini Chatbot (Sidebar) ---
# Requirement: A chatbot that allows users to ask questions about the project [cite: 38]
st.sidebar.title("🤖 Movie Genre Classification Assistant ")
st.sidebar.write("I can answer questions about this portal and guide you through its features.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if chat_input := st.sidebar.chat_input("Type your question (e.g., How to use this?)"):
    # 1. Save and display user input
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with st.sidebar.chat_message("user"):
        st.markdown(chat_input)

    # 2. Call Gemini (with System Prompt constraints)
    with st.sidebar.chat_message("assistant"):
        # System Context defines the bot's behavior and knowledge [cite: 39]
        system_context = (
            "You are a helpful assistant for the 'Movie Classifier AI' portal developed by the BAXI 3413 group. "
            "Project info: Techniques used include TF-IDF and SVM. Model accuracy is 40% due to multi-label complexity. "
            "The dataset is sourced from Kaggle. "
            "If asked how to use the site, guide the user to the text area on the main page to paste a movie description. "
            "Be professional and always try to guide the user to the portal's core features."
        )
        
        response = gemini_model.generate_content(f"{system_context}\nUser question: {chat_input}")
        bot_response = response.text
        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # 3. Dynamic User Guidance [cite: 40]
    if any(keyword in chat_input.lower() for keyword in ["use", "how", "predict", "predict"]):
        st.sidebar.warning("💡 **Quick Guide:** Paste the plot summary in the main text box and click 'Predict Genre'.")

# --- Step 3: Main Website Portal Interface ---
st.title("🎬 Movie Genre Classification Portal")
st.write("An intelligent system based on Natural Language Processing (NLP).")

# User input for prediction
user_input = st.text_area("Paste Movie Plot/Overview (English Only):", height=150)

if st.button("Predict Genre"):
    if user_input:
        cleaned_input = clean_text(user_input)
        input_vector = tfidf.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"🎯 Predicted Genre: **{prediction}**")
    else:
        st.warning("Please enter some text before clicking predict.")

# --- Step 4: Project Information Section ---
# Requirement: Information about the selected project on the website [cite: 37]
st.divider()
st.subheader("About This Project")
col1, col2 = st.columns(2)
with col1:
    st.write("**Technical Stack:**")
    st.write("- NLP Library: NLTK (Stemming, Stopwords)")
    st.write("- Feature Extraction: TF-IDF")
    st.write("- Model: Linear Support Vector Machine (SVM)")
with col2:
    st.write("**Project Objectives:**")
    st.write("- Prepare an intelligent NLP system")
    st.write("- Automation of movie genre tagging")
    st.write("- Potential for commercialization ")