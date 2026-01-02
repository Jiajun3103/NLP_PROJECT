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
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API Key is loaded successfully
if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using stable model version
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

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
# Requirement: A chatbot that allows users to ask questions about the project 
st.sidebar.title("🤖 Movie Genre Classification Assistant ")
st.sidebar.write("I can answer questions about this portal and guide you through its features.")

# 1. Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Create a scrollable container for the chat history
# This keeps the history separated from the input bar
chat_history_container = st.sidebar.container()

# 3. Display history inside that container
with chat_history_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 4. Chat input box - This widget will naturally stay at the bottom of the sidebar
if chat_input := st.sidebar.chat_input("Type your question..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": chat_input})
    with chat_history_container.chat_message("user"):
        st.markdown(chat_input)

    # Generate Response
    with chat_history_container.chat_message("assistant"):
        system_context = (
            "You are a helpful assistant for the 'Movie Classifier AI' portal. "
            "Techniques: TF-IDF and SVM. "
            "Guide users to the main page text area for predictions. "
        )
        
        # Call Gemini model
        response = gemini_model.generate_content(f"{system_context}\nUser question: {chat_input}")
        bot_response = response.text
        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Requirement: Dynamic User Guidance
    if any(keyword in chat_input.lower() for keyword in ["use", "how", "predict"]):
        st.sidebar.warning("💡 **Quick Guide:** Paste the movie description in the main text box and click 'Predict Genre'.")

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