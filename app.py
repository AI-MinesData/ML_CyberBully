import streamlit as st
import pickle
import string
import nltk
import spacy
import contractions
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Define stopwords but keep negations
stop_words = set(stopwords.words('english'))
negation_words = {"not", "no", "nor", "never", "neither", "none", "nothing", "nobody", "nowhere",
                  "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", 
                  "wouldn't", "shouldn't", "couldn't", "mustn't", "don't", "doesn't", "didn't", 
                  "can't", "won't"}
filtered_stop_words = stop_words - negation_words  # Keep negations

# Define text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = contractions.fix(text)  # Expand contractions (e.g., "aren't" → "are not")
    text = nltk.word_tokenize(text)  # Tokenization

    y = [i for i in text if i.isalnum()]  # Remove punctuations

    text = [i for i in y if i not in filtered_stop_words]  # Remove stopwords

    # Use spaCy for lemmatization
    doc = nlp(" ".join(text))
    text = [token.lemma_ for token in doc]  

    return " ".join(text)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

# Streamlit App
st.title("CYBERBULLYING DETECTION")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a message.")
    else:
        # Apply preprocessing
        transformed_sms = transform_text(input_sms)

        # Debugging: Show transformed text
        st.write("Transformed Input:", transformed_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Predict probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector_input)
            st.write(f"Prediction Probabilities: {proba}")

        # Display prediction
        labels = ["Not-Cyberbullying", "Ethnicity/Race Bullying", "Gender/Sexual Bullying", 
                  "Religion Bullying", "Age Bullying"]

        st.header(labels[result])
