import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s|]', '', text)
        text = ' '.join(
            lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
        )
    else:
        text = ''
    return text

# Load the pickled files
with open('bow_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('bow_matrix.pkl', 'rb') as f:
    bow_matrix = pickle.load(f)

with open('merged_df.pkl', 'rb') as f:
    merged_df = pickle.load(f)

# Function to recommend similar movies based on input text
def recommend_similar(input_text, merged_df=merged_df, bow_matrix=bow_matrix, k=7):
    processed_input = preprocess_text(input_text)
    input_vec = vectorizer.transform([processed_input])
    cosine_similarities = cosine_similarity(input_vec, bow_matrix)
    similar_indices = cosine_similarities.argsort()[0, :(-k - 1):-1]
    recommended_movies = merged_df.loc[similar_indices, 'Name'].tolist()
    return recommended_movies

# Streamlit app
st.title('Hybrid Entertainment Recommendor System')

input_text = st.text_input('Enter a description or title of the Movie/TV Show/Anime:', 'Enter title or description')

if st.button('Recommend'):
    recommendations = recommend_similar(input_text)
    st.write(f"Recommended movies based on '{input_text}':")
    for movie in recommendations:
        st.write(movie)
