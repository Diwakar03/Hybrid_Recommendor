import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the lemmatizer and stop words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char if char.isalnum() or char.isspace() or char == '|'])
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    else:
        text = ''
    return text


# Apply preprocessing to the descriptions and names
merged_df['Description'] = merged_df['Description'].apply(preprocess_text)
merged_df['Name'] = merged_df['Name'].apply(preprocess_text)
merged_df['Text'] = merged_df['Name'] + ' ' + merged_df['Description']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(merged_df['Text'])

# Function to recommend similar movies based on input text
def recommend_similar(input_text, merged_df=merged_df, tfidf_matrix=tfidf_matrix, k=7):
    processed_input = preprocess_text(input_text)
    input_vec = vectorizer.transform([processed_input])
    cosine_similarities = cosine_similarity(input_vec, tfidf_matrix)
    similar_indices = cosine_similarities.argsort()[0, :(-k - 1):-1]
    recommended_movies = merged_df.loc[similar_indices, 'Name'].tolist()
    return recommended_movies

# Streamlit app
st.title('Recommendation System')

input_text = st.text_input('Enter a  name or description:')
if input_text:
    recommendations = recommend_similar(input_text)
    st.write(f"Recommended movies based on '{input_text}':")
    for movie in recommendations:
        st.write(movie)
