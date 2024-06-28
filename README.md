# Hybrid Recommendation System

This is a hybrid recommendation system that suggests movies or TV shows based on the input text. The system utilizes both movie/TV show names and descriptions to find the most similar content using CBOW and cosine similarity.

## Features

- Hybrid Recommendations: Recommends similar movies or TV shows based on both the name and description.
- Content Details: Provides information on whether the recommendation is a TV show, Hindi movie, or English movie.
- Text Preprocessing: Cleans and preprocesses text data to enhance recommendation accuracy.

## How It Works

1. Text Preprocessing: The input text and descriptions are converted to lowercase, punctuation is removed (except for '|'), and stop words are filtered out. Lemmatization is applied to standardize words.
2. TF-IDF Vectorization: The preprocessed text is transformed into TF-IDF vectors.
3. Cosine Similarity: The similarity between the input text and the available content is calculated using cosine similarity.
4. Recommendations: The system returns the top `k` most similar movies or TV shows based on the input text.
