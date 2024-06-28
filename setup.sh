mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
