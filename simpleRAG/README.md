# RAG

Demo: To retrieve data from sample.txt and answer user questions

# Import libraries 
from sklearn.feature_extraction.text import TfidfVectorizer    # → converts text to numeric vectors
from sklearn.metrics.pairwise import cosine_similarity     # → measures similarity between vectors
import numpy as np        # → helps with ranking

# Load the document as raw text
with open("sample.txt", "r") as f:
    text = f.read()

# Chunking using period to avoid whole document being treated as one vector
chunks = text.split(". ")

# TF-IDF Vectorization - Term Frequency – Inverse Document Frequency - converts chunk into vector like [0.0, 0.12, 0.45, 0.0, 0.88, ...]
# Each position represents a word in vocabulary
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)    

# Retrievel logic - we use same vector, same vocabulary, same vector space
def ask_question(query):

# Cosine similarity - calculates similarity between every chunk vector, the query vector -- higher score = more similar
similarities = cosine_similarity(chunk_vectors, query_vector).flatten()

# Pick best chunk - retrievel ranking - find highest similarity score
best_chunk = chunks[np.argmax(similarities)]

# Return answer
return f"Based on documents: {best_chunk}"

# Note:
This is:
✔ Classical Information Retrieval
✔ Vector space model
✔ Keyword matching with weighting
✔ Cosine ranking
It is NOT semantic understanding.






