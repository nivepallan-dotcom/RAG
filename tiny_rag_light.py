from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load document
with open("sample.txt", "r") as f:
    text = f.read()

# Simple chunking
chunks = text.split(". ")

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def ask_question(query):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(chunk_vectors, query_vector).flatten()
    best_chunk = chunks[np.argmax(similarities)]

    return f"Based on documents: {best_chunk}"

print("Ultra-Lightweight RAG Ready!")

while True:
    q = input("\nAsk: ")
    print("\nAnswer:", ask_question(q))
