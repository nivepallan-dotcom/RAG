import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================
# Load OpenRouter API Key
# ==============================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY='your_key'")

# ==============================
# Load Document
# ==============================

with open("sample.txt", "r") as f:
    text = f.read()

# ==============================
# Chunking
# ==============================

chunks = text.split(". ")

# ==============================
# Vectorization (TF-IDF)
# ==============================

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
chunk_vectors = vectorizer.fit_transform(chunks)

# ==============================
# Retrieval Function
# ==============================

def retrieve(query, top_k=2):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(chunk_vectors, query_vector).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_chunks = [chunks[i] for i in top_indices]
    return " ".join(retrieved_chunks)

# ==============================
# LLM Call via OpenRouter
# ==============================

def call_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "Answer only using provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    return response.json()["choices"][0]["message"]["content"]

# ==============================
# Main Loop
# ==============================

print("Lightweight RAG + OpenRouter Ready!")

while True:
    question = input("\nAsk: ")

    if question.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant context
    context = retrieve(question)

    # Build grounded prompt
    prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """

    answer = call_llm(prompt)

    print("\nAnswer:", answer)
