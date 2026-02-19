# Install light weight libraries 
pip install scikit-learn numpy requests    

-> scikit-learn → TF-IDF + cosine similarity
-> numpy → vector operations
-> requests → call external LLM API

# Set API key for Openrouter
export OPENROUTER_API_KEY="your_key"

# Load document
with open("sample.txt") as f: ----> In prod, this could be S3, Database, Document store

# Chunking
chunks = text.split(". ")

# Vectorization
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks) ----> Each chunk becomes a vector like: [0.2, 0.0, 0.8, 0.3, ...]

# User query converted to vector
query_vector = vectorizer.transform([query])

# Cosine similarity 
similarities = cosine_similarity(chunk_vectors, query_vector)  --> Higher score = more relevant --> retriever

-> Retrieved relevant chunks

# Prompt construction / context augmentation 
prompt = f"""
Context:
{context}

Question:
{question}
"""    
-> Combine retrieved knowledge + user question --> LLM input

# Call LLM
requests.post("https://openrouter.ai/api/v1/chat/completions")

-> We sent :
Context:
Turbine A experienced repeated overheating...
Root cause was improper lubrication...

Question:
What caused overheating?

# Production notes
sample.txt	      -------> S3 / Document DB
TF-IDF	          -------> OpenSearch / Vector DB
Cosine similarity	-------> Vector search engine
OpenRouter API	  -------> Amazon Bedrock
Python script	    -------> Lambda / ECS service

 
















