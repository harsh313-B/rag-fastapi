from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

document_path = Path("knowledge/fastapi.txt")

text = document_path.read_text()

print(text)

chunks = [
    line.strip()
    for line in text.splitlines()
    if line.strip()
]

print("\nChunks:")

for i, chunk in enumerate(chunks):
    print(f"{i}: {chunk}")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

print("\nEmbedding shape:")
print(embeddings.shape)

dimension = embeddings.shape[1]

print("\nEmbedding dimension:")
print(dimension)

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nNumber of vectors in FAISS:")
print(index.ntotal)

query = "What does FastAPI use to validate data?"
query_embedding = embedding_model.encode(
    [query]
)

distances, indices = index.search(
    query_embedding,
    2
)

print("\nSearch results:")
print("Distances:", distances)
print("Indices:", indices)

llm_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

retrieved_chunks = [
    chunks[idx]
    for idx in indices[0]
]

context = "\n".join(retrieved_chunks)

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

inputs = llm_tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = llm_model.generate(
    **inputs,
    max_new_tokens=50
)

answer = llm_tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nAnswer:")
print(answer)