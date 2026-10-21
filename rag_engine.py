from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Load the knowledge document

document_path = Path("knowledge/fastapi.txt")

text = document_path.read_text()


# 2. Create chunks

chunks = [
    line.strip()
    for line in text.splitlines()
    if line.strip()
]


# 3. Load embedding model

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create embeddings

embeddings = embedding_model.encode(chunks)


# 5. Create FAISS index

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# 6. Load the LLM

llm_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)


# 7. RAG function

def ask_rag(question: str):

    # Convert question into an embedding
    query_embedding = embedding_model.encode(
        [question]
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        2
    )

    #  Reject questions with insufficiently relevant context
    if distances[0][0] > 1.0:
        return (
            "I don't know based on the available knowledge.",
            []
        )

    # Retrieve matching chunks
    retrieved_chunks = [
        chunks[idx]
        for idx in indices[0]
    ]

    # Combine retrieved chunks
    context = "\n".join(retrieved_chunks)

    # Create prompt
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Tokenize
    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Generate answer
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50
    )

    # Convert tokens to text
    answer = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, retrieved_chunks