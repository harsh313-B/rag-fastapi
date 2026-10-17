from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "FastAPI is a modern Python framework for building APIs.",
    "FastAPI uses Pydantic for request data validation.",
    "FastAPI automatically generates interactive API documentation."
]


vectors = embeddings.embed_documents(texts)


print("Number of vectors:", len(vectors))
print("Embedding dimension:", len(vectors[0]))

print("\nFirst vector:")
print(vectors[0])