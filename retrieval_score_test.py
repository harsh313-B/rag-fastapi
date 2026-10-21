
from rag_engine import embedding_model, index, chunks

questions = [
    "What validates FastAPI request data?",
    "Who invented Python?",
]

for question in questions:
    print(f"\nQuestion: {question}")

    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(query_embedding, 2)

    for distance, idx in zip(distances[0], indices[0]):
        print(f"Squared L2 distance: {distance:.4f}")
        print(f"Text: {chunks[idx]}")
