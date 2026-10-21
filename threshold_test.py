from rag_engine import embedding_model, index, chunks

THRESHOLD = 1.0

questions = [
    "What validates FastAPI request data?",
    "Who invented Python?",
]

for question in questions:
    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(query_embedding, 2)

    print(f"\nQuestion: {question}")

    if distances[0][0] > THRESHOLD:
        print("Decision: Reject — insufficiently relevant context")
        continue

    print("Decision: Accept — pass context to the LLM")
    for idx in indices[0]:
        print(f"- {chunks[idx]}")