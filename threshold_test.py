
from rag_engine import vector_store, RELEVANCE_THRESHOLD

questions = [
    "What validates FastAPI request data?",
    "Who invented Python?",
]

for question in questions:
    results = vector_store.similarity_search_with_score(
        question,
        k=2,
    )

    print(f"\nQuestion: {question}")

    if not results or results[0][1] > RELEVANCE_THRESHOLD:
        print("Decision: Reject — insufficiently relevant context")
        continue

    print("Decision: Accept — pass context to the LLM")

    for document, score in results:
        print(f"Distance: {score:.4f}")
        print(f"Text: {document.page_content}")
