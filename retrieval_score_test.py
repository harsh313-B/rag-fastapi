
from rag_engine import vector_store

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

    for document, score in results:
        print(f"Squared L2 distance: {score:.4f}")
        print(f"Text: {document.page_content}")
