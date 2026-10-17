from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


texts = [
    "FastAPI is a modern Python framework for building APIs.",
    "FastAPI supports HTTP methods such as GET, POST, PUT and DELETE.",
    "FastAPI uses Pydantic for request data validation.",
    "FastAPI automatically generates interactive API documentation.",
    "FastAPI can be used to build REST APIs."
]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_texts(
    texts,
    embeddings
)


query = "What is used for request data validation?"

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.invoke(query)


print("\nSearch Results:")

for i, document in enumerate(results):
    print(f"\nResult {i}:")
    print("Type:", type(document).__name__)
    print(document.page_content)
    print("Metadata:", document.metadata)