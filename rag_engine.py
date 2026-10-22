
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Load the knowledge document
document_path = Path(__file__).parent / "knowledge" / "fastapi.txt"

loaded_documents = TextLoader(
    str(document_path),
    encoding="utf-8",
).load()


# 2. Split the document into individual facts
documents = [
    Document(
        page_content=line.strip(),
        metadata=loaded_documents[0].metadata,
    )
    for line in loaded_documents[0].page_content.splitlines()
    if line.strip()
]


# 3. Create embeddings and the FAISS vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents,
    embedding_model,
)


# 4. Load FLAN-T5
model_name = "google/flan-t5-small"

llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 5. Connect FLAN-T5 to LangChain
def generate_answer(prompt_value) -> str:
    prompt_text = prompt_value.to_string()

    inputs = llm_tokenizer(
        prompt_text,
        return_tensors="pt",
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50,
    )

    return llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


llm = RunnableLambda(generate_answer)


# 6. Define the prompt
prompt = PromptTemplate.from_template(
    """Answer the question using only the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)


# 7. Build the answer chain
answer_chain = prompt | llm

RELEVANCE_THRESHOLD = 1.0


# 8. Retrieve, check relevance, and answer
def ask_rag(question: str):
    results = vector_store.similarity_search_with_score(
        question,
        k=2,
    )

    if not results or results[0][1] > RELEVANCE_THRESHOLD:
        return (
            "I don't know based on the available knowledge.",
            [],
        )

    retrieved_documents = [
        document for document, score in results
    ]

    sources = [
        document.page_content
        for document in retrieved_documents
    ]

    context = "\n\n".join(sources)

    answer = answer_chain.invoke({
        "context": context,
        "question": question,
    })

    return answer, sources
