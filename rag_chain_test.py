
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Load the knowledge document
document_path = Path("knowledge/fastapi.txt")
loaded_documents = TextLoader(
    str(document_path),
    encoding="utf-8",
).load()

from langchain_core.documents import Document

documents = [
    Document(
        page_content=line.strip(),
        metadata=loaded_documents[0].metadata,
    )
    for line in loaded_documents[0].page_content.splitlines()
    if line.strip()
]


# 2. Create embeddings and the FAISS vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents,
    embedding_model,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# 3. Load FLAN-T5
model_name = "google/flan-t5-small"

llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 4. Connect FLAN-T5 to LangChain
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


# 5. Define the prompt
prompt = PromptTemplate.from_template(
    """Answer the question using only the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)


# 6. Format retrieved documents
def format_docs(documents) -> str:
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# 7. Build the LangChain RAG pipeline
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)


# 8. Preserve the existing API interface
def ask_rag(question: str):
    retrieved_documents = retriever.invoke(question)
    sources = [
        document.page_content
        for document in retrieved_documents
    ]

    answer = rag_chain.invoke(question)

    return answer, sources
