
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load the existing FLAN-T5 model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 2. Create a Python function that generates an answer
def generate_answer(prompt_text: str) -> str:
    inputs = tokenizer(str(prompt_text), return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=30
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# 3. Create the LangChain prompt
prompt = PromptTemplate.from_template(
    """Answer using only the context.

Context: {context}
Question: {question}
Answer:"""
)


# 4. Connect the prompt to our Python function
llm = RunnableLambda(generate_answer)
chain = prompt.pipe(llm)


# 5. Test the complete chain
result = chain.invoke({
    "context": "FastAPI uses Pydantic for request data validation.",
    "question": "What validates FastAPI request data?"
})

print("Answer:", result)
