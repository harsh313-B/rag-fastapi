from langchain_community.document_loaders import TextLoader


loader = TextLoader("knowledge/fastapi.txt")

documents = loader.load()

print("Number of documents:", len(documents))

for document in documents:
    print("\n--- DOCUMENT ---")
    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)