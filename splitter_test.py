from langchain_text_splitters import RecursiveCharacterTextSplitter


with open("knowledge/fastapi.txt", "r") as file:
    text = file.read()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


chunks = splitter.split_text(text)


print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)


text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=3
)

chunks = splitter.split_text(text)

print("\n\n=== OVERLAP EXPERIMENT ===")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")