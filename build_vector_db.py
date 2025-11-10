import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- Folders ---
CHUNK_FOLDER = "data/chunks"

# --- Chroma Setup ---
client = chromadb.Client()
collection_name = "coffee_disease"
# Delete old collection if exists
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
collection = client.create_collection(name=collection_name)

# --- Embedding Model ---
# MiniLM model (small, fast, multilingual)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_chunks():
    texts = []
    ids = []
    for file in os.listdir(CHUNK_FOLDER):
        if file.endswith("_chunks.txt"):
            path = os.path.join(CHUNK_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                for i, chunk in enumerate(content.split("\n\n==== CHUNK ====\n\n")):
                    if chunk.strip():
                        texts.append(chunk.strip())
                        ids.append(f"{file}_{i}")
    return ids, texts

# --- Create embeddings and store ---
ids, texts = get_chunks()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

for i, emb in enumerate(embeddings):
    collection.add(
        documents=[texts[i]],
        ids=[ids[i]],
        embeddings=[emb.tolist()]
    )

print(f"✅ Vector DB created with {len(texts)} chunks.")
