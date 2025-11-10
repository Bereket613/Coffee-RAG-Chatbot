import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_COLLECTION_NAME = "coffee_disease"

# --- Load Chroma DB ---
client = chromadb.Client()
collection = client.get_collection(name=VECTOR_COLLECTION_NAME)

# --- Load Embedding Model ---
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Load Local LLaMA-2-7B Chat Model ---
model_name = "meta-llama/Llama-2-7b-hf"  # small or 7B
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # automatically uses GPU
    torch_dtype=torch.float16
)

print("✅ LLaMA-2 loaded locally")

# --- Retrieve Top-K Chunks ---
def retrieve_chunks(question, k=3):
    q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k
    )
    return results['documents'][0]  # list of top-k relevant chunks

# --- Generate Answer ---
def generate_answer(question):
    chunks = retrieve_chunks(question, k=3)
    context = "\n".join(chunks)

    prompt = f"""
You are an expert Ethiopian coffee agronomist. 
Use the following context to answer the question. 
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()
