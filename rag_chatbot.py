import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import logging
import os
import hashlib
from typing import List, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_COLLECTION_NAME = "coffee_disease"
PERSIST_DIRECTORY = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Use CPU for better performance if no GPU
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    # Optimize for CPU
    torch.set_num_threads(os.cpu_count() or 4)

print(f"Using device: {DEVICE}")

class EmbeddingHelper:
    """Helper class for handling embeddings and vector database operations."""
    
    def __init__(self, persist_directory: str = PERSIST_DIRECTORY):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = VECTOR_COLLECTION_NAME
        
    def setup_collection(self) -> chromadb.Collection:
        """Setup or get existing ChromaDB collection."""
        try:
            # Try to get existing collection first
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded successfully.")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection '{self.collection_name}'...")
            return self.chroma_client.create_collection(name=self.collection_name)
    
    def get_embedding(self, texts: List[str]):
        """Get embeddings for texts."""
        return self.embedder.encode(texts, convert_to_tensor=True)
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as ID."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def store_embeddings(self, chunks: List[str], metadatas: List[dict] = None):
        """Store chunks and their embeddings in the vector database."""
        if not chunks:
            logger.warning("No chunks to store.")
            return
        
        collection = self.setup_collection()
        embeddings = self.get_embedding(chunks)
        
        # Generate IDs and metadata
        ids = [self.get_text_hash(chunk) for chunk in chunks]
        if metadatas is None:
            metadatas = [{"text": chunk} for chunk in chunks]
        
        # Store in batches for better performance
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_ids = ids[i:end_idx]
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            collection.upsert(
                embeddings=batch_embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Stored batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
    
    def get_top_n_results(self, query: str, n: int = 3) -> List[str]:
        """Retrieve top n relevant chunks for a query."""
        try:
            collection = self.setup_collection()
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.error(f"Error retrieving results: {str(e)}")
            return []

class RAGModel:
    """Main RAG model for generating answers."""
    
    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct"):
        self.device = DEVICE
        self.model_name = model_name
        self.tokenizer, self.llm_model = self._load_llm_model()
        logger.info(f"RAG model initialized with {model_name}")
    
    def _load_llm_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load the LLM model with fallback options."""
        model_priority_list = [
            "microsoft/phi-3-mini-4k-instruct",  # Best for instruction following
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Good fallback
            "microsoft/DialoGPT-medium"  # Lightweight fallback
        ]
        
        for model_name in model_priority_list:
            try:
                logger.info(f"Attempting to load {model_name}...")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                llm_model = llm_model.to(self.device)
                llm_model.eval()
                
                logger.info(f"Successfully loaded {model_name}")
                return tokenizer, llm_model
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        raise Exception("All model loading attempts failed")
    
    def create_prompt(self, question: str, context: str) -> str:
        """Create an optimized prompt for better answer generation."""
        return f"""Context:
{context}

Question: {question}

Instructions: Based on the context above, provide a clear and concise answer. If the information is not in the context, say you don't know.

Answer:"""
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using context and question."""
        try:
            prompt = self.create_prompt(question, context)
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).to(self.device)
                
                output = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
                
                full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                answer = self._clean_response(full_response, prompt)
                
                return answer
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    
    def _clean_response(self, full_response: str, original_prompt: str) -> str:
        """Clean and extract the answer from the full response."""
        # Remove the prompt from response
        if original_prompt in full_response:
            answer = full_response.replace(original_prompt, "").strip()
        else:
            answer = full_response.strip()
        
        # Remove any instruction repetitions
        clean_answer = re.sub(r'(Context:|Question:|Instructions:).*?Answer:', '', answer, flags=re.DOTALL)
        clean_answer = clean_answer.strip()
        
        # Take only the first few sentences
        sentences = re.split(r'[.!?]+', clean_answer)
        if sentences:
            clean_answer = '. '.join(sentences[:2]).strip() + '.'
        
        # Final cleanup
        clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
        
        if not clean_answer or len(clean_answer) < 10:
            return "I don't have enough information to answer that question based on the available context."
        
        return clean_answer

class CoffeeRAGChatbot:
    """Main chatbot class integrating embedding helper and RAG model."""
    
    def __init__(self):
        self.device = DEVICE
        
        # Initialize components
        self.embedding_helper = EmbeddingHelper()
        self.rag_model = RAGModel()
        
        logger.info("✅ Coffee RAG Chatbot initialized successfully!")
    
    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant context for the question."""
        try:
            chunks = self.embedding_helper.get_top_n_results(question, top_k)
            if not chunks:
                return "No relevant information found."
            
            return "\n".join(chunks)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving information."
    
    def generate_answer(self, question: str) -> str:
        """Generate answer for the question using RAG pipeline."""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(question)
            
            if "error" in context.lower() or "no relevant" in context.lower():
                return "I'm sorry, I couldn't find relevant information to answer your question. Please try rephrasing or ask about Ethiopian coffee diseases."
            
            # Generate answer using RAG model
            answer = self.rag_model.generate_answer(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    
    def chat(self):
        """Start interactive chat session."""
        print("\n" + "="*60)
        print("☕ Ethiopian Coffee Disease Research Assistant")
        print("="*60)
        print("Ask me about Ethiopian coffee diseases, symptoms, treatments,")
        print("prevention methods, or any related agricultural practices.")
        print("\nType 'exit', 'quit', or 'bye' to end the conversation")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n🤖 Bot: Thank you for using the Ethiopian Coffee Research Assistant! Have a great day! ☕")
                    break
                
                if not user_input:
                    print("🤖 Bot: Please enter a question about Ethiopian coffee diseases.")
                    continue
                
                print("🤖 Bot: Researching...", end=" ", flush=True)
                answer = self.generate_answer(user_input)
                print(f"\r🤖 Bot: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n🤖 Bot: Sorry, I encountered an error. Please try again.")
                logger.error(f"Chat error: {str(e)}")

def check_vector_database() -> bool:
    """Check if vector database exists and has data."""
    try:
        embedding_helper = EmbeddingHelper()
        collection = embedding_helper.setup_collection()
        count = collection.count()
        
        if count > 0:
            logger.info(f"Vector database found with {count} chunks.")
            return True
        else:
            logger.warning("Vector database exists but is empty.")
            return False
    except Exception as e:
        logger.error(f"Error checking vector database: {str(e)}")
        return False

def main():
    """Main function to run the chatbot."""
    print("🚀 Initializing Ethiopian Coffee RAG Chatbot...")
    
    # Check if vector database exists
    if not check_vector_database():
        print("❌ Vector database not found or empty!")
        print("Please make sure to:")
        print("1. Run 'create_vector_db.py' first to create the vector database")
        print("2. Ensure you have text chunks in 'data/chunks/' folder")
        print("3. Verify the vector database was created successfully")
        return
    
    try:
        # Initialize and start chatbot
        chatbot = CoffeeRAGChatbot()
        chatbot.chat()
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        print("❌ Failed to initialize the chatbot. Error details:")
        print(f"   {str(e)}")

if __name__ == "__main__":
    main()