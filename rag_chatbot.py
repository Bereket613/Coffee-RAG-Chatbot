import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import logging
import os
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_COLLECTION_NAME = "coffee_disease"
PERSIST_DIRECTORY = "vector_db"

# Use CPU for better performance if no GPU
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    # Optimize for CPU
    torch.set_num_threads(os.cpu_count() or 4)

print(f"Using device: {DEVICE}")

class CoffeeRAGChatbot:
    def __init__(self):
        self.device = DEVICE
        self.vector_collection_name = VECTOR_COLLECTION_NAME
        self.persist_directory = PERSIST_DIRECTORY
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._setup_collection()
        
        # Load models
        self.embed_model = self._load_embedding_model()
        self.tokenizer, self.llm_model = self._load_llm_model()
        
        print("✅ Coffee RAG Chatbot initialized successfully!")
    
    def _setup_collection(self):
        """Setup ChromaDB collection with error handling."""
        try:
            # List all available collections
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            logger.info(f"Available collections: {collection_names}")
            
            if self.vector_collection_name not in collection_names:
                logger.error(f"Collection '{self.vector_collection_name}' does not exist.")
                logger.info(f"Available collections: {collection_names}")
                logger.info("Please run 'create_vector_db.py' first to create the vector database.")
                raise ValueError(f"Collection '{self.vector_collection_name}' not found. Available: {collection_names}")
            
            # Get the collection
            collection = self.client.get_collection(name=self.vector_collection_name)
            
            # Verify it has data
            count = collection.count()
            logger.info(f"Collection '{self.vector_collection_name}' loaded successfully with {count} chunks.")
            
            if count == 0:
                logger.warning("Collection exists but is empty.")
            
            return collection
            
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _load_llm_model(self):
        """Load the LLM model optimized for CPU."""
        # Try Phi-3 mini first (better instruction following), fallback to TinyLlama
        model_name = "microsoft/phi-3-mini-4k-instruct"  # Much better at following instructions
        
        try:
            logger.info(f"Loading {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with CPU optimizations
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Move to device
            llm_model = llm_model.to(self.device)
            
            # Set to evaluation mode
            llm_model.eval()
            
            logger.info(f"{model_name} loaded successfully.")
            return tokenizer, llm_model
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {str(e)}")
            logger.info("Falling back to TinyLlama...")
            return self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load TinyLlama as fallback."""
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            logger.info(f"Loading fallback model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True
            )
            
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
            
            logger.info("TinyLlama fallback model loaded successfully.")
            return tokenizer, llm_model
            
        except Exception as e:
            logger.error(f"Fallback model also failed: {str(e)}")
            raise
    
    def retrieve_chunks(self, question: str, k: int = 3) -> List[str]:
        """Retrieve top-k relevant chunks from the vector database."""
        try:
            # Encode the question
            q_emb = self.embed_model.encode([question], convert_to_numpy=True)[0]
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=k
            )
            
            chunks = results['documents'][0] if results['documents'] else []
            logger.info(f"Retrieved {len(chunks)} relevant chunks.")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return ["Error retrieving information. Please try again."]
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a simple, clear prompt for better model understanding."""
        return f"""You are an expert Ethiopian coffee agronomist. Use the context below to answer the question clearly and concisely.

Guidelines:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I don't have enough information about that specific topic."
- Keep answers focused and informative
- Use clear, professional language

Context:
{context}

Question: {question}

Answer:"""
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using RAG pipeline."""
        try:
            # Retrieve relevant context
            chunks = self.retrieve_chunks(question, k=3)
            
            if not chunks or "Error retrieving" in chunks[0]:
                return "I'm sorry, I couldn't retrieve relevant information to answer your question. Please make sure the vector database is properly set up."
            
            context = "\n".join(chunks)
            
            # Create clean prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048,  # Increased for longer context
                    padding=True
                ).to(self.device)
                
                # Generate with optimized settings
                output = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=256,  # Increased for more detailed answers
                    do_sample=True,
                    top_p=0.90,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Increased to reduce repetition
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
                
                # Decode the response
                full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the answer part
                answer = self._extract_answer(full_response, prompt)
                
                return answer
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def _extract_answer(self, full_response: str, original_prompt: str) -> str:
        """Extract just the answer part from the full response."""
        # Remove the original prompt from the response
        if original_prompt in full_response:
            answer = full_response.replace(original_prompt, "").strip()
        else:
            answer = full_response.strip()
        
        # Clean up the answer - remove any repeated instructions or context
        lines = answer.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly part of the system prompt or context repetition
            if not line:
                continue
            if any(skip_word in line.lower() for skip_word in ['you are an expert', 'guidelines:', 'context:', 'question:', 'answer based only']):
                continue
            clean_lines.append(line)
        
        # Join the clean lines
        clean_answer = ' '.join(clean_lines).strip()
        
        # If we have a reasonably long answer, return it
        if len(clean_answer) > 10:
            return clean_answer
        
        # If answer is too short or empty, try a different extraction method
        if original_prompt in full_response:
            # Get everything after "Answer:"
            parts = full_response.split("Answer:")
            if len(parts) > 1:
                answer_part = parts[-1].strip()
                # Clean it up
                answer_part = answer_part.split('\n')[0].strip()  # Take first line after Answer:
                if answer_part and len(answer_part) > 5:
                    return answer_part
        
        return clean_answer if clean_answer else "I don't have enough information to answer that question based on my knowledge."
    
    def chat(self):
        """Start the interactive chat session."""
        print("\n" + "="*50)
        print("☕ Ethiopian Coffee RAG Chatbot")
        print("Type 'exit', 'quit', or 'bye' to end the conversation")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Bot: Thank you for chatting! Have a great day! ☕")
                    break
                
                if not user_input:
                    print("Bot: Please enter a question about Ethiopian coffee diseases.")
                    continue
                
                print("Bot: Thinking...")
                answer = self.generate_answer(user_input)
                print(f"Bot: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nBot: Chat session ended. Goodbye!")
                break
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error. Please try again.")
                logger.error(f"Chat error: {str(e)}")

def main():
    """Main function to run the chatbot."""
    try:
        chatbot = CoffeeRAGChatbot()
        chatbot.chat()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        print("❌ Failed to initialize the chatbot. Please make sure:")
        print("   1. The vector database collection exists")
        print("   2. You have run 'create_vector_db.py' first")
        print("   3. You have a stable internet connection for model download")
        print(f"   Error details: {str(e)}")

if __name__ == "__main__":
    main()