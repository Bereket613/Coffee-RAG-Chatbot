import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_COLLECTION_NAME = "coffee_disease"

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
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.collection = self._setup_collection()
        
        # Load models
        self.embed_model = self._load_embedding_model()
        self.tokenizer, self.llm_model = self._load_llm_model()
        
        print("✅ Coffee RAG Chatbot initialized successfully!")
    
    def _setup_collection(self):
        """Setup ChromaDB collection with error handling."""
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.vector_collection_name not in collection_names:
                logger.error(f"Collection '{self.vector_collection_name}' does not exist.")
                logger.info("Please run the vector database creation script first.")
                raise ValueError(f"Collection '{self.vector_collection_name}' not found")
            
            # Get the collection
            collection = self.client.get_collection(name=self.vector_collection_name)
            logger.info(f"Collection '{self.vector_collection_name}' loaded successfully.")
            
            # Verify it has data
            count = collection.count()
            logger.info(f"Collection contains {count} chunks.")
            
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
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            logger.info("Loading TinyLlama model...")
            
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
                torch_dtype=torch.float32,  # Use float32 for better CPU compatibility
                device_map=None,  # No device map for CPU
                low_cpu_mem_usage=True,  # Optimize CPU memory usage
                trust_remote_code=True
            )
            
            # Move to device
            llm_model = llm_model.to(self.device)
            
            # Set to evaluation mode
            llm_model.eval()
            
            logger.info("TinyLlama model loaded successfully.")
            return tokenizer, llm_model
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
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
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using RAG pipeline."""
        try:
            # Retrieve relevant context
            chunks = self.retrieve_chunks(question, k=3)
            
            if not chunks or "Error retrieving" in chunks[0]:
                return "I'm sorry, I couldn't retrieve relevant information to answer your question. Please make sure the vector database is properly set up."
            
            context = "\n".join(chunks)
            
            # Create enhanced prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            with torch.no_grad():  # Disable gradient calculation for inference
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024,
                    padding=True
                ).to(self.device)
                
                # Generate with optimized settings for CPU
                output = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    top_p=0.85,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
                
                # Decode the response
                full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the answer part
                answer = self._extract_answer(full_response, prompt)
                
                return answer
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for the LLM."""
        return f"""<|system|>
You are an expert Ethiopian coffee agronomist. Use the provided context to answer the question accurately and concisely.

Guidelines:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I don't have enough information about that specific topic."
- Keep answers focused and informative
- Use clear, professional language

Context:
{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""
    
    def _extract_answer(self, full_response: str, original_prompt: str) -> str:
        """Extract just the answer part from the full response."""
        # Remove the original prompt from the response
        if original_prompt in full_response:
            answer = full_response.replace(original_prompt, "").strip()
        else:
            answer = full_response.strip()
        
        # Clean up any extra tokens or tags
        answer = answer.replace("<|assistant|>", "").replace("<|system|>", "").replace("</s>", "")
        answer = answer.split("<|user|>")[0].strip() if "<|user|>" in answer else answer
        
        # If answer is empty, provide default response
        if not answer:
            return "I don't have enough information to answer that question based on my knowledge."
        
        return answer
    
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