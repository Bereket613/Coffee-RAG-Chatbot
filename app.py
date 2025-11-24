import gradio as gr
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import os
import time
from typing import List
import re

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_COLLECTION_NAME = "coffee_disease"
PERSIST_DIRECTORY = "vector_db"

class CoffeeRAGChatbot:
    def __init__(self):
        self.device = DEVICE
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self._setup_collection()
        
        # Load embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Load text generation model
        self.generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            device=-1 if self.device == "cpu" else 0,
            torch_dtype=torch.float32
        )
        
        print("✅ Coffee RAG Chatbot initialized successfully!")
    
    def _setup_collection(self):
        """Setup ChromaDB collection with error handling."""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if VECTOR_COLLECTION_NAME not in collection_names:
                print(f"❌ Collection '{VECTOR_COLLECTION_NAME}' does not exist!")
                return None
            
            collection = self.client.get_collection(name=VECTOR_COLLECTION_NAME)
            count = collection.count()
            print(f"✅ Collection loaded with {count} chunks")
            
            if count == 0:
                print("⚠️ Collection exists but is empty!")
            
            return collection
            
        except Exception as e:
            print(f"❌ Error setting up collection: {str(e)}")
            return None
    
    def retrieve_chunks(self, question: str, k: int = 3) -> List[str]:
        """Retrieve top-k relevant chunks from the vector database."""
        try:
            if not self.collection:
                return []
                
            q_emb = self.embed_model.encode([question], convert_to_numpy=True)[0]
            
            results = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=k
            )
            
            chunks = results['documents'][0] if results['documents'] else []
            print(f"🔍 Retrieved {len(chunks)} relevant chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using RAG pipeline."""
        try:
            # Retrieve relevant context
            chunks = self.retrieve_chunks(question, k=3)
            
            if not chunks:
                return "I couldn't find relevant information about that in my database. Please try asking about Ethiopian coffee diseases, symptoms, or treatments."
            
            # Combine chunks into context
            context = "\n".join(chunks)
            
            # Create a clean, simple prompt
            prompt = f"""Based on this information about Ethiopian coffee diseases:

{context}

Question: {question}

Please provide a clear answer based only on the information above. If you don't know, say so.

Answer:"""
            
            print("🤖 Generating answer...")
            
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract the generated text
            full_text = response[0]['generated_text']
            
            # Extract just the answer part (after the prompt)
            if prompt in full_text:
                answer = full_text.replace(prompt, "").strip()
            else:
                answer = full_text.strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        # Remove any repeated prompts or instructions
        clean_answer = re.sub(r'.*Question:.*', '', answer)
        clean_answer = re.sub(r'.*Based on this information.*', '', clean_answer)
        clean_answer = re.sub(r'.*Please provide.*', '', clean_answer)
        
        # Remove extra whitespace
        clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
        
        # If answer is too short, provide default
        if len(clean_answer) < 10:
            return "I don't have enough specific information to answer that question based on my knowledge base."
        
        return clean_answer

# Initialize the chatbot
chatbot = CoffeeRAGChatbot()

def detect_gradio_version():
    """Detect which Gradio version features are available"""
    version_info = {
        'has_theme_in_blocks': False,
        'has_messages_type': False,
        'has_show_copy_button': False
    }
    
    # Test for theme in Blocks
    try:
        with gr.Blocks(theme=gr.themes.Soft()):
            pass
        version_info['has_theme_in_blocks'] = True
    except:
        pass
    
    # Test for messages type
    try:
        gr.Chatbot(type="messages")
        version_info['has_messages_type'] = True
    except:
        pass
    
    # Test for show_copy_button
    try:
        gr.Chatbot(show_copy_button=True)
        version_info['has_show_copy_button'] = True
    except:
        pass
    
    print(f"📊 Detected Gradio features: {version_info}")
    return version_info

# Detect Gradio version
gradio_features = detect_gradio_version()

def create_gradio_interface():
    """Create the Gradio interface that works with ANY version"""
    
    # Universal Blocks creation that works with all versions
    blocks_kwargs = {"title": "☕ Ethiopian Coffee Disease Assistant"}
    
    # Only add theme if supported
    if gradio_features['has_theme_in_blocks']:
        blocks_kwargs["theme"] = gr.themes.Soft()
    
    with gr.Blocks(**blocks_kwargs) as demo:
        
        # Header with inline styling
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #8B4513, #A0522D); 
                       border-radius: 15px; color: white; margin-bottom: 20px; font-family: Arial, sans-serif;">
                <h1 style="margin: 0;">☕ Ethiopian Coffee Disease Research Assistant</h1>
                <h3 style="margin: 10px 0 0 0; font-weight: normal;">Your AI companion for Ethiopian coffee disease knowledge</h3>
            </div>
            """
        )
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                # Universal Chatbot creation
                chatbot_kwargs = {
                    "value": [],
                    "label": "Coffee Disease Chat",
                    "height": 500
                }
                
                # Only add advanced features if supported
                if gradio_features['has_messages_type']:
                    chatbot_kwargs["type"] = "messages"
                    chatbot_kwargs["placeholder"] = "Chat with your coffee disease expert..."
                
                if gradio_features['has_show_copy_button']:
                    chatbot_kwargs["show_copy_button"] = True
                
                chatbot_interface = gr.Chatbot(**chatbot_kwargs)
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Type your question about Ethiopian coffee diseases",
                        placeholder="e.g., What is coffee leaf rust? How to prevent coffee berry disease?",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    restart_btn = gr.Button("Restart", variant="secondary")
            
            # Sidebar with information and examples
            with gr.Column(scale=1):
                gr.Markdown("### 💡 About This Assistant")
                gr.Markdown("""
                <div style="background: #f8f4e9; padding: 15px; border-radius: 10px; border-left: 4px solid #8B4513;">
                This AI assistant specializes in Ethiopian coffee diseases and can help you with:
                
                • **Disease Identification**
                • **Symptoms & Causes**  
                • **Treatment Methods**
                • **Prevention Strategies**
                • **Agricultural Best Practices**
                
                <em>Powered by RAG technology with specialized knowledge base</em>
                </div>
                """)
                
                gr.Markdown("### 🔍 Sample Questions")
                examples = gr.Examples(
                    examples=[
                        "What is coffee leaf rust and how does it affect plants?",
                        "How can I prevent coffee berry disease naturally?",
                        "What are the main symptoms of coffee wilt disease?",
                        "Best organic treatments for coffee diseases",
                        "How to identify different coffee leaf diseases?",
                        "What causes coffee leaf spot and how to treat it?"
                    ],
                    inputs=msg,
                    label="Click any question to try:"
                )
                
                gr.Markdown("### 📊 System Information")
                status = "✅ Ready" if chatbot.collection else "❌ Setup Required"
                chunk_count = chatbot.collection.count() if chatbot.collection else 0
                gr.Markdown(f"""
                <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #4a90e2;">
                • **Device**: {DEVICE.upper()}<br>
                • **AI Model**: DialoGPT-medium<br>
                • **Knowledge Base**: {chunk_count} documents<br>
                • **Status**: {status}
                </div>
                """)
        
        # UNIVERSAL CHAT HANDLER - works with all Gradio versions
        def respond(message, chat_history):
            """Handle user message - universal compatibility"""
            if not message.strip():
                return chat_history
            
            if not chatbot.collection:
                # Handle both message formats
                try:
                    # Try new format first
                    if gradio_features['has_messages_type']:
                        chat_history.append({"role": "user", "content": message})
                        chat_history.append({"role": "assistant", "content": "❌ Error: Please run 'create_vector_db.py' first."})
                    else:
                        # Fallback to old format
                        chat_history.append((message, "❌ Error: Please run 'create_vector_db.py' first."))
                except:
                    # Ultimate fallback
                    chat_history.append((message, "❌ Error: Please run 'create_vector_db.py' first."))
                return chat_history
            
            # Show typing indicator
            try:
                if gradio_features['has_messages_type']:
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": "🤖 Researching..."})
                else:
                    chat_history.append((message, "🤖 Researching..."))
            except:
                chat_history.append((message, "🤖 Researching..."))
            
            yield chat_history
            time.sleep(1)
            
            # Generate actual response
            response = chatbot.generate_answer(message)
            
            # Update with final response
            try:
                if gradio_features['has_messages_type']:
                    chat_history[-1] = {"role": "assistant", "content": response}
                else:
                    chat_history[-1] = (message, response)
            except:
                chat_history[-1] = (message, response)
            
            yield chat_history
        
        def restart_chat():
            """Restart with initial greeting - universal compatibility"""
            try:
                if gradio_features['has_messages_type']:
                    return [{"role": "assistant", "content": "Hello! I'm your Ethiopian Coffee Disease Assistant. Ask me about coffee diseases, symptoms, treatments, or prevention methods! ☕"}]
                else:
                    return [("🤖", "Hello! I'm your Ethiopian Coffee Disease Assistant. Ask me about coffee diseases, symptoms, treatments, or prevention methods! ☕")]
            except:
                return [("🤖", "Hello! I'm your Ethiopian Coffee Disease Assistant. Ask me about coffee diseases, symptoms, treatments, or prevention methods! ☕")]
        
        def clear_chat():
            """Clear the chat history"""
            return []
        
        # Connect the interface
        msg.submit(
            respond,
            [msg, chatbot_interface],
            [chatbot_interface]
        )
        
        send_btn.click(
            respond,
            [msg, chatbot_interface],
            [chatbot_interface]
        ).then(lambda: "", outputs=[msg])
        
        # Clear and restart functions
        clear_btn.click(clear_chat, outputs=[chatbot_interface])
        restart_btn.click(restart_chat, outputs=[chatbot_interface])
        
        # Add initial greeting when the app starts
        demo.load(restart_chat, outputs=[chatbot_interface])
    
    return demo

if __name__ == "__main__":
    # Check if vector database exists
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"❌ Vector database directory '{PERSIST_DIRECTORY}' not found!")
        print("Please run 'create_vector_db.py' first to create the vector database.")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch options
    print("🚀 Starting Gradio interface...")
    print("📱 Access the chatbot at: http://localhost:7860")
    print("🌐 For network access, use your IP address instead of localhost")
    
    # Universal launch that works with all versions
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "show_error": True,
        "inbrowser": True
    }
    
    # Only add theme to launch if Blocks doesn't support it
    if not gradio_features['has_theme_in_blocks']:
        try:
            launch_kwargs["theme"] = gr.themes.Soft()
        except:
            pass
    
    demo.launch(**launch_kwargs)