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

def chat_function(message, history):
    """Function that handles the chat interaction"""
    if not chatbot.collection:
        return "❌ Error: Vector database not found. Please run 'create_vector_db.py' first to create the knowledge base."
    
    # Add typing animation effect
    for i in range(3):
        time.sleep(0.1)
        yield "🤖 Researching" + "." * (i + 1)
    
    # Generate the actual response
    response = chatbot.generate_answer(message)
    yield response

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #f5f1e6 0%, #e8dfca 100%);
        font-family: 'Arial', sans-serif;
    }
    .chatbot {
        background-color: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #8B4513 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .textbox textarea {
        border-radius: 10px !important;
        border: 2px solid #8B4513 !important;
        font-size: 14px;
    }
    .button-primary {
        background: #8B4513 !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold;
    }
    .button-primary:hover {
        background: #654321 !important;
    }
    .button-secondary {
        background: #A0522D !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
    }
    .examples {
        background: #f8f4e9 !important;
        border-radius: 10px !important;
        border: 1px solid #d4b896 !important;
    }
    """
    
    with gr.Blocks(
        title="☕ Ethiopian Coffee Disease Assistant",
        theme=gr.themes.Soft(),  # Removed invalid color parameters
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #8B4513, #A0522D); 
                       border-radius: 15px; color: white; margin-bottom: 20px;">
                <h1 style="margin: 0;">☕ Ethiopian Coffee Disease Research Assistant</h1>
                <h3 style="margin: 10px 0 0 0; font-weight: normal;">Your AI companion for Ethiopian coffee disease knowledge</h3>
            </div>
            """
        )
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    value=[[None, "Hello! I'm your Ethiopian Coffee Disease Assistant. Ask me about coffee diseases, symptoms, treatments, or prevention methods! ☕"]],
                    height=500,
                    show_copy_button=True,
                    placeholder="Chat with your coffee disease expert...",
                    elem_classes="chatbot"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Type your question",
                        placeholder="e.g., What is coffee leaf rust? How to prevent coffee berry disease?",
                        scale=4,
                        container=False,
                        lines=2
                    )
                    send_btn = gr.Button("Send 📤", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.ClearButton([msg, chatbot_interface], variant="secondary")
                    restart_btn = gr.Button("🔄 Restart Chat", variant="secondary")
            
            # Sidebar with information and examples
            with gr.Column(scale=1):
                with gr.Group():
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
                
                with gr.Group():
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
                        label="Click any question to try:",
                        elem_classes="examples"
                    )
                
                with gr.Group():
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
        
        # Event handlers
        def respond(message, chat_history):
            if not message.strip():
                return chat_history
            
            # Add user message
            chat_history.append([message, None])
            
            # Generate response with streaming effect
            response = ""
            for chunk in chat_function(message, chat_history):
                if "Researching" in chunk:
                    chat_history[-1][1] = chunk
                    yield chat_history
                else:
                    response = chunk
            
            # Final response
            chat_history[-1][1] = response
            yield chat_history
        
        def restart_chat():
            return [[None, "Hello! I'm your Ethiopian Coffee Disease Assistant. Ask me about coffee diseases, symptoms, treatments, or prevention methods! ☕"]]
        
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
        clear_btn.click(lambda: "", outputs=[msg])
        restart_btn.click(restart_chat, outputs=[chatbot_interface])
    
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
    
    demo.launch(
        server_name="0.0.0.0",  # Accessible from other devices
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public link
        show_error=True,
        inbrowser=True          # Open in browser automatically
    )