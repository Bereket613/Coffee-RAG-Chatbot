import os
from sentence_transformers import SentenceTransformer
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
CHUNK_FOLDER = "data/chunks"
COLLECTION_NAME = "coffee_disease"
PERSIST_DIRECTORY = "vector_db"

def create_vector_database():
    """Create and populate the vector database with persistent storage."""
    
    # --- Chroma Setup with Persistence ---
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"Creating new collection: {COLLECTION_NAME}")
    
    # Create new collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
    
    # --- Embedding Model ---
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")
    
    def get_chunks():
        """
        Read all chunk files and extract text chunks with their IDs.
        """
        texts = []
        ids = []
        
        if not os.path.exists(CHUNK_FOLDER):
            logger.error(f"Chunk folder does not exist: {CHUNK_FOLDER}")
            return ids, texts
        
        chunk_files = [f for f in os.listdir(CHUNK_FOLDER) if f.endswith("_chunks.txt")]
        
        if not chunk_files:
            logger.warning(f"No chunk files found in {CHUNK_FOLDER}")
            return ids, texts
        
        for file in chunk_files:
            path = os.path.join(CHUNK_FOLDER, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = content.split("\n\n==== CHUNK ====\n\n")
                    
                    for i, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if chunk:  # Only add non-empty chunks
                            texts.append(chunk)
                            ids.append(f"{file}_{i}")
                            
                logger.info(f"Processed {len(chunks)} chunks from {file}")
                
            except Exception as e:
                logger.error(f"Error reading file {file}: {str(e)}")
                continue
        
        logger.info(f"Total chunks loaded: {len(texts)}")
        return ids, texts
    
    try:
        # --- Get chunks and create embeddings ---
        ids, texts = get_chunks()
        
        if not texts:
            logger.error("No text chunks found to process. Exiting.")
            return False
        
        logger.info("Creating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # --- Store in ChromaDB ---
        logger.info("Storing embeddings in ChromaDB...")
        
        # Add in batches for better performance with large datasets
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_ids = ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            
            collection.add(
                documents=batch_texts,
                ids=batch_ids,
                embeddings=[emb.tolist() for emb in batch_embeddings]
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch_texts)} chunks")
        
        # Verify the collection was populated
        collection_count = collection.count()
        logger.info(f"✅ Vector DB created successfully with {collection_count} chunks.")
        
        # List all collections to verify
        collections = client.list_collections()
        logger.info(f"Available collections: {[col.name for col in collections]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Creating Vector Database...")
    success = create_vector_database()
    if success:
        print("✅ Vector database created successfully! You can now run the chatbot.")
        print(f"📁 Database saved in: {PERSIST_DIRECTORY}")
    else:
        print("❌ Failed to create vector database. Please check the logs.")