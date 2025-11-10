import os
import re
from pathlib import Path
import logging

# Configuration
INPUT_FOLDER = "data/text"
OUTPUT_FOLDER = "data/chunks"
CHUNK_SIZE = 700
CHUNK_SEPARATOR = "\n\n==== CHUNK ====\n\n"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks of approximately chunk_size characters, 
    respecting sentence boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size for each chunk
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Split into sentences while preserving punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed chunk size and we have content, save current chunk
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_text_files() -> None:
    """Process all text files in the input folder and create chunked versions."""
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    # Process each text file
    text_files = list(input_path.glob("*.txt"))
    
    if not text_files:
        logger.warning(f"No text files found in {input_path}")
        return
    
    for text_file in text_files:
        try:
            # Read file content
            with open(text_file, "r", encoding="utf-8") as file:
                text_content = file.read()
            
            # Skip empty files
            if not text_content.strip():
                logger.warning(f"Skipping empty file: {text_file.name}")
                continue
            
            # Chunk the text
            chunks = chunk_text(text_content)
            
            # Generate output filename
            output_filename = text_file.stem + "_chunks.txt"
            output_file_path = output_path / output_filename
            
            # Write chunks to output file
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(CHUNK_SEPARATOR.join(chunks))
            
            logger.info(f"✅ Successfully chunked: {text_file.name} → {output_filename} "
                       f"({len(chunks)} chunks created)")
            
        except UnicodeDecodeError:
            logger.error(f"Encoding error in file: {text_file.name}")
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {str(e)}")

if __name__ == "__main__":
    process_text_files()