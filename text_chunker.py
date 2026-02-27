"""Text chunking utilities for handling large documents."""
import tiktoken


def estimate_tokens(text, model="gpt-4"):
    """
    Estimate the number of tokens in text.
    
    Args:
        text: The text to estimate
        model: The model name (affects tokenizer)
        
    Returns:
        Estimated token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def chunk_text(text, max_tokens=6000, model="gpt-4", overlap=200):
    """
    Split text into chunks that fit within token limits.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (leaving room for prompts)
        model: The model name
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback: use character-based chunking
        encoding = None
    
    if encoding:
        # Token-based chunking
        tokens = encoding.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move forward, but overlap by 'overlap' tokens
            i += max_tokens - overlap
            
            if i >= len(tokens):
                break
    else:
        # Character-based chunking (fallback)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        overlap_chars = overlap * 4
        
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + max_chars]
            chunks.append(chunk)
            i += max_chars - overlap_chars
            
            if i >= len(text):
                break
    
    return chunks
