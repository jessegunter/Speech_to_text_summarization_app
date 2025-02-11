import openai
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load OpenAI API key securely
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
client = openai.OpenAI()

# Load Hugging Face tokenizer for chunking
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_text(text, max_tokens=4000):
    """Splits text into smaller chunks within OpenAI’s token limit."""
    tokens = tokenizer.encode(text, add_special_tokens=False)  # Tokenize text
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]  # Take up to max_tokens
        chunks.append(tokenizer.decode(chunk))  # Convert back to readable text

    return chunks  # Returns a list of chunked text

def summarize_text(text):
    """
    Uses GPT-3.5 to generate a summary that highlights key points, figures, and relevant data.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                 "You are an AI that summarizes text efficiently, focusing on key points, numerical data, and critical takeaways. "
                 "Make the summary easy to read, structured with bullet points if needed, and avoid unnecessary details."},
                {"role": "user", "content": f"Summarize this text:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {str(e)}"

def summarize_large_text(text):
    """Summarizes large text by processing it in smaller chunks."""
    chunks = chunk_text(text, max_tokens=4000)  # Split text into chunks
    summaries = [summarize_text(chunk) for chunk in chunks]  # Summarize each chunk
    return summarize_text(" ".join(summaries))  # Final summary

def translate_text(text, target_language):
    """
    Uses GPT-3.5 to translate text into French, Spanish, or Japanese.
    """
    if target_language.lower() not in ["french", "spanish", "japanese"]:
        return "Error: Unsupported language. Choose French, Spanish, or Japanese."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a professional translator who translates text into {target_language}."},
                {"role": "user", "content": f"Translate this text into {target_language}: {text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"

def translate_large_text(text, target_language):
    """Translates large text by processing it in smaller chunks."""
    chunks = chunk_text(text, max_tokens=4000)
    translated_chunks = [translate_text(chunk, target_language) for chunk in chunks]
    return " ".join(translated_chunks)