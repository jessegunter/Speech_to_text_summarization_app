import openai
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load OpenAI API key securely
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set API key for OpenAI
client = openai.OpenAI()

# Load Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    """Uses Hugging Face to split long text into smaller, structured chunks before sending to GPT."""
    sentences = tokenizer.tokenize(text)
    return " ".join(sentences)  # Returns a properly formatted string

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
                 "Make the summary easy to read, structured with bullet points if needed, and avoid unnecessary details. "
                 "Ensure financial figures, percentages, and important stats are included."},
                {"role": "user", "content": f"Summarize this text by extracting key facts and numbers:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {str(e)}"

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