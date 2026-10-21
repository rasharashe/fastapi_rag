from openai import OpenAI
from google import genai
import os

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts (more efficient)."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [item.embedding for item in response.data]