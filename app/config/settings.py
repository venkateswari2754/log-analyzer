import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    VECTOR_CACHE_DIR = "log_cache"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment.")

settings = Settings()
