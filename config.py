import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, runs locally
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "schema_embeddings"