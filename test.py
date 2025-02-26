from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print(GOOGLE_API_KEY, PINECONE_API_KEY)
