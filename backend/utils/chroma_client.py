import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8001")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "products")

# Global ChromaDB client and collection
chroma_client = None
product_collection = None

def init_chroma():
    """Initialize ChromaDB client and collection"""
    global chroma_client, product_collection
    
    try:
        # Create ChromaDB client
        chroma_client = chromadb.HttpClient(
            host=CHROMA_URL.split('//')[1].split(':')[0],
            port=int(CHROMA_URL.split(':')[-1])
        )
        
        # Get or create collection
        try:
            product_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"Connected to existing ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        except:
            product_collection = chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new ChromaDB collection: {CHROMA_COLLECTION_NAME}")
            
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        chroma_client = None
        product_collection = None

def get_chroma_client():
    """Get ChromaDB client"""
    return chroma_client

def get_product_collection():
    """Get product collection"""
    return product_collection
