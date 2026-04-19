#!/usr/bin/env python3
"""
Setup script for ChromaDB and initial data
"""
import chromadb
from chromadb.config import Settings
import time
import os

def setup_chroma():
    """Setup ChromaDB collection"""
    try:
        # Connect to ChromaDB
        client = chromadb.HttpClient(
            host='localhost',
            port=8001
        )
        
        print("Connected to ChromaDB")
        
        # Delete existing collection if it exists
        try:
            client.delete_collection("products")
            print("Deleted existing products collection")
        except:
            print("No existing collection to delete")
        
        # Create new collection
        collection = client.create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("Created new products collection")
        
        # Add some sample embeddings (placeholder for now)
        sample_embeddings = [
            [0.1] * 384,  # Placeholder embedding
            [0.2] * 384,
            [0.3] * 384
        ]
        
        sample_documents = [
            "Classic White T-Shirt - Comfortable cotton t-shirt",
            "Blue Denim Jeans - Classic fit denim jeans",
            "Running Shoes - Lightweight running shoes"
        ]
        
        sample_ids = ["prod_1", "prod_2", "prod_3"]
        
        collection.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            ids=sample_ids
        )
        
        print("Added sample embeddings to collection")
        print(f"Collection count: {collection.count()}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        return False

if __name__ == "__main__":
    print("Setting up ChromaDB...")
    if setup_chroma():
        print("ChromaDB setup completed successfully!")
    else:
        print("ChromaDB setup failed!")
