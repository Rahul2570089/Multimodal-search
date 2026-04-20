"""
Text embedding service using Sentence-Transformers for semantic search
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TextEmbeddingService:
    """Service for generating and managing text embeddings"""
    
    def __init__(self):
        self.model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dimension
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return [0.0] * self.embedding_dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [[0.0] * self.embedding_dimension] * len(texts)
            
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    def embed_product(self, product_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for product by combining multiple fields"""
        try:
            # Combine relevant text fields
            text_parts = []
            
            if product_data.get('name'):
                text_parts.append(product_data['name'])
            
            if product_data.get('description'):
                text_parts.append(product_data['description'])
            
            if product_data.get('category'):
                text_parts.append(product_data['category'])
            
            if product_data.get('brand'):
                text_parts.append(product_data['brand'])
            
            if product_data.get('color'):
                text_parts.append(product_data['color'])
            
            if product_data.get('material'):
                text_parts.append(product_data['material'])
            
            if product_data.get('tags'):
                text_parts.extend(product_data['tags'])
            
            # Join with weights (name and description get higher weight)
            combined_text = " ".join(text_parts)
            
            if not combined_text.strip():
                return [0.0] * self.embedding_dimension
            
            return self.embed_text(combined_text)
            
        except Exception as e:
            logger.error(f"Error embedding product: {e}")
            return [0.0] * self.embedding_dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            if len(embedding1) != len(embedding2):
                return 0.0
            
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 10) -> List[tuple]:
        """Find most similar embeddings to query"""
        try:
            if not candidate_embeddings:
                return []
            
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate_emb)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar: {e}")
            return []

# Global instance
text_embedding_service = TextEmbeddingService()

def get_text_embedding_service() -> TextEmbeddingService:
    """Get the global text embedding service instance"""
    return text_embedding_service
