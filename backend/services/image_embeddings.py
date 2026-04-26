"""
Image embedding service using OpenCLIP for visual search
"""
import os
import io
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
import open_clip
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ImageEmbeddingService:
    """Service for generating and managing image embeddings"""
    
    def __init__(self):
        self.model_name = os.getenv("OPENCLIP_MODEL", "ViT-B-32-quickgelu")
        self.pretrained = os.getenv("OPENCLIP_PRETRAINED", "laion400m_e32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.embedding_dimension = 512  # Default for ViT-B-32
        self._load_model()
    
    def _load_model(self):
        """Load OpenCLIP model"""
        try:
            logger.info(f"Loading OpenCLIP model: {self.model_name}")
            
            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dimension = self.model.visual.output_dim
            
            logger.info(f"OpenCLIP model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model {self.model_name}: {e}")
            raise
    
    def preprocess_image(self, image: Union[Image.Image, bytes, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing
            processed = self.preprocess(image)
            
            # Add batch dimension
            return processed.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def embed_image(self, image: Union[Image.Image, bytes, np.ndarray]) -> List[float]:
        """Generate embedding for a single image"""
        try:
            with torch.no_grad():
                # Preprocess image
                processed_image = self.preprocess_image(image)
                processed_image = processed_image.to(self.device)
                
                # Generate embedding
                image_features = self.model.encode_image(processed_image)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to list
                embedding = image_features.cpu().numpy().flatten().tolist()
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            return [0.0] * self.embedding_dimension
    
    def embed_images(self, images: List[Union[Image.Image, bytes, np.ndarray]]) -> List[List[float]]:
        """Generate embeddings for multiple images"""
        try:
            embeddings = []
            
            for image in images:
                embedding = self.embed_image(image)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding images: {e}")
            return [[0.0] * self.embedding_dimension] * len(images)
    
    def embed_product_image(self, product_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for product image"""
        try:
            image_url = product_data.get('image_url')
            if not image_url:
                logger.warning(f"No image URL for product {product_data.get('id')}")
                return [0.0] * self.embedding_dimension
            
            # For now, create a placeholder embedding
            # In production, you would download and process the actual image
            embedding = self._create_placeholder_embedding(product_data)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding product image: {e}")
            return [0.0] * self.embedding_dimension
    
    def _create_placeholder_embedding(self, product_data: Dict[str, Any]) -> List[float]:
        """Create a placeholder embedding based on product attributes"""
        try:
            # Create a deterministic embedding based on product attributes
            attributes = [
                product_data.get('name', ''),
                product_data.get('category', ''),
                product_data.get('brand', ''),
                product_data.get('color', ''),
                product_data.get('material', '')
            ]
            
            # Convert to string and hash
            attr_string = '|'.join(attributes)
            hash_value = hash(attr_string) % (10**6)
            
            # Create embedding from hash
            embedding = np.zeros(self.embedding_dimension)
            
            # Distribute hash value across embedding dimensions
            for i in range(min(len(str(hash_value)), self.embedding_dimension)):
                embedding[i] = (hash_value % (i + 1)) / (i + 1)
            
            # Add some randomness for uniqueness
            np.random.seed(hash_value)
            embedding += np.random.normal(0, 0.1, self.embedding_dimension)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error creating placeholder embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two image embeddings"""
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
            logger.error(f"Error computing image similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 10) -> List[tuple]:
        """Find most similar images to query"""
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
            logger.error(f"Error finding most similar images: {e}")
            return []
    
    def extract_image_features(self, image: Union[Image.Image, bytes, np.ndarray]) -> Dict[str, Any]:
        """Extract additional features from image"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Basic image features
            features = {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Color analysis
            if image.mode == 'RGB':
                # Convert to numpy for analysis
                img_array = np.array(image)
                
                # Dominant colors (simplified)
                pixels = img_array.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)
                
                features.update({
                    'unique_colors': len(unique_colors),
                    'dominant_color': unique_colors[0].tolist() if len(unique_colors) > 0 else [0, 0, 0],
                    'brightness': float(np.mean(pixels) / 255.0)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def validate_image(self, image: Union[Image.Image, bytes, np.ndarray]) -> Dict[str, Any]:
        """Validate image and return validation results"""
        try:
            validation_result = {
                'is_valid': False,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }
            
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                validation_result['errors'].append(f"Unsupported image type: {type(image)}")
                return validation_result
            
            # Check image size
            max_size = 10 * 1024 * 1024  # 10MB
            if hasattr(image, 'size') and image.size > max_size:
                validation_result['warnings'].append("Image is large, consider compression")
            
            # Check dimensions
            if image.width < 32 or image.height < 32:
                validation_result['errors'].append("Image too small (minimum 32x32)")
            elif image.width > 4096 or image.height > 4096:
                validation_result['warnings'].append("Image very large, may affect performance")
            
            # Check aspect ratio
            aspect_ratio = image.width / image.height
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                validation_result['warnings'].append("Extreme aspect ratio may affect search quality")
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'WEBP']:
                validation_result['suggestions'].append("Consider using JPEG or PNG for better compatibility")
            
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'suggestions': []}

# Global instance
image_embedding_service = ImageEmbeddingService()

def get_image_embedding_service() -> ImageEmbeddingService:
    """Get global image embedding service instance"""
    return image_embedding_service
