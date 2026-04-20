"""
Document processing and chunking service for RAG implementation
"""
import re
from typing import List, Dict, Any, Tuple
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing and chunking product documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}\:\/\@\#\$\%\&\*\+\=]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        return text.strip()
    
    def create_product_document(self, product_data: Dict[str, Any]) -> Document:
        """Create a document from product data"""
        try:
            # Combine product information into a structured text
            text_parts = []
            
            # Product name (most important)
            if product_data.get('name'):
                text_parts.append(f"Product: {product_data['name']}")
            
            # Description
            if product_data.get('description'):
                text_parts.append(f"Description: {self.clean_text(product_data['description'])}")
            
            # Category and brand
            if product_data.get('category'):
                text_parts.append(f"Category: {product_data['category']}")
            
            if product_data.get('brand'):
                text_parts.append(f"Brand: {product_data['brand']}")
            
            # Attributes
            attributes = []
            if product_data.get('color'):
                attributes.append(f"Color: {product_data['color']}")
            
            if product_data.get('size'):
                attributes.append(f"Size: {product_data['size']}")
            
            if product_data.get('material'):
                attributes.append(f"Material: {product_data['material']}")
            
            if attributes:
                text_parts.append(f"Attributes: {' | '.join(attributes)}")
            
            # Tags
            if product_data.get('tags'):
                text_parts.append(f"Tags: {', '.join(product_data['tags'])}")
            
            # Price and rating
            if product_data.get('price'):
                text_parts.append(f"Price: ${product_data['price']}")
            
            if product_data.get('rating'):
                text_parts.append(f"Rating: {product_data['rating']}/5")
            
            # Join all parts
            full_text = "\n".join(text_parts)
            
            # Create document with metadata
            metadata = {
                'product_id': product_data.get('id'),
                'name': product_data.get('name', ''),
                'category': product_data.get('category', ''),
                'brand': product_data.get('brand', ''),
                'price': product_data.get('price', 0),
                'rating': product_data.get('rating', 0),
                'in_stock': product_data.get('in_stock', 0)
            }
            
            return Document(page_content=full_text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error creating product document: {e}")
            return Document(page_content="", metadata={})
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks"""
        try:
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return [document]
    
    def process_product(self, product_data: Dict[str, Any]) -> List[Document]:
        """Process product and return chunks"""
        try:
            # Create document
            document = self.create_product_document(product_data)
            
            if not document.page_content:
                return []
            
            # Chunk document
            chunks = self.chunk_document(document)
            
            logger.info(f"Processed product {product_data.get('id')} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing product: {e}")
            return []
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        try:
            expanded_queries = [query]
            
            # Common e-commerce synonyms and expansions
            expansions = {
                'shoes': ['footwear', 'sneakers', 'boots', 'sandals'],
                'shirt': ['top', 'blouse', 'tee', 't-shirt'],
                'pants': ['trousers', 'jeans', 'slacks'],
                'dress': ['outfit', 'gown', 'attire'],
                'bag': ['purse', 'handbag', 'backpack'],
                'watch': ['timepiece', 'wristwatch'],
                'phone': ['smartphone', 'mobile', 'cellphone'],
                'laptop': ['notebook', 'computer'],
                'cheap': ['affordable', 'budget', 'inexpensive'],
                'expensive': ['premium', 'luxury', 'high-end'],
                'good': ['quality', 'excellent', 'great'],
                'bad': ['poor', 'low quality', 'terrible']
            }
            
            # Add expansions for words in query
            query_lower = query.lower()
            for word, synonyms in expansions.items():
                if word in query_lower:
                    for synonym in synonyms:
                        expanded_query = query_lower.replace(word, synonym)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
            
            return expanded_queries[:5]  # Limit to 5 queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def extract_search_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query"""
        try:
            intent = {
                'category': None,
                'brand': None,
                'color': None,
                'price_range': None,
                'attributes': [],
                'sentiment': 'neutral'
            }
            
            query_lower = query.lower()
            
            # Common categories
            categories = ['clothing', 'shoes', 'electronics', 'accessories', 'home', 'kitchen']
            for category in categories:
                if category in query_lower:
                    intent['category'] = category
            
            # Common brands (would be expanded with real brand data)
            brands = ['nike', 'adidas', 'apple', 'samsung', 'sony']
            for brand in brands:
                if brand in query_lower:
                    intent['brand'] = brand
            
            # Colors
            colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple']
            for color in colors:
                if color in query_lower:
                    intent['color'] = color
            
            # Price range
            if any(word in query_lower for word in ['cheap', 'affordable', 'budget']):
                intent['price_range'] = 'low'
            elif any(word in query_lower for word in ['expensive', 'premium', 'luxury']):
                intent['price_range'] = 'high'
            
            # Sentiment
            if any(word in query_lower for word in ['good', 'great', 'excellent', 'quality']):
                intent['sentiment'] = 'positive'
            elif any(word in query_lower for word in ['bad', 'poor', 'terrible']):
                intent['sentiment'] = 'negative'
            
            return intent
            
        except Exception as e:
            logger.error(f"Error extracting search intent: {e}")
            return {'category': None, 'brand': None, 'color': None, 'price_range': None, 'attributes': [], 'sentiment': 'neutral'}

# Global instance
document_processor = DocumentProcessor()

def get_document_processor() -> DocumentProcessor:
    """Get global document processor instance"""
    return document_processor
