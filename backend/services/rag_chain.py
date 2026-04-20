"""
RAG (Retrieval-Augmented Generation) chain for semantic search
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from .text_embeddings import get_text_embedding_service
from .document_processor import get_document_processor
from utils import get_chroma_client

logger = logging.getLogger(__name__)

class RAGChain:
    """RAG chain for enhanced product search"""
    
    def __init__(self):
        self.embedding_service = get_text_embedding_service()
        self.document_processor = get_document_processor()
        self.chroma_client = get_chroma_client()
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "products")
        self.vector_store = None
        self.retrieval_chain = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store"""
        try:
            if self.chroma_client is None:
                logger.error("ChromaDB client not initialized")
                return
            
            # Create HuggingFace embeddings (compatible with Chroma)
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_service.model_name,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize vector store
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
            
            logger.info("RAG vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None
    
    def add_product_documents(self, product_data: Dict[str, Any]) -> bool:
        """Add product documents to vector store"""
        try:
            if self.vector_store is None:
                logger.error("Vector store not initialized")
                return False
            
            # Process product into chunks
            chunks = self.document_processor.process_product(product_data)
            
            if not chunks:
                logger.warning(f"No chunks generated for product {product_data.get('id')}")
                return False
            
            # Add to vector store
            product_id = str(product_data.get('id'))
            ids = [f"{product_id}_chunk_{i}" for i in range(len(chunks))]
            
            self.vector_store.add_documents(chunks, ids=ids)
            
            logger.info(f"Added {len(chunks)} chunks for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product documents: {e}")
            return False
    
    def semantic_search(self, query: str, k: int = 10, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform semantic search using vector similarity"""
        try:
            if self.vector_store is None:
                logger.error("Vector store not initialized")
                return []
            
            # Expand query
            expanded_queries = self.document_processor.expand_query(query)
            
            # Search with all query variations
            all_results = []
            for expanded_query in expanded_queries:
                if filters:
                    results = self.vector_store.similarity_search(
                        expanded_query, k=k, filter=filters
                    )
                else:
                    results = self.vector_store.similarity_search(expanded_query, k=k)
                
                all_results.extend(results)
            
            # Remove duplicates and re-rank
            unique_results = self._deduplicate_and_rerank(all_results, query)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _deduplicate_and_rerank(self, documents: List[Document], 
                               query: str) -> List[Document]:
        """Remove duplicates and re-rank results"""
        try:
            # Remove duplicates by product_id
            seen_products = set()
            unique_docs = []
            
            for doc in documents:
                product_id = doc.metadata.get('product_id')
                if product_id and product_id not in seen_products:
                    seen_products.add(product_id)
                    unique_docs.append(doc)
            
            # Re-rank based on multiple factors
            query_embedding = self.embedding_service.embed_text(query)
            
            for doc in unique_docs:
                # Compute semantic similarity
                content_embedding = self.embedding_service.embed_text(doc.page_content)
                semantic_score = self.embedding_service.compute_similarity(
                    query_embedding, content_embedding
                )
                
                # Add scoring factors
                rating = doc.metadata.get('rating', 0)
                in_stock = doc.metadata.get('in_stock', 0)
                price = doc.metadata.get('price', 0)
                
                # Combined score (semantic + rating + stock)
                combined_score = (
                    semantic_score * 0.6 +  # 60% semantic similarity
                    (rating / 5) * 0.3 +    # 30% rating
                    (min(in_stock, 10) / 10) * 0.1  # 10% stock availability
                )
                
                doc.metadata['search_score'] = combined_score
                doc.metadata['semantic_score'] = semantic_score
            
            # Sort by combined score
            unique_docs.sort(key=lambda x: x.metadata.get('search_score', 0), reverse=True)
            
            return unique_docs
            
        except Exception as e:
            logger.error(f"Error in deduplication and reranking: {e}")
            return documents
    
    def hybrid_search(self, query: str, k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Hybrid search combining semantic and keyword search"""
        try:
            # Semantic search results
            semantic_results = self.semantic_search(query, k=k*2, filters=filters)
            
            # Keyword search (using basic text matching)
            keyword_results = self._keyword_search(query, k=k*2, filters=filters)
            
            # Combine and re-rank
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, query
            )
            
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.semantic_search(query, k=k, filters=filters)
    
    def _keyword_search(self, query: str, k: int = 10,
                       filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Basic keyword search as fallback"""
        try:
            if self.vector_store is None:
                return []
            
            # Use Chroma's built-in search (which includes some keyword matching)
            results = self.vector_store.similarity_search(query, k=k, filter=filters)
            
            # Lower scores for pure keyword matches
            for doc in results:
                doc.metadata['search_score'] = doc.metadata.get('search_score', 0) * 0.5
                doc.metadata['search_type'] = 'keyword'
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[Document],
                               keyword_results: List[Document],
                               query: str) -> List[Document]:
        """Combine semantic and keyword search results"""
        try:
            # Mark result types
            for doc in semantic_results:
                doc.metadata['search_type'] = 'semantic'
            
            for doc in keyword_results:
                doc.metadata['search_type'] = 'keyword'
            
            # Combine results
            all_results = semantic_results + keyword_results
            
            # Deduplicate and re-rank
            combined_results = self._deduplicate_and_rerank(all_results, query)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return semantic_results
    
    def get_search_explanation(self, query: str, results: List[Document]) -> Dict[str, Any]:
        """Generate explanation for search results"""
        try:
            intent = self.document_processor.extract_search_intent(query)
            
            explanation = {
                'query': query,
                'intent': intent,
                'total_results': len(results),
                'result_types': {},
                'avg_score': 0,
                'top_categories': [],
                'price_range': {'min': float('inf'), 'max': 0}
            }
            
            if not results:
                return explanation
            
            # Analyze results
            scores = []
            categories = {}
            
            for doc in results:
                # Score analysis
                score = doc.metadata.get('search_score', 0)
                scores.append(score)
                
                # Category analysis
                category = doc.metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Price range
                price = doc.metadata.get('price', 0)
                explanation['price_range']['min'] = min(explanation['price_range']['min'], price)
                explanation['price_range']['max'] = max(explanation['price_range']['max'], price)
                
                # Search types
                search_type = doc.metadata.get('search_type', 'unknown')
                explanation['result_types'][search_type] = explanation['result_types'].get(search_type, 0) + 1
            
            # Calculate averages and top categories
            explanation['avg_score'] = sum(scores) / len(scores) if scores else 0
            explanation['top_categories'] = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating search explanation: {e}")
            return {'query': query, 'error': str(e)}

# Global instance
rag_chain = RAGChain()

def get_rag_chain() -> RAGChain:
    """Get global RAG chain instance"""
    return rag_chain
