#!/usr/bin/env python3
"""
Generate embeddings for existing products and populate ChromaDB
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models.product import Product
from models.database import Base
from services.rag_chain import get_rag_chain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_product_embeddings():
    """Generate embeddings for all products and add to ChromaDB"""
    try:
        # Database setup
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/multimodal_search")
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Get RAG chain
        rag_chain = get_rag_chain()
        
        # Get all products
        products = db.query(Product).all()
        logger.info(f"Found {len(products)} products to process")
        
        if not products:
            logger.warning("No products found in database")
            return False
        
        # Process products in batches
        batch_size = 10
        processed = 0
        failed = 0
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: products {i+1}-{min(i+batch_size, len(products))}")
            
            for product in batch:
                try:
                    # Convert product to dict
                    product_data = {
                        'id': product.id,
                        'name': product.name,
                        'description': product.description,
                        'category': product.category,
                        'brand': product.brand,
                        'color': product.color,
                        'size': product.size,
                        'material': product.material,
                        'price': float(product.price),
                        'rating': float(product.rating),
                        'in_stock': product.in_stock,
                        'tags': product.tags or []
                    }
                    
                    # Add to RAG chain
                    success = rag_chain.add_product_documents(product_data)
                    
                    if success:
                        processed += 1
                        logger.info(f"✅ Processed product {product.id}: {product.name}")
                    else:
                        failed += 1
                        logger.error(f"❌ Failed to process product {product.id}: {product.name}")
                
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error processing product {product.id}: {e}")
            
            # Small delay to prevent overwhelming ChromaDB
            time.sleep(0.5)
        
        logger.info(f"\n📊 Embedding Generation Summary:")
        logger.info(f"✅ Successfully processed: {processed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📈 Success rate: {(processed/len(products)*100):.1f}%")
        
        db.close()
        return processed > 0
        
    except Exception as e:
        logger.error(f"Fatal error in embedding generation: {e}")
        return False

def test_embeddings():
    """Test the generated embeddings with sample searches"""
    try:
        logger.info("🧪 Testing generated embeddings...")
        
        rag_chain = get_rag_chain()
        
        # Test queries
        test_queries = [
            "comfortable t-shirt",
            "running shoes for exercise",
            "leather wallet",
            "wireless headphones with noise cancellation",
            "blue denim jeans"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing query: '{query}'")
            
            try:
                results = rag_chain.semantic_search(query, k=5)
                
                if results:
                    logger.info(f"   Found {len(results)} results:")
                    for i, doc in enumerate(results[:3]):
                        metadata = doc.metadata
                        score = metadata.get('search_score', 0)
                        name = metadata.get('name', 'Unknown')
                        logger.info(f"   {i+1}. {name} (score: {score:.3f})")
                else:
                    logger.warning("   No results found")
                    
            except Exception as e:
                logger.error(f"   Error testing query: {e}")
        
        logger.info("\n✅ Embedding testing completed")
        
    except Exception as e:
        logger.error(f"Error testing embeddings: {e}")

def main():
    """Main function"""
    logger.info("🚀 Starting product embedding generation...")
    
    start_time = time.time()
    
    # Generate embeddings
    success = generate_product_embeddings()
    
    if success:
        # Test embeddings
        test_embeddings()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n🎉 Embedding generation completed in {elapsed_time:.1f} seconds")
        
        logger.info("\n📝 Next steps:")
        logger.info("1. Test the semantic search API: POST /api/search/text with use_rag=true")
        logger.info("2. Try query understanding: POST /api/search/understand")
        logger.info("3. Check search analytics: GET /api/search/analytics")
    else:
        logger.error("❌ Embedding generation failed!")
        logger.error("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
