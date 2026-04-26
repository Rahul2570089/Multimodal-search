#!/usr/bin/env python3
"""
Generate image embeddings for existing products and populate visual search index
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
from services.visual_search import get_visual_search_service

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_product_image_embeddings():
    """Generate image embeddings for all products"""
    try:
        # Database setup
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/multimodal_search")
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        # Get visual search service
        visual_search_service = get_visual_search_service()
        
        # Get all products
        products = db.query(Product).all()
        logger.info(f"Found {len(products)} products to process for image embeddings")
        
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
                        'image_url': product.image_url
                    }
                    
                    # Add to visual search index
                    success = visual_search_service.add_product_image(product_data)
                    
                    if success:
                        processed += 1
                        logger.info(f"✅ Processed image embedding for product {product.id}: {product.name}")
                    else:
                        failed += 1
                        logger.error(f"❌ Failed to process image embedding for product {product.id}: {product.name}")
                
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error processing product {product.id}: {e}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.5)
        
        logger.info(f"\n📊 Image Embedding Generation Summary:")
        logger.info(f"✅ Successfully processed: {processed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📈 Success rate: {(processed/len(products)*100):.1f}%")
        
        db.close()
        return processed > 0
        
    except Exception as e:
        logger.error(f"Fatal error in image embedding generation: {e}")
        return False

def test_visual_search():
    """Test visual search with sample queries"""
    try:
        logger.info("🧪 Testing Visual Search...")
        
        visual_search_service = get_visual_search_service()
        
        # Test finding similar products
        test_product_ids = [1, 2, 3]  # Test with first few products
        
        for product_id in test_product_ids:
            logger.info(f"\n🔍 Finding similar products to product {product_id}")
            
            try:
                similar_products = visual_search_service.find_similar_products(product_id, k=5)
                
                if similar_products:
                    logger.info(f"   Found {len(similar_products)} similar products:")
                    for i, product in enumerate(similar_products[:3], 1):
                        logger.info(f"   {i}. {product.get('name', 'Unknown')} (similarity: {product.get('similarity_score', 0):.3f})")
                else:
                    logger.warning("   No similar products found")
                    
            except Exception as e:
                logger.error(f"   Error finding similar products: {e}")
        
        logger.info("\n✅ Visual search testing completed")
        
    except Exception as e:
        logger.error(f"Error testing visual search: {e}")

def create_sample_images():
    """Create sample image files for testing"""
    try:
        logger.info("🖼️ Creating sample images for testing...")
        
        # Create sample images directory
        sample_images_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_products', 'images')
        os.makedirs(sample_images_dir, exist_ok=True)
        
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Sample image configurations
        sample_configs = [
            {"name": "tshirt", "color": (255, 255, 255), "text": "T-Shirt"},
            {"name": "jeans", "color": (0, 0, 255), "text": "Jeans"},
            {"name": "shoes", "color": (0, 0, 0), "text": "Shoes"},
            {"name": "wallet", "color": (139, 69, 19), "text": "Wallet"},
            {"name": "headphones", "color": (128, 128, 128), "text": "Headphones"}
        ]
        
        for config in sample_configs:
            # Create a simple image
            img = Image.new('RGB', (224, 224), config["color"])
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                # Try to use a larger font
                draw.text((50, 100), config["text"], fill=(255, 255, 255))
            except:
                # Fallback to default font
                draw.text((50, 100), config["text"], fill=(255, 255, 255))
            
            # Save image
            image_path = os.path.join(sample_images_dir, f"{config['name']}.jpg")
            img.save(image_path, 'JPEG')
            logger.info(f"✅ Created sample image: {image_path}")
        
        logger.info("✅ Sample images created successfully")
        
    except Exception as e:
        logger.error(f"Error creating sample images: {e}")

def main():
    """Main function"""
    logger.info("🚀 Starting Image Embedding Generation...")
    
    start_time = time.time()
    
    # Create sample images for testing
    create_sample_images()
    
    # Generate embeddings
    success = generate_product_image_embeddings()
    
    if success:
        # Test visual search
        test_visual_search()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n🎉 Image embedding generation completed in {elapsed_time:.1f} seconds")
        
        logger.info("\n📝 Next steps:")
        logger.info("1. Test image search API: POST /api/search/image")
        logger.info("2. Try multimodal search: POST /api/search/multimodal")
        logger.info("3. Test with sample images from sample_products/images/")
        logger.info("4. Check visual search performance and accuracy")
    else:
        logger.error("❌ Image embedding generation failed!")
        logger.error("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
