#!/usr/bin/env python3
"""
Create sample product data for testing
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import create_engine
from models.product import Product
from models.database import Base
import random

def create_sample_products():
    """Create sample product data"""
    
    # Sample product data
    sample_products = [
        {
            "name": "Classic White T-Shirt",
            "description": "Comfortable 100% cotton t-shirt perfect for everyday wear. Features a classic fit with crew neck and short sleeves.",
            "price": 19.99,
            "category": "Clothing",
            "brand": "Basic Essentials",
            "color": "White",
            "size": "Medium",
            "material": "Cotton",
            "image_url": "https://example.com/images/white-tshirt.jpg",
            "thumbnail_url": "https://example.com/thumbnails/white-tshirt.jpg",
            "in_stock": 150,
            "rating": 4.2,
            "num_reviews": 156,
            "tags": ["casual", "everyday", "cotton", "classic"]
        },
        {
            "name": "Blue Denim Jeans",
            "description": "Classic fit denim jeans with modern styling. Features five-pocket design, zip fly, and button closure.",
            "price": 49.99,
            "category": "Clothing",
            "brand": "Denim Co",
            "color": "Blue",
            "size": "32W x 32L",
            "material": "Denim",
            "image_url": "https://example.com/images/blue-jeans.jpg",
            "thumbnail_url": "https://example.com/thumbnails/blue-jeans.jpg",
            "in_stock": 75,
            "rating": 4.5,
            "num_reviews": 289,
            "tags": ["casual", "denim", "classic", "durable"]
        },
        {
            "name": "Running Shoes",
            "description": "Lightweight running shoes with excellent cushioning and support. Features breathable mesh upper and durable rubber outsole.",
            "price": 89.99,
            "category": "Footwear",
            "brand": "SportTech",
            "color": "Black",
            "size": "10",
            "material": "Synthetic",
            "image_url": "https://example.com/images/running-shoes.jpg",
            "thumbnail_url": "https://example.com/thumbnails/running-shoes.jpg",
            "in_stock": 50,
            "rating": 4.7,
            "num_reviews": 412,
            "tags": ["athletic", "running", "comfortable", "lightweight"]
        },
        {
            "name": "Leather Wallet",
            "description": "Genuine leather bifold wallet with multiple card slots and cash compartment. Features RFID blocking technology.",
            "price": 39.99,
            "category": "Accessories",
            "brand": "Leather Goods",
            "color": "Brown",
            "size": "Standard",
            "material": "Leather",
            "image_url": "https://example.com/images/leather-wallet.jpg",
            "thumbnail_url": "https://example.com/thumbnails/leather-wallet.jpg",
            "in_stock": 60,
            "rating": 4.3,
            "num_reviews": 198,
            "tags": ["accessories", "leather", "rfid", "bifold"]
        },
        {
            "name": "Wireless Headphones",
            "description": "Premium Bluetooth headphones with active noise cancellation. Features 30-hour battery life and superior sound quality.",
            "price": 129.99,
            "category": "Electronics",
            "brand": "AudioTech",
            "color": "Black",
            "size": "One Size",
            "material": "Plastic/Metal",
            "image_url": "https://example.com/images/wireless-headphones.jpg",
            "thumbnail_url": "https://example.com/thumbnails/wireless-headphones.jpg",
            "in_stock": 35,
            "rating": 4.6,
            "num_reviews": 334,
            "tags": ["electronics", "bluetooth", "noise-cancelling", "wireless"]
        },
        {
            "name": "Summer Dress",
            "description": "Light and flowy summer dress perfect for warm weather. Features floral print and adjustable straps.",
            "price": 34.99,
            "category": "Clothing",
            "brand": "Summer Style",
            "color": "Floral",
            "size": "Medium",
            "material": "Polyester",
            "image_url": "https://example.com/images/summer-dress.jpg",
            "thumbnail_url": "https://example.com/thumbnails/summer-dress.jpg",
            "in_stock": 40,
            "rating": 4.1,
            "num_reviews": 87,
            "tags": ["summer", "dress", "floral", "casual"]
        },
        {
            "name": "Smart Watch",
            "description": "Feature-rich smartwatch with fitness tracking, heart rate monitor, and smartphone integration.",
            "price": 199.99,
            "category": "Electronics",
            "brand": "TechTime",
            "color": "Silver",
            "size": "42mm",
            "material": "Aluminum",
            "image_url": "https://example.com/images/smart-watch.jpg",
            "thumbnail_url": "https://example.com/thumbnails/smart-watch.jpg",
            "in_stock": 25,
            "rating": 4.4,
            "num_reviews": 267,
            "tags": ["electronics", "smartwatch", "fitness", "wearable"]
        },
        {
            "name": "Coffee Maker",
            "description": "Programmable coffee maker with thermal carafe and customizable brew strength. Makes up to 12 cups.",
            "price": 79.99,
            "category": "Home & Kitchen",
            "brand": "BrewMaster",
            "color": "Black",
            "size": "12 Cup",
            "material": "Stainless Steel",
            "image_url": "https://example.com/images/coffee-maker.jpg",
            "thumbnail_url": "https://example.com/thumbnails/coffee-maker.jpg",
            "in_stock": 30,
            "rating": 4.0,
            "num_reviews": 145,
            "tags": ["kitchen", "coffee", "programmable", "thermal"]
        }
    ]
    
    # More sample products for variety
    additional_products = []
    categories = ["Clothing", "Footwear", "Electronics", "Accessories", "Home & Kitchen"]
    brands = ["TechBrand", "StyleCo", "QualityGoods", "ModernLiving", "EverydayEssentials"]
    colors = ["Black", "White", "Red", "Blue", "Green", "Gray", "Brown", "Silver"]
    
    for i in range(20):  # Add 20 more random products
        additional_products.append({
            "name": f"Product {i+9}",
            "description": f"High-quality product {i+9} with excellent features and durability.",
            "price": round(random.uniform(15.99, 299.99), 2),
            "category": random.choice(categories),
            "brand": random.choice(brands),
            "color": random.choice(colors),
            "size": "Standard",
            "material": "Mixed Materials",
            "image_url": f"https://example.com/images/product-{i+9}.jpg",
            "thumbnail_url": f"https://example.com/thumbnails/product-{i+9}.jpg",
            "in_stock": random.randint(10, 100),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "num_reviews": random.randint(10, 500),
            "tags": ["quality", "popular", "bestseller", "recommended"]
        })
    
    all_products = sample_products + additional_products
    
    try:
        # Connect to database
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/multimodal_search")
        engine = create_engine(DATABASE_URL)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        # Insert products
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        try:
            # Clear existing products
            db.query(Product).delete()
            db.commit()
            
            # Add new products
            for product_data in all_products:
                product = Product(**product_data)
                db.add(product)
            
            db.commit()
            print(f"Successfully created {len(all_products)} sample products")
            
        except Exception as e:
            db.rollback()
            print(f"Error inserting products: {e}")
        finally:
            db.close()
            
    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    print("Creating sample product data...")
    create_sample_products()
    print("Sample data creation completed!")
