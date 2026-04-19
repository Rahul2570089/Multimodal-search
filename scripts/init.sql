-- Initialize database schema for multi-modal search

-- Create extensions for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(100) NOT NULL,
    brand VARCHAR(100),
    color VARCHAR(50),
    size VARCHAR(50),
    material VARCHAR(100),
    image_url VARCHAR(500),
    thumbnail_url VARCHAR(500),
    in_stock INTEGER DEFAULT 0,
    rating DECIMAL(3, 2) DEFAULT 0.00,
    num_reviews INTEGER DEFAULT 0,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Search logs for analytics
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_type VARCHAR(20) NOT NULL, -- 'text', 'image', 'multimodal'
    results_count INTEGER,
    click_through_rate DECIMAL(5, 4),
    response_time_ms INTEGER,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Product embeddings (for backup/analysis)
CREATE TABLE IF NOT EXISTS product_embeddings (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    embedding_type VARCHAR(20) NOT NULL, -- 'text', 'image', 'multimodal'
    embedding vector(384), -- Adjust size based on model
    model_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, embedding_type)
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating);
CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at);
CREATE INDEX IF NOT EXISTS idx_search_logs_query_type ON search_logs(query_type);
CREATE INDEX IF NOT EXISTS idx_search_logs_created_at ON search_logs(created_at);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample categories
INSERT INTO products (name, description, price, category, brand, color, in_stock, rating, num_reviews) VALUES
('Classic White T-Shirt', 'Comfortable cotton t-shirt perfect for everyday wear', 19.99, 'Clothing', 'Basic Brand', 'White', 100, 4.2, 156),
('Blue Denim Jeans', 'Classic fit denim jeans with modern styling', 49.99, 'Clothing', 'Denim Co', 'Blue', 75, 4.5, 289),
('Running Shoes', 'Lightweight running shoes with excellent cushioning', 89.99, 'Footwear', 'SportTech', 'Black', 50, 4.7, 412),
('Leather Wallet', 'Genuine leather bifold wallet with multiple card slots', 39.99, 'Accessories', 'Leather Goods', 'Brown', 60, 4.3, 198),
('Wireless Headphones', 'Bluetooth headphones with noise cancellation', 129.99, 'Electronics', 'AudioTech', 'Black', 35, 4.6, 334)
ON CONFLICT DO NOTHING;
