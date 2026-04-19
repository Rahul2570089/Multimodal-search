#!/usr/bin/env python3
"""
Startup script to verify the setup and run basic tests
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2-binary',
        'chromadb', 'langchain', 'sentence-transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        'backend/api', 'backend/models', 'backend/services', 'backend/utils',
        'scripts', 'data', 'sample_products', 'docs'
    ]
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✅ Project structure is correct")
    return True

def test_api_imports():
    """Test if API modules can be imported"""
    try:
        from api.products import router as products_router
        from api.search import router as search_router
        from models.product import Product
        from utils.database import init_db
        
        print("✅ API modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all checks"""
    print("🚀 Starting Multi-modal Search Setup Verification\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("API Imports", test_api_imports)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Your Weekend 1 setup is complete.")
        print("\n📝 Next Steps:")
        print("1. Start Docker Desktop")
        print("2. Run: docker-compose up -d")
        print("3. Run: python scripts/create_sample_data.py")
        print("4. Run: python scripts/setup_chroma.py")
        print("5. Visit: http://localhost:8000/docs")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")

if __name__ == "__main__":
    main()
