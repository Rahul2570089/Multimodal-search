from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.products import router as products_router
from api.search import router as search_router
from utils.database import init_db
from utils.chroma_client import init_chroma


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    init_chroma()
    
    # Initialize RAG chain (this will load the embedding model)
    try:
        from services.rag_chain import get_rag_chain
        rag_chain = get_rag_chain()
        print("✅ RAG chain initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG chain: {e}")
    
    # Initialize visual search services
    try:
        from services.visual_search import get_visual_search_service
        from services.image_embeddings import get_image_embedding_service
        from services.image_processor import get_image_processor
        
        visual_search_service = get_visual_search_service()
        image_embedding_service = get_image_embedding_service()
        image_processor = get_image_processor()
        
        print("✅ Visual search services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize visual search services: {e}")
    
    # Initialize advanced multimodal fusion services
    try:
        from services.multimodal_fusion import get_multimodal_fusion_service
        from services.cross_modal_retrieval import get_cross_modal_retrieval_service
        from services.advanced_ranking import get_advanced_ranking_service
        from services.performance_optimizer import get_performance_optimizer
        from services.fusion_analytics import get_fusion_analytics_service
        
        fusion_service = get_multimodal_fusion_service()
        cross_modal_service = get_cross_modal_retrieval_service()
        ranking_service = get_advanced_ranking_service()
        performance_optimizer = get_performance_optimizer()
        analytics_service = get_fusion_analytics_service()
        
        print("✅ Advanced multimodal fusion services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize advanced fusion services: {e}")
    
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Multi-modal Product Search API",
    description="AI-powered e-commerce search with text and image capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(products_router, prefix="/api/products", tags=["products"])
app.include_router(search_router, prefix="/api/search", tags=["search"])


@app.get("/")
async def root():
    return {"message": "Multi-modal Product Search API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "multimodal-search-api"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
