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
