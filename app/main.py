from fastapi import FastAPI
from app.api.routers.rag import router as rag_router

app = FastAPI(title="GitAsk API")

app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])