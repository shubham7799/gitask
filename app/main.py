from fastapi import FastAPI
from app.api.routers.rag import router as rag_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GitAsk API")

app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)