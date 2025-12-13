import os
from fastapi import FastAPI
from app.api.routers.rag import router as rag_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GitAsk API")

app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",os.getenv("FRONTEND_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)