import os
import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, AsyncGenerator
from app.core.rag_engine import process_repo_to_chroma
import uuid
from datetime import datetime, timedelta
import asyncio
import json
from app.utils import set_interval

router = APIRouter()


# In-memory storage for sessions (use Redis/DB in production)
sessions: Dict[str, dict] = {}

# Event queues for each session
event_queues: Dict[str, asyncio.Queue] = {}

class SetRepoRequest(BaseModel):
    github_url: HttpUrl
    session_id: Optional[str] = None


class SetRepoResponse(BaseModel):
    session_id: str
    status: str
    message: str
    indexed_at: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    timestamp: str


async def emit_event(session_id: str, event_data: dict):
    """Emit an event to the session's queue."""
    if session_id in event_queues:
        await event_queues[session_id].put(event_data)


def index_repository(session_id: str, github_url: str):
    """Background task to index repository."""
    try:
        sessions[session_id]["status"] = "indexing"
        
        # Emit indexing started event
        asyncio.run(emit_event(session_id, {
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Starting Repository Indexing...",
            "status": "info",
            "session_status": "indexing"
        }))
        
        # Create unique persist directory for this session
        persist_dir = f"chroma_store/{session_id}"
        
        # Process repository and create QA chain
        qa_chain = process_repo_to_chroma(str(github_url), persist_dir=persist_dir)
        
        sessions[session_id].update({
            "qa_chain": qa_chain,
            "status": "ready",
            "indexed_at": datetime.utcnow().isoformat(),
            "error": None
        })
        
        # Emit success event
        asyncio.run(emit_event(session_id, {
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Repository is ready for queries!",
            "status": "success",
            "session_status": "ready"
        }))
        
    except Exception as e:
        sessions[session_id].update({
            "status": "failed",
            "error": str(e)
        })

        print(f"Indexing failed for session {session_id}: {str(e)}")
        
        # Emit error event
        asyncio.run(emit_event(session_id, {
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Indexing failed: {str(e)}",
            "status": "error",
            "session_status": "failed"
        }))


@router.post("/set-repo")
async def set_repository(request: SetRepoRequest, background_tasks: BackgroundTasks):
    """
    Set and index a GitHub repository with real-time SSE progress updates.
    
    - If session_id is provided, it will reuse/update that session
    - If not provided, a new session will be created
    - Returns SSE stream with progress updates as they occur
    """
    
    # Generate or use existing session ID
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    
    # Initialize session
    sessions[session_id] = {
        "github_url": str(request.github_url),
        "status": "pending",
        "qa_chain": None,
        "created_at": datetime.utcnow().isoformat(),
        "indexed_at": None,
        "error": None
    }
    
    # Create event queue for this session
    event_queues[session_id] = asyncio.Queue()

    # Start background indexing
    asyncio.create_task(
        asyncio.to_thread(index_repository, session_id, request.github_url)
    )
    
    # Event generator that yields as events arrive
    async def event_generator() -> AsyncGenerator[str, None]:
        # First event: session created
        initial_event = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Session created, starting indexing...",
            "status": "info",
            "session_status": "pending"
        }
        yield f"data: {json.dumps(initial_event)}\n\n"
        
        # Stream events as they arrive
        queue = event_queues[session_id]
        
        while True:
            try:
                # Wait for next event (with timeout for heartbeat)
                event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                # Add session_id to event
                event_data["session_id"] = session_id
                
                # Send event
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Check if this is a terminal state
                if event_data.get("session_status") in ["ready", "failed"]:
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat if no events for 30 seconds
                if session_id in sessions:
                    heartbeat = {
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": f"Status: {sessions[session_id]['status']}",
                        "status": "heartbeat",
                        "session_status": sessions[session_id]["status"]
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
        
        # Cleanup
        if session_id in event_queues:
            del event_queues[session_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_repo(request: ChatRequest):
    """
    Chat with the indexed repository.
    
    Requires a valid session_id from /set-repo endpoint.
    The repository must be fully indexed (status: ready).
    """
    
    # Validate session exists
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please set a repository first using /set-repo"
        )
    
    session = sessions[request.session_id]
    
    # Check session status
    if session["status"] == "pending" or session["status"] == "indexing":
        raise HTTPException(
            status_code=425,
            detail=f"Repository is still being indexed. Current status: {session['status']}"
        )
    
    if session["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Repository indexing failed: {session.get('error', 'Unknown error')}"
        )
    
    if session["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session status: {session['status']}"
        )
    
    # Get QA chain
    qa_chain = session["qa_chain"]
    
    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail="QA chain not initialized"
        )
    
    try:
        # Invoke with session_id in config for conversational memory
        answer = qa_chain.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        return ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions.
    """
    
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "github_url": data["github_url"],
                "status": data["status"],
                "created_at": data["created_at"]
            }
            for sid, data in sessions.items()
        ]
    }


def cleanup_inactive_sessions():
    """
    Remove sessions inactive for more than 24 hours AND remove any directories
    in chroma_store/ that do not correspond to existing session IDs.
    Returns total count of deleted sessions + deleted orphan directories.
    """
    now = datetime.utcnow()
    deleted_count = 0
    sessions_to_delete = []

    # --- Remove inactive sessions ---
    for session_id, session_data in list(sessions.items()):
        created_at = datetime.fromisoformat(session_data["created_at"])
        
        if now - created_at > timedelta(hours=24):
            sessions_to_delete.append(session_id)

    for session_id in sessions_to_delete:
        # Remove from sessions dict
        if session_id in sessions:
            del sessions[session_id]

        # Remove ChromaDB directory
        persist_dir = f"chroma_store/{session_id}"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)

        deleted_count += 1

    # --- Remove orphan directories not present in sessions dict ---
    chroma_base = "chroma_store"
    orphan_deleted = 0

    if os.path.exists(chroma_base):
        for dirname in os.listdir(chroma_base):
            dirpath = os.path.join(chroma_base, dirname)

            # Only consider directories
            if not os.path.isdir(dirpath):
                continue

            # Directory name not in active sessions -> delete
            if dirname not in sessions:
                shutil.rmtree(dirpath, ignore_errors=True)
                orphan_deleted += 1

    return deleted_count + orphan_deleted

set_interval(cleanup_inactive_sessions, 100)