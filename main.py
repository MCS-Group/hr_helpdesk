from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterator
import json
import os

from dotenv import load_dotenv
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.jinaai import JinaEmbedding

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity

from config import Config
from bot import GreetingBot, index

_ = load_dotenv()

INDEX_PATH = "./storage"

app = FastAPI()
bot = GreetingBot()

settings = BotFrameworkAdapterSettings(
    app_id=Config.APP_ID,
    app_password=Config.APP_PASSWORD,
    channel_auth_tenant=Config.TENANT_ID
)

adapter = BotFrameworkAdapter(settings)

@app.post("/")
async def messages(request: Request):
    body = await request.json()
    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    async def call_bot(context):
        await bot.on_turn(context)
    
    await adapter.process_activity(activity, auth_header, call_bot)
    return Response(status_code=200)

# Request/Response Models

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    teams_webhook: str = None  # Optional webhook URL for Teams

# Streaming RAG Function

async def query_rag_stream(index, query: str, top_k: int = 5) -> AsyncIterator[str]:
    """
    Generator function that yields tokens from the RAG query response.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        streaming=True,
    )

    streaming_response = query_engine.query(query)
    for token in streaming_response.response_gen:
        yield token

# FastAPI Endpoints
@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Endpoint that streams the RAG query response as Server-Sent Events (SSE).
    """
    try:

        async def event_generator():
            async for token in query_rag_stream(index, request.query, request.top_k):
                yield f"data: {json.dumps({"token": token})}\n\n"

            yield f"data: {json.dumps({"done": True})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query/stream-text")
async def stream_query_plain(request: QueryRequest):
    """
    Alternative streaming endpoint that streams plain text tokens.
    """
    try:
        return StreamingResponse(
            query_rag_stream(index, request.query, request.top_k),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)