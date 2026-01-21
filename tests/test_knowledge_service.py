import asyncio
import logging
from domain.entities import MCSDocument, MCSDocumentChunk, QueryResult
from infrastructure.interfaces import IKnowledgeBase
from infrastructure.adapters.llamaindex_adapter import LlamaIndexKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from application.services import KnowledgeService

async def main():
    kb = LlamaIndexKnowledgeBase("./storage", False)
    kb_service = KnowledgeService(kb)
    
    chunks = await kb_service.query(
        "Өвчтэй үед чөлөөг авах арга зам" 
    )
    for chunk in chunks:
        logger.info(f"Chunk ID: {chunk.chunk_id}\nContent: {chunk.content}\nScore: {chunk.metadata["score"]}\n\n-----------\n")
        

if __name__ == "__main__":
    asyncio.run(main=main())