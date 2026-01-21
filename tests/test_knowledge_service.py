import asyncio
import logging
from domain.entities import MCSDocument, MCSDocumentChunk, QueryResult
from infrastructure.interfaces import IKnowledgeBase
from infrastructure.adapters.llamaindex_adapter import LlamaIndexKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.knowledge_service import KnowledgeService

async def main():
    kb = LlamaIndexKnowledgeBase("./storage", False)
    kb_service = KnowledgeService(kb)
    
    query_result = await kb_service.retrieve(
        "Цалингүй чөлөө авах арга" 
    )
    for chunk in query_result.chunks:
        logger.info(f"Chunk ID: {chunk.chunk_id}\nContent: {chunk.content}\nScore: {chunk.metadata['score']}\n\n-----------\n")
        

if __name__ == "__main__":
    asyncio.run(main=main())