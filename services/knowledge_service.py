import logging
from typing import List

from infrastructure.interfaces import IKnowledgeBase, IKnowledgeService
from domain.entities import QueryResult, MCSDocumentChunk

from infrastructure.adapters.llamaindex_adapter import LlamaIndexKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KnowledgeService")

class KnowledgeService(IKnowledgeService):
    """Service layer for knowledge base operations."""

    def __init__(self, knowledge_base: IKnowledgeBase | None = None):
        
        if knowledge_base is None:
            knowledge_base = LlamaIndexKnowledgeBase()

        self._kb = knowledge_base

    async def query(self, question: str, top_k: int = 2) -> QueryResult:
        """Handles a query to the knowledge base."""
        
        query_result = await self._kb.query(
            question, 
            top_k=top_k
        )

        return query_result
    
    async def insert(self, chunks: List[MCSDocumentChunk]) -> bool:
        """Inserts document chunks into the knowledge base and persists the state."""
        try:
            await self._kb.insert(chunks)
            await self._kb.persist()

            return True
        
        except Exception as e:

            logger.setLevel(logging.ERROR)
            logger.error(f"Error inserting chunks: {e}")
            
            return False