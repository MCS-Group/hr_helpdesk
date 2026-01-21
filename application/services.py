from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from infrastructure.interfaces import IKnowledgeBase
from domain.entities import MCSDocument, MCSDocumentChunk, QueryResult

class KnowledgeService:
    """Service layer for knowledge base operations."""

    def __init__(self, knowledge_base: IKnowledgeBase):
        self._kb = knowledge_base

    async def ask(self, question: str, top_k: int = 2) -> QueryResult:
        """Handles a query to the knowledge base."""
        query_result = await self._kb.query(question, top_k=top_k)
        return query_result.response
    
    async def query(self, question: str, top_k: int = 2) -> List[MCSDocumentChunk]:
        """Handles a query to the knowledge base and returns relevant document chunks."""
        query_result = await self._kb.query(question, top_k=top_k)
        return query_result.chunks
    
