from typing import List
from abc import ABC, abstractmethod

from infrastructure.interfaces import IKnowledgeBase, IKnowledgeService
from domain.entities import QueryResult

class KnowledgeService(IKnowledgeService):
    """Service layer for knowledge base operations."""

    def __init__(self, knowledge_base: IKnowledgeBase):
        self._kb = knowledge_base

    async def retrieve(self, question: str, top_k: int = 2) -> QueryResult:
        """Handles a query to the knowledge base."""
        query_result = await self._kb.query(question, top_k=top_k)
        return query_result
    
