from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from domain.entities import MCSDocument, MCSDocumentChunk, QueryResult

class IKnowledgeBase(ABC):
    """
    Interface for a knowledge base that can store and retrieve documents and their chunks.
    """

    @abstractmethod
    async def insert(self, chunks: List[MCSDocumentChunk]) -> None:
        """
        Add MCS document chunks to the knowledge base.

        :param chunks: The list of document chunks to be inserted.
        """
        pass

    @abstractmethod
    async def query(self, query_text: str, top_k: int = 5) -> QueryResult:
        """
        Query the knowledge base for relevant document chunks.

        :param query_text: The text of the query.
        :param top_k: The number of the closest chunks in the coordinate space to return.
        :return: A QueryResult containing the relevant chunks and response.
        """
        pass

    @abstractmethod
    async def persist(self) -> None:
        """
        Persist the current state of the knowledge base to storage.
        """
        pass

    @abstractmethod
    async def load(self) -> None:
        """
        Load the knowledge base state from storage.
        """
        pass