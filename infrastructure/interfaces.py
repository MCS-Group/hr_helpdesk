from typing import List

from abc import ABC, abstractmethod
from domain.entities import MCSDocumentChunk, QueryResult

class IKnowledgeService(ABC):
    """Interface for knowledge service operations."""

    @abstractmethod
    async def query(self, question: str, top_k: int = 2) -> QueryResult:
        """Handles a query to the knowledge base."""
        pass

    @abstractmethod
    async def insert(self, chunks: List[MCSDocumentChunk]) -> bool:
        """Inserts document chunks into the knowledge base."""
        pass

class IDocumentService(ABC):
    """Interface for document loading and ingestion services."""
    @abstractmethod
    async def ingest_documents(self, directory_path: str) -> List[MCSDocumentChunk]:
        """Ingest documents from the specified directory and return document chunks."""
        pass

class ILLMService(ABC):
    """Interface for LLM generation on grounds of prompt set and context."""
    @abstractmethod
    async def synthesize_response(self, context: List[MCSDocumentChunk]) -> str:
        """Generate a response from the LLM based on the given context."""
        pass

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