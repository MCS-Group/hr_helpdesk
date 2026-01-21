import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode
from infrastructure.interfaces import IKnowledgeBase
from domain.entities import MCSDocumentChunk, QueryResult, MCSDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLamaIndexKB")

class LlamaIndexKnowledgeBase(IKnowledgeBase):
    """
    LlamaIndex-based implementation of the IKnowledgeBase interface.
    """
    def __init__(
            self, 
            persist_dir: str = "./storage",
            generate_response: bool = False
        ):
        
        self._persist_dir = persist_dir
        self._index: Optional[VectorStoreIndex] = None
        self._generate_response = generate_response

        self.set_configuration()

        if os.path.exists(self._persist_dir):
            self._load()
            logger.info("Loaded existing knowledge base index from storage.")
        else:
            self._index = VectorStoreIndex([])
            logger.info("Initialized new knowledge base index.")


    async def insert(self, chunks: List[MCSDocumentChunk]) -> None:
        """Converts MCSDocumentChunks to LlamaIndex Documents and inserts them into the index."""
        nodes = [self._chunk_to_node(chunk) for chunk in chunks]

        if self._index is None:
            self._index = VectorStoreIndex(nodes)
        else:
            self._index.insert_nodes(nodes)

        logger.info(f"Inserted {len(chunks)} chunks into the knowledge base index.")

    async def query(self, query_text: str, top_k: int = 5) -> QueryResult:
        """Queries the knowledge base index and returns relevant document chunks."""
        if self._index is None:
            raise ValueError("Knowledge base index is not initialized.")
    
        if self._generate_response:
            logger.info("Generating response with relevant chunks.")
            query_engine = self._index.as_query_engine(
                similarity_top_k=top_k,
                response_mode=ResponseMode.DEFAULT
            )
        else:
            logger.info("Generating response is disabled; only retrieving relevant chunks.")
            query_engine = self._index.as_query_engine(
                similarity_top_k=top_k,
                response_mode=ResponseMode.NO_TEXT
            )

        response = query_engine.query(query_text)

        chunks = [
            self._node_to_chunk(node)
            for node in response.source_nodes
        ]

        query_result = QueryResult(
            chunks=chunks,
            response=str(response.response),
            metadata={"score": getattr(response, "score", None)}
        )

        logger.info(f"Queried knowledge base with text: '{query_text}'. Retrieved {len(chunks)} chunks.")

        return query_result

    async def load(self) -> None:
        """Loads the knowledge base state from storage."""
        if os.path.exists(self._persist_dir):
            self._load()
        else:
            logger.info("Persist directory does not exist. Cannot load knowledge base.")

    def _load(self) -> None:

        storage_context = StorageContext.from_defaults(persist_dir=self._persist_dir)
        self._index = load_index_from_storage(storage_context)
        logger.info("Knowledge base loaded from storage.")

    async def persist(self) -> None:
        """Persists the current state of the index to storage."""
        if self._index:
            self._index.storage_context.persist(persist_dir=self._persist_dir)
            logger.info("Knowledge base persisted to storage.")
    
    def set_configuration(self) -> None:
        """Sets up the LlamaIndex configuration with JinaEmbedding and OpenAI LLM."""

        load_dotenv()

        Settings.embed_model = JinaEmbedding(
            model="jina-embeddings-v3",
            api_key=os.getenv("JINAAI_API_KEY")
        )

        Settings.llm = OpenAI(
            model="chatgpt-4o-latest",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        logger.info("Knowledge base LLM and Embedding model configs set.")

    def _chunk_to_node(self, chunk: MCSDocumentChunk) -> TextNode:
        """Converts an MCSDocumentChunk (DTO) to a LlamaIndex TextNode (DAO)."""
        return TextNode(
            id=chunk.chunk_id,
            text=chunk.content,
            metadata=chunk.metadata,
            embedding=chunk.embedding
        )
    
    def _node_to_chunk(self, node: NodeWithScore) -> MCSDocumentChunk:
        """Converts a LlamaIndex TextNode (DAO) to an MCSDocumentChunk (DTO)."""
        return MCSDocumentChunk(
            chunk_id=node.id_,
            content=node.text,
            metadata={**node.metadata, "score": node.score},
            embedding=node.embedding
        )
    
    def count(self) -> int:
        """Returns the number of document chunks in the knowledge base."""
        if self._index is None:
            return 0
        return len(self._index.docstore.docs)