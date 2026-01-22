from infrastructure.interfaces import IDocumentService, IKnowledgeBase
from domain.entities import MCSDocumentChunk
from typing import List

import logging
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentService")

class DocumentService(IDocumentService):
    """Service layer for document loading and ingestion."""

    def __init__(self, allowed_extensions: List[str] = [".pdf"]):

        self._allowed_extensions = allowed_extensions

        self._file_extractor = {
            ".pdf": PDFReader(return_full_document=True)
        }
        self._encoding = "utf-8"
        self._parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=20
        )

    async def ingest_documents(self, directory_path: str) -> List[MCSDocumentChunk]:
        """Ingest documents from the specified directory and return document chunks."""

        reader = SimpleDirectoryReader(
            directory_path, 
            required_exts=self._allowed_extensions, 
            file_extractor=self._file_extractor, 
            encoding=self._encoding
        )

        documents = reader.load_data(show_progress=True)
        
        nodes = self._parser.get_nodes_from_documents(
            documents=documents, 
            show_progress=True
        )

        logger.info(f"Extracted {len(nodes)} nodes from documents. Type of a node: {type(nodes[0])}")

        chunks = []
        for i, node in enumerate(nodes):
            chunk = MCSDocumentChunk(
                chunk_id=f"{node.id_}_chunk_{i+1}",
                content=node.get_content(),
                metadata=node.metadata,
                embedding=None
            )
            chunks.append(chunk)
        
        return chunks