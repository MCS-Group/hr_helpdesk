import asyncio
import logging

from services.document_service import DocumentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestDocumentService")

async def main():
    document_service = DocumentService([".pdf"])
    logger.info("Starting document ingestion from 'test_docs/' directory...")
    chunks = await document_service.ingest_documents("test_docs/")

    for chunk in chunks:
        logger.info(f"Chunk ID: {chunk.chunk_id}\nContent Preview: {chunk.content[:100]}\nMetadata: {chunk.metadata}\n{'-'*40}")

if __name__ == "__main__":
    asyncio.run(main())