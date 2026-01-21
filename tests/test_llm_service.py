import logging
import asyncio

from services.llm_service import LLMService
from services.knowledge_service import KnowledgeService
from infrastructure.adapters.llamaindex_adapter import LlamaIndexKnowledgeBase
from domain.entities import MCSDocumentChunk
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLLMService")

async def main():
    kb = LlamaIndexKnowledgeBase("./storage", False)
    knowledge_service = KnowledgeService(kb)  # Replace None with an actual IKnowledgeBase implementation
    query = "Өвчтэй үед хэрхэн чөлөөгөө авах вэ?"
    context = await knowledge_service.query(query, top_k=3)
    logger.info("Retrieved Context Chunks:")
    for chunk in context.chunks:
        logger.info(f"Chunk ID: {chunk.chunk_id}\nContent Preview: {chunk.content[:100]}\nMetadata: {chunk.metadata}\n{'-'*40}\n\n")
    llm_service = LLMService("""
You are a helpful assistant that provides information about HR policies based on the provided context in Mongolian language.
Given the following context from HR documents:
{{context}}
Please provide a detailed answer to the question: "{{question}}"
If appropriate, you have to structure your output using bullet points or numbered lists for clarity.
Respond in Mongolian language.
""")
    response = await llm_service.synthesize_response(query, context.chunks)
    logger.info(f"LLM Response:\n{response}")

if __name__ == '__main__':
    asyncio.run(main())