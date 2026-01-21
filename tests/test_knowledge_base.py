import asyncio
import logging
from llama_index.core import SimpleDirectoryReader, Document
from domain.entities import MCSDocumentChunk
from infrastructure.adapters.llamaindex_adapter import LlamaIndexKnowledgeBase
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestKB")

async def main():
    # Load a document and chunk into MCS Documents Chunk
    
    # file_extractor = {
    #     ".pdf": PDFReader(return_full_document=True)
    # }
    # documents = SimpleDirectoryReader(
    #     input_files=[
    #         "docs\\HR-0506R_Чөлөө -240205.pdf"
    #     ], 
    #     file_extractor=file_extractor, 
    #     encoding="utf-8"
    # ).load_data()

    # parser = SentenceSplitter(
    #     chunk_size=800,
    #     chunk_overlap=60
    # )

    # nodes = parser.get_nodes_from_documents(documents)
    # logger.info(f"Extracted {len(nodes)} nodes from documents.")

    # logger.info(type(nodes[0]))

    kb = LlamaIndexKnowledgeBase(persist_dir="./storage")
    # chunks = []
    # for i, node in enumerate(nodes):
    #     chunk = MCSDocumentChunk(
    #         chunk_id=f"{node.id_}_chunk_{i}",
    #         content=node.text,
    #         metadata=node.metadata,
    #         embedding=None  # Embeddings will be handled by LlamaIndex internally
    #     )
    #     chunks.append(chunk)

    # await kb.insert(chunks)
    responses = await kb.query("Цалинтай чөлөөний тухай мэдээлэл", top_k=2)
    for response in responses.chunks:
        logger.info(f"Chunk ID: {response.chunk_id}")
        logger.info(f"Content: {response.content}")
        logger.info(f"Metadata: {response.metadata}")
        logger.info("-----")
    logger.info(f"Response: {responses.response}")


if __name__ == "__main__":
    # chunk1 = MCSDocumentChunk(
    #     chunk_id="chunk1",
    #     content="This is a sample chunk content.",
    #     metadata={"source": "sample_source", "score": 0.95, "good": True},
    #     embedding=[0.1, 0.2, 0.3]
    # )
    # print(chunk1)

    asyncio.run(main())