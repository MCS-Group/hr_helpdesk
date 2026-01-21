from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

primitive = Union[str, int, float, bool, None]

class MCSDocumentChunk(BaseModel):
    """
    Domain model for document chunks 
    A framework-agnostic representation of a document chunk.
    """
    chunk_id: str
    content: str
    metadata: Dict[str, primitive] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class MCSDocument(BaseModel):
    """
    Domain model for source documents
    A framework-agnostic representation of a document.
    """
    doc_id: str
    content: str
    metadata: Dict[str, primitive] = Field(default_factory=dict)
    source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class QueryResult(BaseModel):
    """
    Domain model for query results
    A framework-agnostic representation of a query result.
    """
    chunks: List[MCSDocumentChunk]
    response: str
    metadata: Dict[str, primitive] = Field(default_factory=dict)