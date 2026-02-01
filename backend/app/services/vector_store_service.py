"""Vector store service for Chroma operations."""

import logging
from typing import ClassVar, Optional

from fastapi import Depends
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.core.exceptions import CollectionNotFoundError
from app.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector search."""

    content: str = Field(..., description="The chunk text content")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    chunk_index: int = Field(..., ge=0, description="Position in chunk sequence")
    relevance_score: float = Field(..., ge=0, le=1, description="Similarity score")
    source: str = Field(..., description="Original filename")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "This is the chunk text...",
                "page_number": 3,
                "chunk_index": 7,
                "relevance_score": 0.89,
                "source": "report.pdf",
            }
        }
    )


class VectorStoreService:
    """
    Manages Chroma vector store operations.

    Each session gets its own collection.
    Collections are ephemeral (in-memory for v1, configurable persist dir).

    This is a singleton to maintain collection references across requests.
    """

    _instance: ClassVar["VectorStoreService | None"] = None
    _embedding_service: EmbeddingService
    _persist_directory: Optional[str]
    _collections: dict[str, Chroma]

    def __new__(cls, *args, **kwargs) -> "VectorStoreService":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Creating singleton VectorStoreService instance")
        return cls._instance

    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize vector store service.

        Args:
            embedding_service: For generating embeddings
            persist_directory: Optional dir for Chroma persistence.
                             If None, uses in-memory storage.
        """
        # Only initialize once (singleton)
        if not hasattr(self, '_collections'):
            logger.info("Initializing VectorStoreService")
            self._embedding_service = embedding_service
            self._persist_directory = persist_directory
            self._collections: dict[str, Chroma] = {}  # Track active collections
            logger.info(f"VectorStoreService initialized with persist_dir: {persist_directory}")

    def create_collection(
        self, collection_name: str, documents: list[Document]
    ) -> int:
        """
        Create a new Chroma collection and add documents.

        Args:
            collection_name: Unique name for collection (from session)
            documents: Langchain Documents with metadata

        Returns:
            Number of documents added

        Raises:
            ValueError: If collection already exists
        """
        logger.info(f"Creating ChromaDB collection '{collection_name}' with {len(documents)} documents")

        if collection_name in self._collections:
            logger.error(f"❌ Collection '{collection_name}' already exists")
            raise ValueError(f"Collection '{collection_name}' already exists")

        # Create Chroma collection from documents
        logger.debug(f"Embedding {len(documents)} documents...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self._embedding_service.embeddings,
            collection_name=collection_name,
            persist_directory=self._persist_directory,
        )

        # Store reference to collection
        self._collections[collection_name] = vectorstore
        logger.info(
            f"✓ Created collection '{collection_name}' with {len(documents)} documents. "
            f"Total collections: {len(self._collections)}"
        )

        return len(documents)

    def get_collection(self, collection_name: str) -> Chroma:
        """
        Get existing Chroma collection.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            Chroma vectorstore instance

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        if collection_name not in self._collections:
            available = list(self._collections.keys())
            logger.error(
                f"❌ Collection '{collection_name}' not found. "
                f"Available collections: {available if available else 'None'}"
            )
            raise CollectionNotFoundError(collection_name)

        logger.debug(f"Retrieved collection: {collection_name}")
        return self._collections[collection_name]

    def query(
        self, collection_name: str, query_text: str, top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        """
        Query collection for similar documents.

        Args:
            collection_name: Collection to query
            query_text: User's question
            top_k: Number of results to return (defaults to settings.top_k_chunks)

        Returns:
            List of RetrievedChunk, sorted by relevance (highest first)

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        logger.debug(f"Querying collection '{collection_name}' with query: '{query_text[:100]}'")

        # Get the collection
        collection = self.get_collection(collection_name)

        # Use default from settings if not provided
        if top_k is None:
            top_k = settings.top_k_chunks

        logger.debug(f"Requesting top-{top_k} chunks from ChromaDB...")

        # Query with relevance scores (normalized 0-1, higher is better)
        try:
            results = collection.similarity_search_with_relevance_scores(
                query_text, k=top_k
            )
            logger.debug(f"ChromaDB returned {len(results)} results")
        except Exception as e:
            logger.error(f"❌ ChromaDB query failed: {e}", exc_info=True)
            raise

        # Convert to RetrievedChunk objects
        chunks = []
        for i, (doc, score) in enumerate(results):
            chunk = RetrievedChunk(
                content=doc.page_content,
                page_number=doc.metadata.get("page", 0) + 1,  # Convert 0-indexed to 1-indexed
                chunk_index=doc.metadata.get("chunk_index", 0),
                relevance_score=score,
                source=doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
            )
            chunks.append(chunk)
            logger.debug(
                f"  Result {i+1}: Page {chunk.page_number}, "
                f"Score {chunk.relevance_score:.3f}, "
                f"Length {len(chunk.content)} chars"
            )

        logger.debug(f"Converted {len(chunks)} results to RetrievedChunk objects")

        # Results are already sorted by relevance (highest first) from Chroma
        return chunks

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deleted, False if didn't exist
        """
        if collection_name not in self._collections:
            return False

        # Get and delete the collection
        collection = self._collections[collection_name]
        collection.delete_collection()

        # Remove from tracking dict
        del self._collections[collection_name]

        return True

    def list_collections(self) -> list[str]:
        """
        List all active collection names.

        Returns:
            List of collection names
        """
        collection_list = list(self._collections.keys())
        logger.debug(f"Active collections: {collection_list if collection_list else 'None'}")
        return collection_list


def get_vector_store_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> VectorStoreService:
    """
    FastAPI dependency function to get the singleton vector store service instance.

    Args:
        embedding_service: Injected embedding service

    Returns:
        VectorStoreService singleton instance
    """
    # Get or create singleton instance
    return VectorStoreService(
        embedding_service=embedding_service,
        persist_directory=settings.chroma_persist_dir,
    )