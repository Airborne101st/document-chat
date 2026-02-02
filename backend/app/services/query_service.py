"""Query service that orchestrates the RAG pipeline."""

import logging
from typing import AsyncIterator

from fastapi import Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.exceptions import LLMError, NoActiveSessionError, QueryError
from app.core.session import SessionManager, get_session_manager
from app.models.schemas import Source
from app.services.llm_service import LLMService, get_llm_service
from app.services.vector_store_service import (
    RetrievedChunk,
    VectorStoreService,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)

# Prompt templates
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct in your answers
- Reference specific parts of the document when relevant
- Do not make up information not present in the context"""

RAG_USER_PROMPT_TEMPLATE = """Context from document:
{context}

---

Question: {question}

Answer:"""


class QueryContext(BaseModel):
    """Context built from retrieved chunks for the LLM."""

    chunks: list[RetrievedChunk]
    formatted_context: str  # The actual text to inject into prompt


class StreamingQueryResponse(BaseModel):
    """Wrapper for streaming response data."""

    token: str | None = None
    sources: list[Source] | None = None
    is_done: bool = False
    error: str | None = None


class QueryService:
    """
    Orchestrates the RAG query pipeline:
    retrieve relevant chunks → build prompt → stream LLM response
    """

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        llm_service: LLMService,
        session_manager: SessionManager,
        top_k: int = 3,
    ):
        """
        Initialize with required dependencies.

        Args:
            vector_store_service: For retrieving relevant chunks
            llm_service: For generating responses
            session_manager: For checking session state
            top_k: Number of chunks to retrieve (default: 3)
        """
        self._vector_store = vector_store_service
        self._llm = llm_service
        self._session_manager = session_manager
        self._top_k = top_k

    async def query_stream(
        self, question: str
    ) -> AsyncIterator[StreamingQueryResponse]:
        """
        Execute RAG query with streaming response.

        Steps:
        1. Validate session is active
        2. Retrieve top-k relevant chunks from vector store
        3. Build prompt with context
        4. Stream LLM response
        5. Yield sources after streaming completes

        Args:
            question: User's question

        Yields:
            StreamingQueryResponse objects:
            - First N yields: tokens (token field set)
            - Then: sources (sources field set)
            - Finally: done signal (is_done=True)

        Raises:
            NoActiveSessionError: If no document is loaded
            QueryError: If query processing fails
        """
        try:
            logger.info(f"▶ Starting RAG query: '{question[:100]}'")

            # Step 1: Validate session is active
            logger.debug("Step 1: Validating session...")
            if not self._session_manager.is_active:
                logger.error("❌ Query failed: No active session")
                raise NoActiveSessionError()

            collection_name = self._session_manager.collection_name
            if not collection_name:
                logger.error("❌ Query failed: No collection name in session")
                raise NoActiveSessionError()

            logger.info(f"✓ Using collection: {collection_name}")

            # Step 2: Retrieve relevant chunks
            logger.debug(f"Step 2: Retrieving top-{self._top_k} relevant chunks...")
            try:
                context = self._retrieve_context(question, collection_name)
                avg_score = sum(c.relevance_score for c in context.chunks) / len(context.chunks) if context.chunks else 0
                logger.info(
                    f"✓ Retrieved {len(context.chunks)} chunks (avg score: {avg_score:.3f})"
                )
                if logger.isEnabledFor(logging.DEBUG):
                    for i, chunk in enumerate(context.chunks, 1):
                        logger.debug(
                            f"  Chunk {i}: Page {chunk.page_number}, "
                            f"Score {chunk.relevance_score:.3f}, "
                            f"Length {len(chunk.content)} chars"
                        )
            except Exception as e:
                logger.error(f"❌ Failed to retrieve context: {e}", exc_info=True)
                raise QueryError(
                    "Failed to retrieve relevant context from vector store",
                    original_error=e,
                ) from e

            # Step 3: Build prompt
            logger.debug("Step 3: Building prompt...")
            prompt = self._build_prompt(question, context)
            logger.debug(f"✓ Prompt built: {len(prompt)} characters")

            # Step 4: Stream LLM response
            logger.debug("Step 4: Streaming LLM response...")
            token_count = 0
            try:
                async for token in self._llm.generate_stream(
                    prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT
                ):
                    token_count += 1
                    yield StreamingQueryResponse(token=token)

                logger.info(f"✓ Streamed {token_count} tokens from LLM")
            except LLMError as e:
                logger.error(f"❌ LLM generation failed after {token_count} tokens: {e}", exc_info=True)
                raise QueryError(
                    "Failed to generate response from LLM", original_error=e
                ) from e
            except Exception as e:
                logger.error(f"❌ Unexpected error during LLM streaming: {e}", exc_info=True)
                raise

            # Step 5: Yield sources after streaming completes
            logger.debug("Step 5: Sending source citations...")
            sources = self._chunks_to_sources(context.chunks)
            yield StreamingQueryResponse(sources=sources)
            logger.info(f"✓ Sent {len(sources)} source citations")

            # Finally: Done signal
            logger.debug("Sending completion signal")
            yield StreamingQueryResponse(is_done=True)
            logger.info("■ Query completed successfully")

        except NoActiveSessionError:
            logger.error("❌ Query aborted: No active session")
            raise
        except QueryError:
            logger.error("❌ Query aborted: QueryError raised")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during query: {e}", exc_info=True)
            raise QueryError("An unexpected error occurred during query", e) from e

    def _retrieve_context(
        self, question: str, collection_name: str
    ) -> QueryContext:
        """
        Retrieve relevant chunks and format as context.

        Args:
            question: User's question
            collection_name: ChromaDB collection to query

        Returns:
            QueryContext with chunks and formatted string
        """
        # Retrieve chunks from vector store
        chunks = self._vector_store.query(
            collection_name=collection_name, query_text=question, top_k=self._top_k
        )

        # Format chunks into context string
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            formatted_parts.append(
                f"[Chunk {i} - Page {chunk.page_number}, Score: {chunk.relevance_score:.2f}]\n{chunk.content}"
            )

        formatted_context = "\n\n".join(formatted_parts)

        return QueryContext(chunks=chunks, formatted_context=formatted_context)

    def _build_prompt(self, question: str, context: QueryContext) -> str:
        """
        Build the user prompt with context injected.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return RAG_USER_PROMPT_TEMPLATE.format(
            context=context.formatted_context, question=question
        )

    def _chunks_to_sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        """
        Convert RetrievedChunks to Source models for response.

        Args:
            chunks: List of retrieved chunks

        Returns:
            List of Source objects
        """
        sources = []
        for chunk in chunks:
            source = Source(
                content=chunk.content,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                relevance_score=chunk.relevance_score,
                filename=chunk.source,
            )
            sources.append(source)

        return sources


def get_query_service(
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    llm_service: LLMService = Depends(get_llm_service),
    session_manager: SessionManager = Depends(get_session_manager),
    settings: Settings = Depends(get_settings),
) -> QueryService:
    """
    FastAPI dependency function to get a query service instance.

    Args:
        vector_store_service: Injected vector store service
        llm_service: Injected LLM service
        session_manager: Injected session manager
        settings: Injected application settings

    Returns:
        QueryService instance
    """
    return QueryService(
        vector_store_service=vector_store_service,
        llm_service=llm_service,
        session_manager=session_manager,
        top_k=settings.top_k_chunks,
    )