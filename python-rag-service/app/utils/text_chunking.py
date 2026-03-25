from __future__ import annotations

import importlib
from typing import Any


class SemanticChunkingError(RuntimeError):
    pass


def _window_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    content = text.strip()
    if not content:
        return []
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(content):
        end = min(len(content), start + chunk_size)
        piece = content[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(content):
            break
        start += step
    return chunks


def semantic_chunk_text(text: str, chunk_size: int, chunk_overlap: int, embeddings: Any) -> list[str]:
    content = text.strip()
    if not content:
        return []

    # Split very large documents into safe pre-segments to avoid embedding context overflow.
    pre_segments = _window_text(
        content,
        chunk_size=max(1500, chunk_size * 2),
        chunk_overlap=max(120, chunk_overlap),
    )

    try:
        module = importlib.import_module('langchain_experimental.text_splitter')
        semantic_chunker_cls = getattr(module, 'SemanticChunker')
        splitter = semantic_chunker_cls(embeddings=embeddings)
    except Exception as exc:
        raise SemanticChunkingError('Khong the khoi tao LangChain SemanticChunker.') from exc

    chunks: list[str] = []

    for segment in pre_segments:
        try:
            documents = splitter.create_documents([segment])
        except Exception as exc:
            raise SemanticChunkingError('LangChain SemanticChunker that bai khi tao semantic segments.') from exc

        if not documents:
            raise SemanticChunkingError('LangChain SemanticChunker khong tao duoc segment nao.')

        for doc in documents:
            chunks.extend(_window_text(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    if not chunks:
        raise SemanticChunkingError('Khong tao duoc chunk hop le tu ket qua semantic chunking.')

    return chunks
