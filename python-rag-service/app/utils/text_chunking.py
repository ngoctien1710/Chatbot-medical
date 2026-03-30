from __future__ import annotations

import re


class SemanticChunkingError(RuntimeError):
    pass


def _normalize_text(text: str) -> str:
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


def _segment_text(text: str) -> list[str]:
    """Split by loose structure boundaries before token windows."""
    content = _normalize_text(text)
    if not content:
        return []
    segments = [part.strip() for part in re.split(r'\n\s*\n+', content) if part.strip()]
    return segments or [content]


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\S+', text)


def _window_tokens(tokens: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [' '.join(tokens)]

    pieces: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = ' '.join(tokens[start:end]).strip()
        if chunk:
            pieces.append(chunk)
        if end >= len(tokens):
            break
        start += step

    return pieces


def _merge_short_chunks(chunks: list[str], min_chunk_chars: int, max_chunk_chars: int) -> list[str]:
    if not chunks:
        return []

    merged: list[str] = []
    for chunk in chunks:
        piece = chunk.strip()
        if not piece:
            continue
        can_merge = merged and (len(merged[-1]) + 1 + len(piece) <= max_chunk_chars)
        if len(piece) < min_chunk_chars and can_merge:
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)

    if len(merged) >= 2 and len(merged[0]) < min_chunk_chars:
        merged[1] = f"{merged[0]} {merged[1]}".strip()
        merged = merged[1:]

    return [item for item in merged if item]


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


def semantic_chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[str]:
    """Token-based chunking with minimum chunk length post-processing."""
    content = _normalize_text(text)
    if not content:
        return []

    if chunk_overlap >= chunk_size:
        raise SemanticChunkingError('chunk_overlap phai nho hon chunk_size trong token-based chunking.')

    chunks: list[str] = []
    for segment in _segment_text(content):
        tokens = _tokenize(segment)
        if not tokens:
            continue
        chunks.extend(_window_tokens(tokens=tokens, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    max_chunk_chars = min(max(min_chunk_chars + 50, chunk_size * 12), 1200)
    chunks = _merge_short_chunks(
        chunks,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )

    bounded_chunks: list[str] = []
    max_window_overlap = min(200, max_chunk_chars // 8)
    for chunk in chunks:
        if len(chunk) <= max_chunk_chars:
            bounded_chunks.append(chunk)
            continue
        bounded_chunks.extend(
            _window_text(
                chunk,
                chunk_size=max_chunk_chars,
                chunk_overlap=max_window_overlap,
            )
        )

    chunks = _merge_short_chunks(
        bounded_chunks,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )
    chunks = [item for item in chunks if len(item) >= min_chunk_chars]
    if not chunks:
        raise SemanticChunkingError('Khong tao duoc chunk hop le tu token-based chunking.')

    return chunks
