from __future__ import annotations


def split_semantic_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.split('\n\n') if block.strip()]


def split_sentence_like(block: str) -> list[str]:
    parts = []
    current = []
    for ch in block:
        current.append(ch)
        if ch in '.!?\n':
            piece = ''.join(current).strip()
            if piece:
                parts.append(piece)
            current = []
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts if parts else [block.strip()]


def semantic_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    blocks = split_semantic_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    current = ''

    for block in blocks:
        for sentence in split_sentence_like(block):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            overlap = current[max(0, len(current) - chunk_overlap) :].strip() if current else ''
            current = f"{overlap} {sentence}".strip() if overlap else sentence

            while len(current) > chunk_size:
                chunks.append(current[:chunk_size].strip())
                current = current[max(0, chunk_size - chunk_overlap) :].strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks
