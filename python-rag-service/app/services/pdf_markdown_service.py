from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import settings


class PdfMarkdownError(RuntimeError):
    pass


@dataclass
class ConversionResult:
    source_pdf: Path
    output_md: Path
    used_ocr_pages: int
    total_pages: int
    skipped: bool


@dataclass
class ExtractedPage:
    text: str
    from_ocr: bool


class PdfMarkdownConverter:
    """Convert PDF documents to cleaned markdown for downstream RAG ingestion."""

    _meta_line_patterns = [
        re.compile(r'^\s*(source pdf|total pages|converted at)\s*[:：]', re.IGNORECASE),
        re.compile(r'^\s*#+\s*page\s+\d+\s*$', re.IGNORECASE),
        re.compile(r'^\s*(page|trang)\s+\d+(\s*/\s*\d+)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*\[empty_page\]\s*$', re.IGNORECASE),
        re.compile(r'^\s*trang\s+chủ\s*/.*$', re.IGNORECASE),
        re.compile(r'^\s*\d+\s*$', re.IGNORECASE),
    ]
    _toc_markers = (
        'muc luc',
        'table of contents',
        'contents',
    )
    _reference_markers = (
        'tai lieu tham khao',
        'danh muc tai lieu tham khao',
        'references',
        'reference',
        'bibliography',
    )

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        min_text_chars: int | None = None,
        ocr_dpi: int | None = None,
    ) -> None:
        # Initialize converter and load OCR/cleanup settings.
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.min_text_chars = min_text_chars if min_text_chars is not None else settings.ocr_min_text_chars
        self.ocr_dpi = ocr_dpi if ocr_dpi is not None else settings.ocr_dpi
        self._ocr_engine: Any = None
        self._glm_client: Any = None
        self._glm_unavailable = False
        self._manifest_path = self.output_dir / '.pdf_manifest.json'
        self._cleaner_signature = self._build_cleaner_signature()

    def _build_cleaner_signature(self) -> str:
        # Build a config signature to know when markdown must be regenerated.
        signature_payload = {
            'min_text_chars': self.min_text_chars,
            'ocr_dpi': self.ocr_dpi,
            'strip_toc': settings.clean_strip_toc,
            'strip_references': settings.clean_strip_references,
            'glm_cleanup_enabled': settings.glm_cleanup_enabled,
            'glm_cleanup_model': settings.glm_cleanup_model,
            'glm_cleanup_max_chunk_chars': settings.glm_cleanup_max_chunk_chars,
            'glm_cleanup_max_chunks_per_doc': settings.glm_cleanup_max_chunks_per_doc,
        }
        raw = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _load_manifest(self) -> dict[str, dict[str, str]]:
        # Load cached conversion manifest if it exists.
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise PdfMarkdownError('Unable to parse conversion manifest file.') from exc

    def _save_manifest(self, manifest: dict[str, dict[str, str]]) -> None:
        # Persist manifest after conversion.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _sha256(self, file_path: Path) -> str:
        # Compute file content hash to detect PDF changes.
        digest = hashlib.sha256()
        with file_path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(65536), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_stem(self, file_path: Path) -> str:
        # Normalize output filename stem for filesystem safety.
        stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', file_path.stem).strip('_')
        return stem or 'document'

    def _normalize_text(self, text: str) -> str:
        # Normalize newlines and reduce excessive blank lines.
        collapsed = text.replace('\r\n', '\n').replace('\r', '\n')
        collapsed = re.sub(r'\n{3,}', '\n\n', collapsed)
        return collapsed.strip()

    @staticmethod
    def _collapse_inline_spaces(text: str) -> str:
        # Collapse repeated inline whitespace into a single space.
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _fold_for_match(text: str) -> str:
        # Remove diacritics and lowercase text for robust matching.
        normalized = unicodedata.normalize('NFKD', text)
        without_marks = ''.join(char for char in normalized if not unicodedata.combining(char))
        return without_marks.casefold()

    def _is_noise_line(self, line: str) -> bool:
        # Detect lines that are likely metadata or symbol-heavy noise.
        if not line:
            return True

        for pattern in self._meta_line_patterns:
            if pattern.match(line):
                return True

        symbol_chars = sum(1 for char in line if not char.isalnum() and not char.isspace())
        if symbol_chars >= max(5, int(len(line) * 0.45)) and len(re.findall(r'[A-Za-zÀ-ỹ0-9]', line)) < 4:
            return True

        return False

    @staticmethod
    def _is_navigation_line(folded_line: str) -> bool:
        # Detect website navigation or footer lines for removal.
        if folded_line.startswith('bai viet khac'):
            return True
        if folded_line.startswith('xem nhieu nhat'):
            return True
        if 'facebook.com' in folded_line or 'youtube.com' in folded_line:
            return True
        if folded_line.startswith('benh vien nguyen tri phuong'):
            return True
        return False

    def _looks_like_toc_line(self, line: str) -> bool:
        # Check whether a line looks like a table-of-contents entry.
        lowered = self._fold_for_match(line)
        if any(marker in lowered for marker in self._toc_markers):
            return True

        if re.search(r'\.{3,}\s*\d+\s*$', line):
            return True
        if re.search(r'_{3,}\s*\d+\s*$', line):
            return True

        if len(line) <= 180 and re.search(r'\b(chương|chuong|mục|muc|phần|phan|chapter|section)\b.+\d+\s*$', lowered):
            return True

        return False

    def _drop_reference_tail(self, lines: list[str]) -> list[str]:
        # Trim trailing reference section near document end.
        if not lines or not settings.clean_strip_references:
            return lines

        start_idx = -1
        scan_start = max(0, int(len(lines) * 0.35))
        for idx in range(scan_start, len(lines)):
            lowered = self._fold_for_match(lines[idx]).strip(' :.-')
            if any(lowered.startswith(marker) for marker in self._reference_markers):
                start_idx = idx
                break

        if start_idx == -1:
            return lines

        return lines[:start_idx]

    def _clean_text_deterministic(self, text: str) -> str:
        # Rule-based cleanup: filter noise, merge paragraphs, deduplicate lines.
        normalized = self._normalize_text(text)
        if not normalized:
            return ''

        raw_lines = [line.strip() for line in normalized.split('\n')]
        filtered_lines: list[str] = []
        for line in raw_lines:
            if not line:
                filtered_lines.append('')
                continue

            folded = self._fold_for_match(line)
            if self._is_navigation_line(folded):
                continue
            if settings.clean_strip_references and any(marker in folded for marker in self._reference_markers):
                continue
            if self._is_noise_line(line):
                continue
            if settings.clean_strip_toc and self._looks_like_toc_line(line):
                continue
            filtered_lines.append(line)

        filtered_lines = self._drop_reference_tail(filtered_lines)

        paragraphs: list[str] = []
        current: list[str] = []

        for line in filtered_lines:
            if not line:
                if current:
                    paragraph = self._collapse_inline_spaces(' '.join(current))
                    if paragraph:
                        paragraphs.append(paragraph)
                    current = []
                continue
            current.append(line)

        if current:
            paragraph = self._collapse_inline_spaces(' '.join(current))
            if paragraph:
                paragraphs.append(paragraph)

        cleaned_paragraphs: list[str] = []
        previous = None
        for paragraph in paragraphs:
            candidate = paragraph.strip()
            if not candidate:
                continue
            if previous and candidate == previous:
                continue
            previous = candidate
            cleaned_paragraphs.append(candidate)

        return '\n\n'.join(cleaned_paragraphs).strip()

    def _split_for_glm_cleanup(self, text: str, max_chars: int) -> list[str]:
        # Split text into chunks within the character limit before GLM cleanup.
        paragraphs = [part.strip() for part in text.split('\n\n') if part.strip()]
        if not paragraphs:
            return []

        segments: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                segments.append(paragraph)
                continue

            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                piece = paragraph[start:end].strip()
                if piece:
                    segments.append(piece)
                if end >= len(paragraph):
                    break
                start = end

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for segment in segments:
            segment_len = len(segment)

            if current and (current_len + 2 + segment_len) > max_chars:
                chunks.append('\n\n'.join(current))
                current = [segment]
                current_len = segment_len
                continue

            current.append(segment)
            current_len += segment_len + (2 if current_len else 0)

        if current:
            chunks.append('\n\n'.join(current))

        return chunks

    def _ensure_glm_client(self) -> Any | None:
        # Initialize and cache Hugging Face Inference client for GLM cleanup.
        if not settings.glm_cleanup_enabled or self._glm_unavailable:
            return None
        if self._glm_client is not None:
            return self._glm_client

        try:
            from huggingface_hub import InferenceClient
        except Exception:
            self._glm_unavailable = True
            return None

        token = settings.hf_token or None
        self._glm_client = InferenceClient(
            model=settings.glm_cleanup_model,
            token=token,
            timeout=settings.glm_cleanup_timeout_seconds,
        )
        return self._glm_client

    @staticmethod
    def _normalize_glm_output(text: str) -> str:
        # Normalize GLM output by stripping code fences and redundant prefixes.
        cleaned = text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```[a-zA-Z0-9_\-]*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)

        cleaned = re.sub(r'^\s*(cleaned text|van ban da lam sach|văn bản đã làm sạch)\s*[:：]\s*', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _safe_glm_result(self, original: str, candidate: str) -> str:
        # Keep original text when GLM output is too short or invalid.
        normalized_candidate = self._normalize_glm_output(candidate)
        if not normalized_candidate:
            return original

        original_len = len(original.strip())
        candidate_len = len(normalized_candidate)
        min_len = max(60, int(original_len * 0.12))
        if candidate_len < min_len:
            return original

        return normalized_candidate

    def _glm_cleanup_chunk(self, chunk: str) -> str:
        # Call GLM to clean a single text chunk with a constrained prompt.
        client = self._ensure_glm_client()
        if client is None:
            return chunk

        system_prompt = (
            'Ban la bo lam sach van ban OCR cho tai lieu y khoa tieng Viet. '
            'Chi sua loi OCR, ngat dong, dau cau va xoa metadata/page marker/muc luc/tai lieu tham khao. '
            'Khong duoc bo sung kien thuc moi, khong duoc tom tat, khong duoc suy dien. '
            'Tra ve duy nhat van ban da lam sach.'
        )
        user_prompt = (
            '<SOURCE_TEXT>\n'
            f'{chunk}\n'
            '</SOURCE_TEXT>\n\n'
            'Hay tra ve van ban da lam sach.'
        )

        try:
            completion = client.chat.completions.create(
                model=settings.glm_cleanup_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0,
            )
        except Exception:
            self._glm_unavailable = True
            return chunk

        choices = getattr(completion, 'choices', [])
        if not choices:
            return chunk

        message = getattr(choices[0], 'message', None)
        content = ''
        if message is not None:
            content = getattr(message, 'content', '') or ''

        return self._safe_glm_result(chunk, content)

    def _clean_with_glm(self, text: str) -> str:
        # Run GLM cleanup across chunks and merge the results.
        if not text or not settings.glm_cleanup_enabled:
            return text

        max_chars = max(600, settings.glm_cleanup_max_chunk_chars)
        chunks = self._split_for_glm_cleanup(text, max_chars=max_chars)
        if not chunks:
            return text

        max_chunks = max(1, settings.glm_cleanup_max_chunks_per_doc)
        cleaned_chunks: list[str] = []

        for idx, chunk in enumerate(chunks):
            if idx >= max_chunks:
                cleaned_chunks.extend(chunks[idx:])
                break
            cleaned_chunks.append(self._glm_cleanup_chunk(chunk))

        return '\n\n'.join([piece.strip() for piece in cleaned_chunks if piece.strip()]).strip()

    def _ensure_ocr_engine(self) -> Any:
        # Initialize and cache OCR engine for fallback extraction.
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:
            raise PdfMarkdownError(
                'OCR fallback requested but rapidocr_onnxruntime is unavailable.'
            ) from exc
        self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _run_ocr(self, page: Any) -> str:
        # Render PDF page to image and run OCR to extract text.
        try:
            import numpy as np
        except Exception as exc:
            raise PdfMarkdownError('OCR fallback requires numpy to be installed.') from exc

        pixmap = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
        channels = max(1, pixmap.n)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, channels)

        ocr_engine = self._ensure_ocr_engine()
        results, _ = ocr_engine(image)
        if not results:
            return ''

        lines: list[str] = []
        for item in results:
            if len(item) < 2:
                continue
            text_candidate = item[1]
            if isinstance(text_candidate, tuple) and text_candidate:
                lines.append(str(text_candidate[0]).strip())
            else:
                lines.append(str(text_candidate).strip())
        return self._normalize_text('\n'.join([line for line in lines if line]))

    def _extract_pages(self, pdf_path: Path) -> tuple[list[ExtractedPage], int]:
        # Extract text page by page with OCR fallback for low-text pages.
        try:
            import fitz
        except Exception as exc:
            raise PdfMarkdownError('PyMuPDF is required for PDF extraction.') from exc

        pages: list[ExtractedPage] = []
        used_ocr_pages = 0

        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = self._normalize_text(page.get_text('text'))
                from_ocr = False
                if len(page_text) < self.min_text_chars:
                    page_text = self._run_ocr(page)
                    from_ocr = bool(page_text)
                    if page_text:
                        used_ocr_pages += 1
                pages.append(ExtractedPage(text=page_text, from_ocr=from_ocr))

        return pages, used_ocr_pages

    def _markdown_from_pages(self, pages: list[ExtractedPage]) -> str:
        # Merge page text and run the cleanup pipeline into markdown.
        raw_parts = [page.text for page in pages if page.text]
        if not raw_parts:
            return ''

        cleaned = self._clean_text_deterministic('\n\n'.join(raw_parts))
        cleaned = self._clean_with_glm(cleaned)
        cleaned = self._clean_text_deterministic(cleaned)

        if not cleaned:
            return ''

        return cleaned.strip() + '\n'

    def convert_all(self) -> list[ConversionResult]:
        # Convert all PDFs in source directory and update manifest cache.
        if not self.source_dir.exists():
            raise PdfMarkdownError(f'Source PDF directory not found: {self.source_dir}')

        self.output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted([path for path in self.source_dir.iterdir() if path.suffix.lower() == '.pdf'])
        if not pdf_files:
            return []

        manifest = self._load_manifest()
        results: list[ConversionResult] = []

        for pdf_path in pdf_files:
            file_hash = self._sha256(pdf_path)
            manifest_key = str(pdf_path.resolve())
            output_name = f"{self._safe_stem(pdf_path)}.md"
            output_md = self.output_dir / output_name

            existing = manifest.get(manifest_key)
            if (
                existing
                and existing.get('sha256') == file_hash
                and existing.get('cleaner_signature') == self._cleaner_signature
                and output_md.exists()
            ):
                results.append(
                    ConversionResult(
                        source_pdf=pdf_path,
                        output_md=output_md,
                        used_ocr_pages=int(existing.get('used_ocr_pages', '0')),
                        total_pages=int(existing.get('total_pages', '0')),
                        skipped=True,
                    )
                )
                continue

            pages, used_ocr_pages = self._extract_pages(pdf_path)
            markdown_text = self._markdown_from_pages(pages)
            output_md.write_text(markdown_text, encoding='utf-8')

            manifest[manifest_key] = {
                'sha256': file_hash,
                'cleaner_signature': self._cleaner_signature,
                'output_md': str(output_md.resolve()),
                'used_ocr_pages': str(used_ocr_pages),
                'total_pages': str(len(pages)),
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            }

            results.append(
                ConversionResult(
                    source_pdf=pdf_path,
                    output_md=output_md,
                    used_ocr_pages=used_ocr_pages,
                    total_pages=len(pages),
                    skipped=False,
                )
            )

        self._save_manifest(manifest)
        return results
