from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PdfMarkdownError(RuntimeError):
    pass


@dataclass
class ConversionResult:
    source_pdf: Path
    output_md: Path
    used_ocr_pages: int
    total_pages: int
    skipped: bool


class PdfMarkdownConverter:
    """Convert PDF documents to markdown files for downstream RAG ingestion."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        min_text_chars: int = 60,
        ocr_dpi: int = 220,
    ) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.min_text_chars = min_text_chars
        self.ocr_dpi = ocr_dpi
        self._ocr_engine: Any = None
        self._manifest_path = self.output_dir / '.pdf_manifest.json'

    def _load_manifest(self) -> dict[str, dict[str, str]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise PdfMarkdownError('Unable to parse conversion manifest file.') from exc

    def _save_manifest(self, manifest: dict[str, dict[str, str]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _sha256(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(65536), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_stem(self, file_path: Path) -> str:
        stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', file_path.stem).strip('_')
        return stem or 'document'

    def _normalize_text(self, text: str) -> str:
        collapsed = text.replace('\r\n', '\n').replace('\r', '\n')
        collapsed = re.sub(r'\n{3,}', '\n\n', collapsed)
        return collapsed.strip()

    def _ensure_ocr_engine(self) -> Any:
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

    def _extract_pages(self, pdf_path: Path) -> tuple[list[str], int]:
        try:
            import fitz
        except Exception as exc:
            raise PdfMarkdownError('PyMuPDF is required for PDF extraction.') from exc

        pages: list[str] = []
        used_ocr_pages = 0

        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = self._normalize_text(page.get_text('text'))
                if len(page_text) < self.min_text_chars:
                    page_text = self._run_ocr(page)
                    if page_text:
                        used_ocr_pages += 1
                pages.append(page_text)

        return pages, used_ocr_pages

    def _markdown_from_pages(self, source_pdf: Path, pages: list[str]) -> str:
        now_text = datetime.now(timezone.utc).isoformat()
        lines = [
            f"# {source_pdf.stem}",
            '',
            f"- Source PDF: {source_pdf.name}",
            f"- Total pages: {len(pages)}",
            f"- Converted at (UTC): {now_text}",
            '',
        ]

        for idx, page_text in enumerate(pages, start=1):
            lines.append(f"## Page {idx}")
            lines.append('')
            lines.append(page_text or '[EMPTY_PAGE]')
            lines.append('')

        return '\n'.join(lines).strip() + '\n'

    def convert_all(self) -> list[ConversionResult]:
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
            if existing and existing.get('sha256') == file_hash and output_md.exists():
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
            markdown_text = self._markdown_from_pages(pdf_path, pages)
            output_md.write_text(markdown_text, encoding='utf-8')

            manifest[manifest_key] = {
                'sha256': file_hash,
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
