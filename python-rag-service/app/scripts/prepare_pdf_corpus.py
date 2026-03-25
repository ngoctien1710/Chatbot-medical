from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.pdf_markdown_service import PdfMarkdownConverter
from app.services.rag_service import rag_service
from app.settings import settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert PDF corpus to markdown and ingest into RAG index.')
    parser.add_argument('--source-dir', type=Path, default=settings.pdf_source_dir)
    parser.add_argument('--output-dir', type=Path, default=settings.data_raw_dir)
    parser.add_argument('--no-reset', action='store_true', help='Keep existing vector index and append documents.')
    return parser


def main() -> None:
    args = _parser().parse_args()

    converter = PdfMarkdownConverter(source_dir=args.source_dir, output_dir=args.output_dir)
    converted = converter.convert_all()

    documents, chunks = rag_service.ingest(reset=not args.no_reset)

    payload = {
        'source_dir': str(args.source_dir),
        'output_dir': str(args.output_dir),
        'pdf_files_seen': len(converted),
        'pdf_files_converted': len([item for item in converted if not item.skipped]),
        'pdf_files_skipped': len([item for item in converted if item.skipped]),
        'ocr_pages_total': sum(item.used_ocr_pages for item in converted),
        'ingest_documents': documents,
        'ingest_chunks': chunks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
