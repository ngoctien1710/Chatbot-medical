from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.services.rag_service import RetrievalItem, rag_service

DEFAULT_MODES = [
    'dense_only',
    'sparse_only',
    'hybrid_rrf',
    'cross_encoder_only',
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Benchmark retrieval latency and outputs across retrieval modes.')
    parser.add_argument('--query', type=str, default='', help='Single query to benchmark.')
    parser.add_argument('--query-file', type=Path, default=None, help='UTF-8 text file with one query per line.')
    parser.add_argument('--output', type=Path, default=None, help='Optional output JSON path.')
    parser.add_argument(
        '--include-hybrid-original',
        action='store_true',
        help='Also benchmark the default hybrid_original pipeline for baseline comparison.',
    )
    return parser


def _queries_from_args(args: argparse.Namespace) -> list[str]:
    if args.query.strip():
        return [args.query.strip()]

    if args.query_file:
        if not args.query_file.exists():
            raise FileNotFoundError(f'Query file not found: {args.query_file}')
        lines = [line.strip() for line in args.query_file.read_text(encoding='utf-8').splitlines()]
        queries = [line for line in lines if line]
        if queries:
            return queries

    return [
        'triệu chứng lâm sàng của bệnh ung thư bạch cầu cấp dòng lympho là gì',
    ]


def _serialize_items(items: list[RetrievalItem]) -> list[dict]:
    payload: list[dict] = []
    for item in items:
        payload.append(
            {
                'doc_id': item.doc_id,
                'chunk_id': item.chunk_id,
                'score': round(item.score, 4),
                'snippet_preview': item.text[:220],
            }
        )
    return payload


def _round_summary(summary: dict[str, float] | None) -> dict[str, float] | None:
    if not summary:
        return None
    return {
        'max': round(summary['max'], 4),
        'min': round(summary['min'], 4),
        'avg': round(summary['avg'], 4),
    }


def main() -> None:
    args = _parser().parse_args()
    queries = _queries_from_args(args)

    modes = list(DEFAULT_MODES)
    if args.include_hybrid_original:
        modes.append('hybrid_original')

    benchmark_started = time.perf_counter()
    results: list[dict] = []

    for query in queries:
        query_result = {
            'query': query,
            'modes': [],
        }

        for mode in modes:
            started = time.perf_counter()
            items, status, score_summary = rag_service.retrieve(query=query, mode=mode)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            query_result['modes'].append(
                {
                    'mode': mode,
                    'retrieval_status': status,
                    'latency_ms': elapsed_ms,
                    'top_k': len(items),
                    'score_summary': _round_summary(score_summary),
                    'snippets': _serialize_items(items),
                }
            )

        results.append(query_result)

    payload = {
        'query_count': len(queries),
        'modes': modes,
        'total_benchmark_ms': int((time.perf_counter() - benchmark_started) * 1000),
        'results': results,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
