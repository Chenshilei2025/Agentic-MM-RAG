#!/usr/bin/env python3
"""Build the document RAG store with HKUDS RAG-Anything.

The output directory is intentionally the same format consumed by
agentic_mm_rag.tools.runtime.stores.DocRAGStore.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_ROOT = PROJECT_ROOT / "data" / "doc_rag"
DEFAULT_PARSE_ROOT = PROJECT_ROOT / "data" / "doc_parse"
DEFAULT_RAGANYTHING_SOURCE = PROJECT_ROOT / "data" / "vendor"


def _add_import_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path.resolve()))


def _existing_paths(values: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        paths.append(str(path.resolve()))
    return paths


def _build_lightrag_functions(api_key: str | None, base_url: str | None):
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    llm_model = os.getenv("RAGANYTHING_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    vision_model = os.getenv(
        "RAGANYTHING_VISION_MODEL",
        os.getenv("VISION_MODEL", llm_model),
    )
    embedding_model = os.getenv(
        "RAGANYTHING_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    embedding_dim = int(
        os.getenv("RAGANYTHING_EMBEDDING_DIM", os.getenv("EMBEDDING_DIM", "1536"))
    )
    max_token_size = int(os.getenv("RAGANYTHING_EMBEDDING_MAX_TOKENS", "8192"))

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            payload = [
                {"role": "system", "content": system_prompt}
                if system_prompt
                else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                },
            ]
            return openai_complete_if_cache(
                vision_model,
                "",
                messages=[item for item in payload if item is not None],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )
    return llm_model_func, vision_model_func, embedding_func


async def _run(args: argparse.Namespace) -> None:
    _add_import_path(Path(args.raganything_source).expanduser())

    try:
        from raganything import RAGAnything, RAGAnythingConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot import raganything. Pass --raganything-source pointing to a "
            "directory that contains raganything/, or install RAG-Anything in "
            "this environment."
        ) from exc

    inputs = _existing_paths(args.inputs)
    working_dir = Path(args.working_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        parser=args.parser,
        parse_method=args.parse_method,
        parser_output_dir=str(output_dir),
        enable_image_processing=not args.disable_image_processing,
        enable_table_processing=not args.disable_table_processing,
        enable_equation_processing=not args.disable_equation_processing,
        max_concurrent_files=args.max_workers,
        recursive_folder_processing=not args.no_recursive,
        use_full_path=args.use_full_path,
    )
    llm_model_func, vision_model_func, embedding_func = _build_lightrag_functions(
        args.api_key,
        args.base_url,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    try:
        if len(inputs) == 1 and Path(inputs[0]).is_file():
            await rag.process_document_complete(
                inputs[0],
                output_dir=str(output_dir),
                parse_method=args.parse_method,
            )
        else:
            await rag.process_folder_complete(
                inputs[0] if len(inputs) == 1 and Path(inputs[0]).is_dir() else str(Path.cwd()),
                output_dir=str(output_dir),
                parse_method=args.parse_method,
                recursive=not args.no_recursive,
                max_workers=args.max_workers,
                file_extensions=args.extensions,
            ) if len(inputs) == 1 and Path(inputs[0]).is_dir() else await rag.process_documents_with_rag_batch(
                inputs,
                output_dir=str(output_dir),
                parse_method=args.parse_method,
                recursive=not args.no_recursive,
                max_workers=args.max_workers,
            )
    finally:
        finalize = getattr(rag, "finalize_storages", None)
        if callable(finalize):
            await finalize()

    print(f"Document RAG store written to: {working_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process documents with HKUDS RAG-Anything into data/doc_rag.",
    )
    parser.add_argument("inputs", nargs="+", help="Document files or directories.")
    parser.add_argument("--working-dir", default=str(DEFAULT_DOC_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_PARSE_ROOT))
    parser.add_argument(
        "--raganything-source",
        default=str(DEFAULT_RAGANYTHING_SOURCE),
        help="Directory containing the raganything Python package.",
    )
    parser.add_argument("--parser", default=os.getenv("PARSER", "mineru"))
    parser.add_argument("--parse-method", default=os.getenv("PARSE_METHOD", "auto"))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_CONCURRENT_FILES", "1")))
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--use-full-path", action="store_true")
    parser.add_argument("--disable-image-processing", action="store_true")
    parser.add_argument("--disable-table-processing", action="store_true")
    parser.add_argument("--disable-equation-processing", action="store_true")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional extension allowlist for directory ingestion, e.g. .pdf .docx .md.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY or LLM_BINDING_API_KEY is required")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
