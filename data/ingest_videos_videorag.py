#!/usr/bin/env python3
"""Build the video RAG store with HKUDS VideoRAG.

The output directory is intentionally the same format consumed by
agentic_mm_rag.tools.runtime.stores.VideoRAGStore.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data" / "video_rag"
DEFAULT_VIDEORAG_SOURCE = PROJECT_ROOT / "data" / "vendor"
DEFAULT_VIDEORAG_CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "checkpoints" / "videorag"


def _add_import_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path.resolve()))


def _require_existing_files(paths: list[str]) -> list[str]:
    resolved: list[str] = []
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        resolved.append(str(path.resolve()))
    return resolved


def _select_llm_config(name: str):
    try:
        from videorag import _llm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot import videorag. Pass --videorag-source pointing to a "
            "directory that contains videorag/, or install VideoRAG in this "
            "environment."
        ) from exc

    configs = {
        "openai": _llm.openai_config,
        "openai-mini": _llm.openai_4o_mini_config,
        "azure-openai": _llm.azure_openai_config,
        "ollama": _llm.ollama_config,
    }
    if hasattr(_llm, "deepseek_bge_config"):
        configs["deepseek-bge"] = _llm.deepseek_bge_config
    try:
        return configs[name]
    except KeyError as exc:
        choices = ", ".join(sorted(configs))
        raise ValueError(f"unknown --llm-config {name!r}; choose one of: {choices}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process videos with HKUDS VideoRAG into data/video_rag.",
    )
    parser.add_argument("videos", nargs="+", help="Video files to ingest.")
    parser.add_argument("--working-dir", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument(
        "--videorag-source",
        default=str(DEFAULT_VIDEORAG_SOURCE),
        help="Directory containing the videorag Python package.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_VIDEORAG_CHECKPOINT_ROOT),
        help=(
            "Directory containing MiniCPM-V-2_6-int4, faster-distil-whisper-large-v3, "
            "and .checkpoints/imagebind_huge.pth."
        ),
    )
    parser.add_argument(
        "--llm-config",
        default=os.getenv("VIDEORAG_LLM_CONFIG", "openai-mini"),
        help="openai, openai-mini, azure-openai, ollama, or deepseek-bge.",
    )
    parser.add_argument("--segment-length", type=int, default=int(os.getenv("VIDEORAG_SEGMENT_LENGTH", "30")))
    parser.add_argument("--rough-frames", type=int, default=int(os.getenv("VIDEORAG_ROUGH_FRAMES", "5")))
    parser.add_argument("--fine-frames", type=int, default=int(os.getenv("VIDEORAG_FINE_FRAMES", "15")))
    parser.add_argument("--segment-top-k", type=int, default=int(os.getenv("VIDEORAG_SEGMENT_TOP_K", "4")))
    parser.add_argument("--video-batch-size", type=int, default=int(os.getenv("VIDEORAG_VIDEO_BATCH_SIZE", "2")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.videorag_source).expanduser().resolve()
    _add_import_path(source_dir)

    try:
        from videorag import VideoRAG
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot import videorag. Pass --videorag-source pointing to a "
            "directory that contains videorag/, or install VideoRAG in this "
            "environment."
        ) from exc

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    videos = _require_existing_files(args.videos)
    working_dir = Path(args.working_dir).expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    if checkpoint_root.exists():
        os.chdir(checkpoint_root)

    llm_config = _select_llm_config(args.llm_config)
    rag = VideoRAG(
        llm=llm_config,
        working_dir=str(working_dir),
        video_segment_length=args.segment_length,
        rough_num_frames_per_segment=args.rough_frames,
        fine_num_frames_per_segment=args.fine_frames,
        segment_retrieval_top_k=args.segment_top_k,
        video_embedding_batch_num=args.video_batch_size,
    )
    rag.insert_video(video_path_list=videos)
    print(f"Video RAG store written to: {working_dir}")


if __name__ == "__main__":
    main()
