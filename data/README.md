# Data Ingestion

This folder keeps the processing scripts that build the stores consumed by the
runtime retrieval tools.

Scope: data parsing, multimodal analysis, graph construction, and vector-store
insertion only. These scripts do not run retrieval or question answering.

Outputs:

- `data/doc_rag`: RAG-Anything / LightRAG document store.
- `data/doc_parse`: parsed document assets from RAG-Anything.
- `data/video_rag`: VideoRAG video store.

## Configuration Files

- `data/ingestion.env.example`: environment variable template for both
  document and video ingestion.
- `data/README.md`: explains every command-line flag and environment variable.

Typical setup:

```bash
cp data/ingestion.env.example .env.ingestion
# edit .env.ingestion, then load it
set -a
. ./.env.ingestion
set +a
```

## Shared Paths

| Setting | Default | Stage | Meaning |
| --- | --- | --- | --- |
| `--working-dir` for docs | `data/doc_rag` | document output | LightRAG KV, graph, and vector store directory. |
| `--output-dir` for docs | `data/doc_parse` | document parsing | Parser artifacts and visual assets. |
| `--working-dir` for videos | `data/video_rag` | video output | VideoRAG KV, graph, and vector store directory. |
| `AGENTIC_MM_RAG_DOC_ROOT` | `data/doc_rag` | retrieval runtime | Runtime override for processed document store. |
| `AGENTIC_MM_RAG_VIDEO_ROOT` | `data/video_rag` | retrieval runtime | Runtime override for processed video store. |
| `AGENTIC_MM_RAG_DOC_VISUAL_ASSET_ROOTS` | empty | retrieval runtime | Extra roots for parsed document images/tables/charts. |

## Documents

Install RAG-Anything dependencies and parser dependencies before running the
script. The repository does not vendor RAG-Anything; install it in the active
environment or pass `--raganything-source` pointing to a local directory that
contains `raganything/`. The default path is `data/vendor`, which is ignored by
git and can be used for a local upstream checkout.

Pipeline covered by `ingest_docs_raganything.py`:

- parse documents with RAG-Anything parser, such as MinerU
- process text, images, tables, and equations
- insert text and multimodal chunks into LightRAG
- build the document knowledge graph
- write chunk/entity/relationship vectors under `data/doc_rag`

```bash
python data/ingest_docs_raganything.py /path/to/docs_or_file \
  --working-dir data/doc_rag \
  --output-dir data/doc_parse \
  --parser mineru
```

Document command-line flags:

| Flag | Default | Stage | Meaning |
| --- | --- | --- | --- |
| `inputs` | required | input | Files or directories to ingest. |
| `--working-dir` | `data/doc_rag` | output | Final LightRAG/RAG-Anything store. |
| `--output-dir` | `data/doc_parse` | parsing | Parser outputs and visual assets. |
| `--raganything-source` | `data/vendor` | import | Local directory containing `raganything/`, if not installed in the environment. |
| `--parser` | `PARSER` or `mineru` | parsing | `mineru`, `docling`, or `paddleocr`. |
| `--parse-method` | `PARSE_METHOD` or `auto` | parsing | `auto`, `ocr`, or `txt`. |
| `--max-workers` | `MAX_CONCURRENT_FILES` or `1` | batch | Concurrent document count. Keep low for GPU/OCR stability. |
| `--no-recursive` | false | input scan | Do not recurse into input directories. |
| `--use-full-path` | false | metadata | Store full source paths instead of basenames. |
| `--disable-image-processing` | false | multimodal | Skip image caption/analysis insertion. |
| `--disable-table-processing` | false | multimodal | Skip table interpretation insertion. |
| `--disable-equation-processing` | false | multimodal | Skip equation interpretation insertion. |
| `--extensions` | parser defaults | input scan | Directory allowlist, such as `.pdf .docx .md`. |
| `--api-key` | `OPENAI_API_KEY` or `LLM_BINDING_API_KEY` | model | OpenAI-compatible API key. |
| `--base-url` | `OPENAI_BASE_URL` or `LLM_BINDING_HOST` | model | OpenAI-compatible base URL. |

Document model environment:

| Variable | Default | Used For |
| --- | --- | --- |
| `RAGANYTHING_LLM_MODEL` | `LLM_MODEL` or `gpt-4o-mini` | Text analysis and graph extraction. |
| `RAGANYTHING_VISION_MODEL` | `VISION_MODEL` or LLM model | Image/chart/table visual analysis. |
| `RAGANYTHING_EMBEDDING_MODEL` | `EMBEDDING_MODEL` or `text-embedding-3-small` | Chunk/entity/relation vectors. |
| `RAGANYTHING_EMBEDDING_DIM` | `EMBEDDING_DIM` or `1536` | Vector dimension. Must match the embedding model. |
| `RAGANYTHING_EMBEDDING_MAX_TOKENS` | `8192` | Embedding batch token limit. |

Document output files expected under `data/doc_rag`:

| File | Meaning |
| --- | --- |
| `kv_store_doc_status.json` | Document status and source file metadata. |
| `kv_store_text_chunks.json` | Text and multimodal chunk payloads. |
| `kv_store_full_docs.json` | Full inserted document text, when LightRAG writes it. |
| `graph_chunk_entity_relation.graphml` | Document knowledge graph. |
| `vdb_chunks.json` | Chunk vectors. |
| `vdb_entities.json` | Entity vectors. |
| `vdb_relationships.json` | Relationship vectors. |

## Videos

Install VideoRAG dependencies and provide the required model checkpoints before
running the script. The repository does not vendor VideoRAG; install it in the
active environment or pass `--videorag-source` pointing to a local directory
that contains `videorag/`. The default path is `data/vendor`, which is ignored
by git and can be used for a local upstream checkout. VideoRAG still expects the
MiniCPM, Whisper, and ImageBind checkpoint paths described by the upstream
project.

Pipeline covered by `ingest_videos_videorag.py`:

- split videos into segments
- extract transcripts with Whisper
- caption segments with the VideoRAG vision model
- encode video segment features
- chunk merged segment text
- build the video knowledge graph
- write text/entity/video-segment vectors under `data/video_rag`

```bash
python data/ingest_videos_videorag.py /path/to/video.mp4 \
  --working-dir data/video_rag \
  --checkpoint-root /path/to/videorag-checkpoints \
  --llm-config openai-mini
```

`--llm-config` supports `openai`, `openai-mini`, `azure-openai`, `ollama`, and
`deepseek-bge` when that config exists in the local VideoRAG code.

Video command-line flags:

| Flag | Default | Stage | Meaning |
| --- | --- | --- | --- |
| `videos` | required | input | Video files to ingest. |
| `--working-dir` | `data/video_rag` | output | Final VideoRAG store. |
| `--videorag-source` | `data/vendor` | import | Local directory containing `videorag/`, if not installed in the environment. |
| `--checkpoint-root` | `data/checkpoints/videorag` | model files | Directory containing VideoRAG checkpoints. |
| `--llm-config` | `VIDEORAG_LLM_CONFIG` or `openai-mini` | model | LLM/embedding provider preset. |
| `--segment-length` | `VIDEORAG_SEGMENT_LENGTH` or `30` | parsing | Seconds per video segment. |
| `--rough-frames` | `VIDEORAG_ROUGH_FRAMES` or `5` | captioning | Frames sampled for segment captioning. |
| `--fine-frames` | `VIDEORAG_FINE_FRAMES` or `15` | reserved | Kept for VideoRAG compatibility. |
| `--segment-top-k` | `VIDEORAG_SEGMENT_TOP_K` or `4` | vector config | Stored VideoRAG segment retrieval top-k config. |
| `--video-batch-size` | `VIDEORAG_VIDEO_BATCH_SIZE` or `2` | vector encoding | Batch size for ImageBind video segment encoding. |

Video provider credentials:

| `--llm-config` | Required Environment |
| --- | --- |
| `openai` / `openai-mini` | `OPENAI_API_KEY`; optional OpenAI SDK base URL variables if your SDK setup uses them. |
| `azure-openai` | Azure OpenAI SDK environment, such as `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`. |
| `ollama` | `OLLAMA_HOST` if not using `http://127.0.0.1:11434`. |
| `deepseek-bge` | `DEEPSEEK_API_KEY` for chat and `SILICONFLOW_API_KEY` for `BAAI/bge-m3` embeddings. |

Video checkpoint layout expected by upstream VideoRAG:

```text
<checkpoint-root>/
  MiniCPM-V-2_6-int4/
  faster-distil-whisper-large-v3/
  .checkpoints/
    imagebind_huge.pth
```

Video output files expected under `data/video_rag`:

| File | Meaning |
| --- | --- |
| `kv_store_video_path.json` | Video ID to source path mapping. |
| `kv_store_video_segments.json` | Segment times, transcripts, captions, sampled frame times. |
| `kv_store_text_chunks.json` | Text chunks built from segment captions/transcripts. |
| `graph_chunk_entity_relation.graphml` | Video knowledge graph. |
| `vdb_chunks.json` | Text chunk vectors. |
| `vdb_entities.json` | Entity vectors. |
| `vdb_video_segment_feature.json` | ImageBind video segment vectors. |

## Runtime Paths

The default runtime paths in `config.py` now point at this folder. Override them
only when you store processed data elsewhere:

```bash
export AGENTIC_MM_RAG_DOC_ROOT="$PWD/data/doc_rag"
export AGENTIC_MM_RAG_VIDEO_ROOT="$PWD/data/video_rag"
export AGENTIC_MM_RAG_DOC_VISUAL_ASSET_ROOTS="$PWD/data/doc_parse"
```

The scripts are lightweight Python entrypoints. Parser, model, vector DB,
video, and checkpoint dependencies still need to be installed or available in
the active environment.
