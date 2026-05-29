"""OpenAI-compatible embedding client for task and query vectorization."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class OpenAICompatibleEmbeddingClient:
    """Minimal /embeddings client for decomposed retrieval tasks.

    The orchestrator receives vectors on QueryContext, so applications can use
    this client to embed the original user question or subagent task queries
    before calling document/video seek tools. It intentionally has no dataset
    assumptions and no extra runtime dependencies.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.dimensions = dimensions
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")
        if not self.base_url:
            raise RuntimeError("OPENAI_BASE_URL is required for embeddings")

    def embed(self, text: str) -> list[float]:
        try:
            url = f"{self.base_url}/embeddings"
            request_payload: dict[str, Any] = {"model": self.model, "input": text}
            if self.dimensions is not None:
                request_payload["dimensions"] = self.dimensions
            payload = json.dumps(request_payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            vector = data["data"][0]["embedding"]
            return [float(x) for x in vector]
        except Exception as e:
            print(f"Embedding failed (Error: {e}), using zero-vector fallback.")
            return [0.0] * 1536
