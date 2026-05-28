"""Lightweight GraphML reader for exported LightRAG/VideoRAG graphs."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
CACHE_VERSION = 1


@dataclass(slots=True)
class GraphNode:
    id: str
    data: dict[str, str]


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    data: dict[str, str]


@dataclass(slots=True)
class TypedEdgeEvidence:
    edge_type: str
    source_chunk_ids: list[str]
    neighbor_chunk_ids: list[str]
    weight: float


def graph_cache_path(path: str | Path) -> Path:
    graph_path = Path(path)
    return graph_path.with_suffix(f"{graph_path.suffix}.typed_cache.json")


@lru_cache(maxsize=8)
def load_graph(path: str) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    tree = ET.parse(path)
    root = tree.getroot()

    key_names: dict[str, str] = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib.get("id")
        attr_name = key.attrib.get("attr.name")
        if key_id and attr_name:
            key_names[key_id] = attr_name

    nodes: dict[str, GraphNode] = {}
    for node in root.findall(".//g:node", GRAPHML_NS):
        node_id = html.unescape(node.attrib.get("id", ""))
        data: dict[str, str] = {}
        for data_el in node.findall("g:data", GRAPHML_NS):
            key = key_names.get(data_el.attrib.get("key", ""), data_el.attrib.get("key", ""))
            value = html.unescape(data_el.text or "")
            data[key] = value
        nodes[node_id] = GraphNode(id=node_id, data=data)

    edges: list[GraphEdge] = []
    for edge in root.findall(".//g:edge", GRAPHML_NS):
        data = {}
        for data_el in edge.findall("g:data", GRAPHML_NS):
            key = key_names.get(data_el.attrib.get("key", ""), data_el.attrib.get("key", ""))
            value = html.unescape(data_el.text or "")
            data[key] = value
        edges.append(
            GraphEdge(
                source=html.unescape(edge.attrib.get("source", "")),
                target=html.unescape(edge.attrib.get("target", "")),
                data=data,
            )
        )
    return nodes, edges


def split_source_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split("<SEP>") if part]


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token not in {"the", "and", "for", "with", "from", "that", "this", "what", "when", "where"}
    }


def _normalize_edge_type(data: dict[str, str]) -> str:
    raw = (
        data.get("edge_type")
        or data.get("relation_type")
        or data.get("type")
        or data.get("label")
        or ""
    )
    text = raw.strip().strip('"').lower()
    if not text:
        return "semantic"
    if any(token in text for token in ("table", "tabular")):
        return "table"
    if any(token in text for token in ("figure", "image", "visual")):
        return "visual"
    if any(token in text for token in ("time", "date", "before", "after", "temporal")):
        return "temporal"
    if any(token in text for token in ("same", "cooccur", "semantic", "entity", "relation")):
        return "semantic"
    return text


def _typed_edge_adjacency_from_graph(
    nodes: dict[str, GraphNode],
    edges: list[GraphEdge],
) -> tuple[dict[str, dict[str, list[TypedEdgeEvidence]]], dict[str, int]]:
    adjacency: dict[str, dict[str, list[TypedEdgeEvidence]]] = {node_id: {} for node_id in nodes}
    degree_by_node: dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in edges:
        edge_type = _normalize_edge_type(edge.data)
        weight_s = edge.data.get("weight", "1")
        try:
            weight = float(weight_s.strip('"'))
        except ValueError:
            weight = 1.0
        source_chunk_ids = split_source_ids(edge.data.get("source_id"))
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
        forward = TypedEdgeEvidence(
            edge_type=edge_type,
            source_chunk_ids=source_chunk_ids,
            neighbor_chunk_ids=split_source_ids(target_node.data.get("source_id")) if target_node else [],
            weight=weight,
        )
        backward = TypedEdgeEvidence(
            edge_type=edge_type,
            source_chunk_ids=source_chunk_ids,
            neighbor_chunk_ids=split_source_ids(source_node.data.get("source_id")) if source_node else [],
            weight=weight,
        )
        adjacency.setdefault(edge.source, {}).setdefault(edge_type, []).append(forward)
        adjacency.setdefault(edge.target, {}).setdefault(edge_type, []).append(backward)
        degree_by_node[edge.source] = degree_by_node.get(edge.source, 0) + 1
        degree_by_node[edge.target] = degree_by_node.get(edge.target, 0) + 1
    return adjacency, degree_by_node


def build_typed_graph_cache_payload(path: str | Path) -> dict[str, Any]:
    graph_path = Path(path).resolve()
    nodes, edges = load_graph(str(graph_path))
    typed_adjacency, degree_by_node = _typed_edge_adjacency_from_graph(nodes, edges)
    return {
        "version": CACHE_VERSION,
        "graph_path": str(graph_path),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": {node_id: node.data for node_id, node in nodes.items()},
        "keyword_index": [
            [node_id, sorted(list(keywords))]
            for node_id, keywords in load_keyword_index(str(graph_path))
        ],
        "degree_by_node": degree_by_node,
        "typed_edge_adjacency": {
            node_id: {
                edge_type: [
                    {
                        "weight": evidence.weight,
                        "source_chunk_ids": evidence.source_chunk_ids,
                        "neighbor_chunk_ids": evidence.neighbor_chunk_ids,
                    }
                    for evidence in evidences
                ]
                for edge_type, evidences in bucket.items()
            }
            for node_id, bucket in typed_adjacency.items()
            if bucket
        },
    }


def build_typed_graph_cache(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    graph_path = Path(path).resolve()
    destination = Path(output_path).resolve() if output_path else graph_cache_path(graph_path)
    payload = build_typed_graph_cache_payload(graph_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    temp_path.replace(destination)
    return destination


@lru_cache(maxsize=8)
def load_keyword_index(path: str) -> list[tuple[str, set[str]]]:
    nodes, _edges = load_graph(path)
    index: list[tuple[str, set[str]]] = []
    for node in nodes.values():
        description = node.data.get("description", "")
        entity_type = node.data.get("entity_type", "")
        keywords = _keywords(f"{node.id} {description} {entity_type}")
        if keywords:
            index.append((node.id, keywords))
    return index


@lru_cache(maxsize=8)
def load_edge_adjacency(path: str) -> dict[str, list[GraphEdge]]:
    nodes, edges = load_graph(path)
    adjacency: dict[str, list[GraphEdge]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge)
        adjacency.setdefault(edge.target, []).append(edge)
    return adjacency


@lru_cache(maxsize=8)
def load_typed_edge_adjacency(path: str) -> dict[str, dict[str, list[TypedEdgeEvidence]]]:
    nodes, edges = load_graph(path)
    adjacency, _degree_by_node = _typed_edge_adjacency_from_graph(nodes, edges)
    return adjacency


@lru_cache(maxsize=8)
def load_graph_cache(path: str) -> dict[str, Any] | None:
    cache_file = graph_cache_path(path)
    if not cache_file.exists():
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != CACHE_VERSION:
        return None
    return payload


@lru_cache(maxsize=8)
def load_cached_nodes(path: str) -> dict[str, GraphNode] | None:
    payload = load_graph_cache(path)
    if payload is None:
        return None
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, dict):
        return None
    return {
        node_id: GraphNode(id=node_id, data=data if isinstance(data, dict) else {})
        for node_id, data in raw_nodes.items()
    }


@lru_cache(maxsize=8)
def load_cached_keyword_index(path: str) -> list[tuple[str, set[str]]] | None:
    payload = load_graph_cache(path)
    if payload is None:
        return None
    raw_keyword_index = payload.get("keyword_index")
    if not isinstance(raw_keyword_index, list):
        return None
    index: list[tuple[str, set[str]]] = []
    for item in raw_keyword_index:
        if not isinstance(item, list) or len(item) != 2:
            continue
        node_id, keywords = item
        if not isinstance(node_id, str) or not isinstance(keywords, list):
            continue
        index.append((node_id, {str(keyword) for keyword in keywords}))
    return index


@lru_cache(maxsize=8)
def load_cached_degree_by_node(path: str) -> dict[str, int] | None:
    payload = load_graph_cache(path)
    if payload is None:
        return None
    raw_degree = payload.get("degree_by_node")
    if not isinstance(raw_degree, dict):
        return None
    return {str(node_id): int(value) for node_id, value in raw_degree.items()}


@lru_cache(maxsize=8)
def load_cached_typed_edge_adjacency(path: str) -> dict[str, dict[str, list[TypedEdgeEvidence]]] | None:
    payload = load_graph_cache(path)
    if payload is None:
        return None
    raw_adjacency = payload.get("typed_edge_adjacency")
    if not isinstance(raw_adjacency, dict):
        return None
    adjacency: dict[str, dict[str, list[TypedEdgeEvidence]]] = {}
    for node_id, bucket in raw_adjacency.items():
        if not isinstance(bucket, dict):
            continue
        typed_bucket: dict[str, list[TypedEdgeEvidence]] = {}
        for edge_type, raw_evidences in bucket.items():
            if not isinstance(edge_type, str) or not isinstance(raw_evidences, list):
                continue
            evidences: list[TypedEdgeEvidence] = []
            for raw_evidence in raw_evidences:
                if not isinstance(raw_evidence, dict):
                    continue
                evidences.append(
                    TypedEdgeEvidence(
                        edge_type=edge_type,
                        source_chunk_ids=[
                            str(chunk_id) for chunk_id in raw_evidence.get("source_chunk_ids", [])
                        ],
                        neighbor_chunk_ids=[
                            str(chunk_id)
                            for chunk_id in raw_evidence.get("neighbor_chunk_ids", [])
                        ],
                        weight=float(raw_evidence.get("weight", 1.0)),
                    )
                )
            if evidences:
                typed_bucket[edge_type] = evidences
        if typed_bucket:
            adjacency[str(node_id)] = typed_bucket
    return adjacency


class GraphMLStore:
    def __init__(self, path: Path):
        self.path = path

    @property
    def nodes(self) -> dict[str, GraphNode]:
        path = str(self.path.resolve())
        cached = load_cached_nodes(path)
        if cached is not None:
            return cached
        return load_graph(path)[0]

    @property
    def edges(self) -> list[GraphEdge]:
        return load_graph(str(self.path.resolve()))[1]

    @property
    def keyword_index(self) -> list[tuple[str, set[str]]]:
        path = str(self.path.resolve())
        cached = load_cached_keyword_index(path)
        if cached is not None:
            return cached
        return load_keyword_index(path)

    @property
    def edge_adjacency(self) -> dict[str, list[GraphEdge]]:
        return load_edge_adjacency(str(self.path.resolve()))

    @property
    def typed_edge_adjacency(self) -> dict[str, dict[str, list[TypedEdgeEvidence]]]:
        path = str(self.path.resolve())
        cached = load_cached_typed_edge_adjacency(path)
        if cached is not None:
            return cached
        return load_typed_edge_adjacency(path)

    def get_node(self, entity_name: str) -> GraphNode | None:
        if entity_name in self.nodes:
            return self.nodes[entity_name]
        quoted = f'"{entity_name.strip(chr(34))}"'
        return self.nodes.get(quoted)

    @staticmethod
    def source_chunks(node: GraphNode | None) -> list[str]:
        if node is None:
            return []
        return split_source_ids(node.data.get("source_id"))

    def node_edges(self, entity_name: str) -> list[GraphEdge]:
        node = self.get_node(entity_name)
        if node is None:
            return []
        return self.edge_adjacency.get(node.id, [])

    def node_degree(self, entity_name: str) -> int:
        node = self.get_node(entity_name)
        if node is None:
            return 0
        cached_degree = load_cached_degree_by_node(str(self.path.resolve()))
        if cached_degree is not None:
            return cached_degree.get(node.id, 0)
        return len(self.node_edges(entity_name))

    def keyword_seed_entities(self, query_text: str, *, top_k: int = 8) -> list[str]:
        query_terms = _keywords(query_text)
        if not query_terms:
            return []
        scored: list[tuple[int, str]] = []
        for node_id, keywords in self.keyword_index:
            overlap = len(query_terms & keywords)
            if overlap > 0:
                scored.append((overlap, node_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [node_id for _score, node_id in scored[:top_k]]

    @staticmethod
    def edge_weight(edge: GraphEdge) -> float:
        weight_s = edge.data.get("weight", "1")
        try:
            return float(weight_s.strip('"'))
        except ValueError:
            return 1.0

    def edge_type(self, edge: GraphEdge) -> str:
        return _normalize_edge_type(edge.data)

    def typed_edge_evidence(
        self,
        entity_names: list[str],
    ) -> dict[str, list[TypedEdgeEvidence]]:
        typed: dict[str, list[TypedEdgeEvidence]] = {}
        for entity_name in entity_names:
            node = self.get_node(entity_name)
            if node is None:
                continue
            for edge_type, evidences in self.typed_edge_adjacency.get(node.id, {}).items():
                typed.setdefault(edge_type, []).extend(evidences)
        return typed

    def clean_data(self, node: GraphNode) -> dict[str, Any]:
        """Convert GraphML data dictionary to a cleaner version for Agent inspection."""
        data = dict(node.data)
        data["entity_name"] = node.id.strip('"')
        return data

    def get_typed_edges(self, entity_name: str) -> list[dict[str, Any]]:
        """Return raw edge data with normalized types for Agent inspection."""
        node = self.get_node(entity_name)
        if node is None:
            return []
        edges = self.node_edges(entity_name)
        results = []
        for edge in edges:
            results.append({
                "source": edge.source.strip('"'),
                "target": edge.target.strip('"'),
                "type": self.edge_type(edge),
                "description": edge.data.get("description", ""),
                "weight": self.edge_weight(edge)
            })
        return results

    def related_chunk_scores(self, entity_names: list[str]) -> dict[str, float]:
        """Score chunks using LightRAG/VideoRAG-style source and one-hop relation evidence."""

        scores: dict[str, float] = {}
        for order, entity_name in enumerate(entity_names):
            node = self.get_node(entity_name)
            if node is None:
                continue
            base = 1.0 / (order + 1)
            for chunk_id in self.source_chunks(node):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + base
            typed_edges = self.typed_edge_evidence([entity_name])
            for evidences in typed_edges.values():
                for evidence in evidences:
                    relation_boost = base * min(evidence.weight, 10.0) / 10.0
                    for chunk_id in evidence.source_chunk_ids:
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + relation_boost
                    for chunk_id in evidence.neighbor_chunk_ids:
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + relation_boost * 0.5
        return scores

    def typed_related_chunk_scores(
        self,
        entity_names: list[str],
        *,
        edge_type_weights: dict[str, float] | None = None,
        max_neighbor_chunks_per_edge: int | None = None,
        allowed_edge_types: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Score chunks while preserving typed-edge contributions."""

        edge_type_weights = edge_type_weights or {
            "semantic": 1.0,
            "temporal": 1.15,
            "table": 1.2,
            "visual": 1.1,
        }
        allowed_edge_type_set = set(allowed_edge_types) if allowed_edge_types is not None else None
        scores: dict[str, float] = {}
        typed_breakdown: dict[str, dict[str, float]] = {}
        for order, entity_name in enumerate(entity_names):
            node = self.get_node(entity_name)
            if node is None:
                continue
            base = 1.0 / (order + 1)
            for chunk_id in self.source_chunks(node):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + base
                typed_breakdown.setdefault(chunk_id, {})["entity_source"] = (
                    typed_breakdown.setdefault(chunk_id, {}).get("entity_source", 0.0) + base
                )
            typed_edges = self.typed_edge_evidence([entity_name])
            for edge_type, evidences in typed_edges.items():
                if allowed_edge_type_set is not None and edge_type not in allowed_edge_type_set:
                    continue
                edge_multiplier = edge_type_weights.get(edge_type, 0.9)
                for evidence in evidences:
                    relation_boost = base * min(evidence.weight, 10.0) / 10.0 * edge_multiplier
                    for chunk_id in evidence.source_chunk_ids:
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + relation_boost
                        typed_breakdown.setdefault(chunk_id, {})[edge_type] = (
                            typed_breakdown.setdefault(chunk_id, {}).get(edge_type, 0.0)
                            + relation_boost
                        )
                    neighbor_chunk_ids = evidence.neighbor_chunk_ids
                    if max_neighbor_chunks_per_edge is not None:
                        neighbor_chunk_ids = neighbor_chunk_ids[:max_neighbor_chunks_per_edge]
                    for chunk_id in neighbor_chunk_ids:
                        neighbor_boost = relation_boost * 0.5
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + neighbor_boost
                        typed_breakdown.setdefault(chunk_id, {})[f"{edge_type}_neighbor"] = (
                            typed_breakdown.setdefault(chunk_id, {}).get(
                                f"{edge_type}_neighbor", 0.0
                            )
                            + neighbor_boost
                        )
        return scores, typed_breakdown

    @staticmethod
    def clean_data(node: GraphNode | None) -> dict[str, Any]:
        if node is None:
            return {}
        return {
            "entity_name": node.id,
            "entity_type": node.data.get("entity_type"),
            "entity_id": node.data.get("entity_id"),
            "description": node.data.get("description", ""),
            "source_id": node.data.get("source_id", ""),
            "file_path": node.data.get("file_path"),
        }
