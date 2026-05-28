from agentic_mm_rag.schemas import EvidenceCard, Locator, ScoreParts, ToolResponse


def test_tool_response_serializes_evidence_with_content_limit():
    card = EvidenceCard(
        id="ev-1",
        source_type="doc",
        modality="text",
        source_id="doc-1",
        locator=Locator(file_path="paper.pdf", page_idx=2),
        content="abcdef",
        score=0.5,
        score_parts=ScoreParts(text=0.4, graph=0.1),
    )

    payload = ToolResponse(ok=True, tool="doc_text_seek", evidence=[card]).to_dict(
        content_chars=3
    )

    assert payload["ok"] is True
    assert payload["evidence"][0]["content"] == "abc..."
    assert payload["evidence"][0]["locator"] == {
        "file_path": "paper.pdf",
        "page_idx": 2,
    }
    assert payload["evidence"][0]["score_parts"]["text"] == 0.4
