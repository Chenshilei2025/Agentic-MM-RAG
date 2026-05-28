import pytest

from agentic_mm_rag import create_runtime


def test_registry_profiles_are_curated():
    runtime = create_runtime()

    assert [item["name"] for item in runtime.tool_manifest("decision_agent")] == [
        "read_evidence"
    ]
    assert {item["name"] for item in runtime.tool_manifest("doc_text_subagent")} == {
        "doc_text_seek",
        "write_evidence",
    }
    assert {item["name"] for item in runtime.tool_manifest("video_visual_subagent")} == {
        "video_visual_seek",
        "write_evidence",
    }


def test_unknown_registry_profile_raises_value_error():
    runtime = create_runtime()

    with pytest.raises(ValueError):
        runtime.tool_manifest("missing")  # type: ignore[arg-type]
