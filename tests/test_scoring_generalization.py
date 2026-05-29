from agentic_mm_rag.tools.runtime.scoring import fuse_evidence_items


def test_temporal_delta_boost_uses_generic_group_terms():
    evidence = [
        {
            "id": "generic-delta",
            "source_type": "doc",
            "modality": "text",
            "source_id": "doc",
            "content": "The age cohort increased by +12 percentage points between 2010 and 2020.",
            "score": 0.1,
            "score_parts": {"text": 0.1},
        },
        {
            "id": "background",
            "source_type": "doc",
            "modality": "text",
            "source_id": "doc",
            "content": "The report discusses many results from 2010 and 2020.",
            "score": 0.5,
            "score_parts": {"text": 0.5},
        },
    ]

    response = fuse_evidence_items(
        evidence,
        query_text="Which age cohort had the largest increase from 2010 to 2020?",
        top_k=2,
    )

    assert response.data["items"][0]["id"] == "generic-delta"
