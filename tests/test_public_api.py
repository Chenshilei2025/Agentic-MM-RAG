from agentic_mm_rag import QueryContext, create_runtime


def test_public_runtime_manifest_names():
    runtime = create_runtime()
    names = {item["name"] for item in runtime.tool_manifest()}

    assert names == {
        "doc_text_seek",
        "doc_visual_seek",
        "doc_graph_seek",
        "video_text_seek",
        "video_visual_seek",
        "video_graph_seek",
        "read_evidence",
        "write_evidence",
    }


def test_query_context_to_dict_is_stable():
    query = QueryContext(
        query_text="find evidence",
        doc_query_vector=[0.1, 0.2],
        top_k=3,
        metadata={"source": "test"},
    )

    data = query.to_dict()

    assert data["query_text"] == "find evidence"
    assert data["doc_query_vector"] == [0.1, 0.2]
    assert data["top_k"] == 3
    assert data["metadata"] == {"source": "test"}
