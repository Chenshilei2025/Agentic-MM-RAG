from agentic_mm_rag.config import ModelDefaults


def test_model_defaults_route_strong_decision_and_specialists():
    models = ModelDefaults()

    assert models.decision == "gpt-4o"
    assert models.text_expert == "gpt-4o-mini"
    assert models.visual_expert == "gpt-4.1"
    assert models.graph_expert == "gpt-4o-mini"
