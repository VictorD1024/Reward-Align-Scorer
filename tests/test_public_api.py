def test_flat_import_paths_remain_compatible():
    from raise_scorer.backends import EmbeddingBackend as GroupedEmbeddingBackend
    from raise_scorer.core import SemanticRewardScorer as GroupedSemanticRewardScorer
    from raise_scorer.embedding import EmbeddingBackend as FlatEmbeddingBackend
    from raise_scorer.runtime import RuntimeEventAdapter as GroupedRuntimeEventAdapter
    from raise_scorer.runtime_adapters import (
        RuntimeEventAdapter as FlatRuntimeEventAdapter,
    )
    from raise_scorer.scorer import SemanticRewardScorer as FlatSemanticRewardScorer
    from raise_scorer.workflow import WorkflowScorer as FlatWorkflowScorer
    from raise_scorer.workflows import WorkflowScorer as GroupedWorkflowScorer

    assert FlatEmbeddingBackend is GroupedEmbeddingBackend
    assert FlatSemanticRewardScorer is GroupedSemanticRewardScorer
    assert FlatRuntimeEventAdapter is GroupedRuntimeEventAdapter
    assert FlatWorkflowScorer is GroupedWorkflowScorer


def test_grouped_namespaces_have_focused_public_exports():
    import raise_scorer.backends as backends
    import raise_scorer.core as core
    import raise_scorer.experiments as experiments
    import raise_scorer.integrations as integrations
    import raise_scorer.runtime as runtime
    import raise_scorer.workflows as workflows

    assert "load_embedding_backend" in backends.__all__
    assert "SemanticRewardScorer" in core.__all__
    assert "assign_turn_rewards" in experiments.__all__
    assert "compute_score" in integrations.__all__
    assert "TraceEvent" in runtime.__all__
    assert "WorkflowGroundTruth" in workflows.__all__
    assert "WorkflowPotentialScorer" in workflows.__all__
