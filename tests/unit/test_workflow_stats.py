from app.graph.workflow import merge_stats


def test_merge_stats_combines_both_sources():
    triplet_stats = {
        "total_frames": 2,
        "filtered_not_in_dict": 0,
        "polarity_counts": {"affirmed": 2, "denied": 0, "terminated": 0},
        "tense_counts": {
            "past_or_present_fact": 2,
            "future_or_planned": 0,
            "modal_possibility": 0,
        },
        "passed": 2,
    }
    annotation_stats = {
        "dropped_annotation_mismatch": 1,
        "dropped_evidence_grounding": 3,
    }

    merged = merge_stats(triplet_stats, annotation_stats)

    assert merged["total_frames"] == 2
    assert merged["passed"] == 2
    assert merged["dropped_annotation_mismatch"] == 1
    assert merged["dropped_evidence_grounding"] == 3


def test_merge_stats_defaults_missing_annotation_stats_to_zero():
    merged = merge_stats({"total_frames": 0, "passed": 0}, {})

    assert merged["dropped_annotation_mismatch"] == 0
    assert merged["dropped_evidence_grounding"] == 0


def test_merge_stats_does_not_mutate_inputs():
    triplet_stats = {"total_frames": 1}
    annotation_stats = {"dropped_annotation_mismatch": 2, "dropped_evidence_grounding": 0}

    merge_stats(triplet_stats, annotation_stats)

    assert triplet_stats == {"total_frames": 1}
