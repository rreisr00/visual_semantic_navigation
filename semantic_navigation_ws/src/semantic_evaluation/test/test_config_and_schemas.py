import hashlib

import pytest
import yaml

from semantic_evaluation.core.config_validation import (
    validate_offline_isolation,
    validate_separate_roots,
)
from semantic_evaluation.core.experimental_schemas import (
    CampaignSpec,
    QuerySpec,
    SchemaValidationError,
)
from semantic_evaluation.core.reproducibility import (
    freeze_retrieval_config,
    verify_frozen_config,
)


def test_offline_rejects_ros_and_simulation_paths():
    with pytest.raises(SchemaValidationError, match=".ros"):
        validate_offline_isolation({"paths": {"cache": "~/.ros/cache"}}, "offline.yaml")
    with pytest.raises(SchemaValidationError, match="aws_small_house"):
        validate_offline_isolation({"dataset": "aws_small_house"}, "offline.yaml")


def test_offline_and_simulation_roots_must_be_distinct():
    with pytest.raises(SchemaValidationError):
        validate_separate_roots(
            {"paths": {"results_root": "experiments/results"}},
            {"paths": {"results_root": "experiments/results"}},
        )


def test_query_schema_allows_dataset_without_scene():
    query = QuerySpec.from_mapping({
        "query_id": "q1", "text": "kitchen", "language": "en",
        "query_type": "room", "dataset_id": "rooms",
    })
    assert query.dataset_id == "rooms" and query.scene_id is None


def test_campaign_schema_rejects_opaque_method():
    data = {
        "campaign_id": "c", "scene_id": "s", "run_id": "r", "seed": 1,
        "method": "M1", "start_pose_id": "p", "query_suite_id": "q",
        "frozen_config_hash": "abc", "git_commit": "def",
        "timestamp": "2026-01-01T00:00:00Z", "status": "complete",
    }
    with pytest.raises(SchemaValidationError, match="method"):
        CampaignSpec.from_mapping(data)


def test_frozen_config_hash_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "frozen_retrieval_config.yaml"
    _, sidecar, digest = freeze_retrieval_config(
        path,
        {"models": {"siglip": "checkpoint"}, "weights": {"global": 1.0}},
        {"validated_offline": True, "n_validation_queries": 3,
         "simulation_sources": []},
    )
    loaded, verified = verify_frozen_config(path, digest)
    assert loaded["status"] == "frozen" and verified == digest
    path.write_text(path.read_text() + "tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_frozen_config(path)


def test_freeze_refuses_unvalidated_data(tmp_path):
    with pytest.raises(ValueError, match="validated_offline"):
        freeze_retrieval_config(tmp_path / "f.yaml", {}, {})

