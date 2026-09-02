import pytest

from semantic_navigation_core.configuration import (
    configuration_hash,
    load_frozen_config,
    verify_configuration_hash,
)


def test_hash_is_independent_of_mapping_order():
    assert configuration_hash({"b": 2, "a": 1}) == configuration_hash({"a": 1, "b": 2})


def test_verify_hash_rejects_drift():
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        verify_configuration_hash({"value": 2}, configuration_hash({"value": 1}))


def test_frozen_config_requires_all_contract_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("siglip_checkpoint: local\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_frozen_config(path)

