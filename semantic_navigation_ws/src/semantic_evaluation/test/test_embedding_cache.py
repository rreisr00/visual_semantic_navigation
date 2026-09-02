"""Unit tests for the embedding cache and reproducibility helpers."""
import numpy as np

from semantic_evaluation.core.embedding_cache import (
    EmbeddingCache,
    collect_manifest,
    save_manifest,
    set_seeds,
)


def test_array_round_trip(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    key = cache.text_key("model-a", "hola")
    assert cache.get_array(key) is None
    cache.put_array(key, np.array([1.0, 2.0]))
    np.testing.assert_allclose(cache.get_array(key), [1.0, 2.0])


def test_keys_depend_on_model_and_content(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    assert cache.text_key("m1", "hola") != cache.text_key("m2", "hola")
    assert cache.text_key("m1", "hola") != cache.text_key("m1", "adios")

    image = tmp_path / "img.bin"
    image.write_bytes(b"aaa")
    key_before = cache.image_key("m1", str(image))
    image.write_bytes(b"bbb")   # content change → new key
    assert cache.image_key("m1", str(image)) != key_before


def test_json_and_arrays_storage(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    image = tmp_path / "img.bin"
    image.write_bytes(b"x")
    det_key = cache.detections_key("yolo.pt", str(image), 0.4)
    cache.put_json(det_key, [{"label": "cup", "confidence": 0.9}])
    assert cache.get_json(det_key)[0]["label"] == "cup"

    crops_key = cache.crops_key("m1", str(image), det_key)
    cache.put_arrays(crops_key, [np.zeros(3), np.ones(3)])
    arrays = cache.get_arrays(crops_key)
    assert len(arrays) == 2
    np.testing.assert_allclose(arrays[1], [1.0, 1.0, 1.0])


def test_clear(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    cache.put_array(cache.text_key("m", "t"), np.array([1.0]))
    assert cache.clear() == 1
    assert cache.get_array(cache.text_key("m", "t")) is None


def test_set_seeds_makes_numpy_deterministic():
    set_seeds(123)
    first = np.random.rand(3)
    set_seeds(123)
    np.testing.assert_allclose(first, np.random.rand(3))


def test_manifest_contents(tmp_path):
    manifest = collect_manifest({"seed": 42}, device="cpu", extra={"scene": "s1"})
    assert manifest["config"]["seed"] == 42
    assert manifest["scene"] == "s1"
    assert "numpy" in manifest["versions"]
    path = save_manifest(str(tmp_path / "run" / "manifest.json"), manifest)
    assert (tmp_path / "run" / "manifest.json").exists() and path
