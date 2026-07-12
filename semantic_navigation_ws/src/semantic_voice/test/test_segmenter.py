"""Unit tests for the RMS speech segmenter (numpy only, no ROS)."""

import numpy as np

from semantic_voice.core.segmenter import SpeechSegmenter

SR = 16000
CHUNK = int(0.03 * SR)  # 30 ms


def silence(n_chunks: int):
    return [np.zeros(CHUNK, dtype=np.float32) for _ in range(n_chunks)]


def speech(n_chunks: int, amplitude: float = 0.1):
    rng = np.random.default_rng(42)
    return [
        (amplitude * rng.standard_normal(CHUNK)).astype(np.float32)
        for _ in range(n_chunks)
    ]


def feed(seg, chunks):
    out = []
    for c in chunks:
        u = seg.push(c)
        if u is not None:
            out.append(u)
    return out


def test_silence_only_yields_nothing():
    seg = SpeechSegmenter(sample_rate=SR)
    assert feed(seg, silence(100)) == []


def test_burst_plus_hangover_yields_one_utterance():
    seg = SpeechSegmenter(sample_rate=SR, hangover_s=0.3)
    # 1 s of speech then enough silence to close the utterance
    utterances = feed(seg, speech(34) + silence(20))
    assert len(utterances) == 1
    # utterance holds the speech plus the hangover silence
    assert utterances[0].size >= 34 * CHUNK


def test_max_utterance_forces_cut():
    seg = SpeechSegmenter(sample_rate=SR, max_utterance_s=1.0, hangover_s=5.0)
    # 2 s of continuous speech must be cut at ~1 s despite no silence
    utterances = feed(seg, speech(67))
    assert len(utterances) >= 1
    assert utterances[0].size <= int(1.0 * SR) + CHUNK


def test_sub_minimum_burst_dropped():
    seg = SpeechSegmenter(sample_rate=SR, hangover_s=0.3, min_utterance_s=0.5)
    # only ~90 ms of speech -> dropped
    assert feed(seg, speech(3) + silence(20)) == []


def test_two_separate_utterances():
    seg = SpeechSegmenter(sample_rate=SR, hangover_s=0.3)
    utterances = feed(seg, speech(20) + silence(15) + speech(20) + silence(15))
    assert len(utterances) == 2
