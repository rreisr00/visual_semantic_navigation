import math

import pytest

from semantic_navigation_core.path_metrics import path_length_2d, spl


def test_path_length_2d_accumulates_segments():
    assert path_length_2d([(0.0, 0.0), (3.0, 4.0), (6.0, 4.0)]) == 8.0


def test_spl_contract():
    assert spl(True, 4.0, 5.0) == pytest.approx(0.8)
    assert spl(True, 0.0, 0.0) == 1.0
    assert spl(False, 4.0, 4.0) == 0.0
    assert math.isnan(spl(True, math.nan, 4.0))
