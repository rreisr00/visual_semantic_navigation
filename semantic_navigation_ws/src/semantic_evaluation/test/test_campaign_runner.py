from semantic_evaluation.campaign_runner_widget import (
    _mean_boolean,
    _mean_number,
    _percent,
)


def test_campaign_result_summary_ignores_undefined_navigation():
    rows = [
        {'semantic_success': 'true', 'navigation_success': ''},
        {'semantic_success': 'false', 'navigation_success': 'true'},
    ]

    assert _mean_boolean(rows, 'semantic_success') == 0.5
    assert _mean_boolean(rows, 'navigation_success') == 1.0
    assert _percent(0.5) == '50.0%'
    assert _percent(None) == 'n/d'


def test_campaign_result_summary_reads_numeric_values():
    rows = [
        {'reciprocal_rank': '1.0'},
        {'reciprocal_rank': '0.5'},
        {'reciprocal_rank': ''},
    ]

    assert _mean_number(rows, 'reciprocal_rank') == 0.75
