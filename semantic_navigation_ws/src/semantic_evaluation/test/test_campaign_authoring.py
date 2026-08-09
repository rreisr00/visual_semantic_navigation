from semantic_evaluation.core.campaign_authoring import (
    campaign_issues,
    confirm_object_ground_truth,
    identifier_from_query,
    unique_case_id,
    upsert_query_case,
)

from test_campaign_designer import _workspace


def test_identifier_and_unique_case_id(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.queries['cases'].append({'case_id': 'sofa_del_salon'})

    assert identifier_from_query('¡Sofá del salón!') == 'sofa_del_salon'
    assert unique_case_id(workspace, 'sofa del salon') == 'sofa_del_salon_2'


def test_upsert_query_case_synchronizes_negative_ground_truth(tmp_path):
    workspace = _workspace(tmp_path)
    case = upsert_query_case(
        workspace,
        case_id='negative_pool',
        query_id='negative_pool_es',
        query_text='la piscina',
        query_type='negative',
        language='es',
        start_pose_id='origin',
        exact_valid_nodes=['node_01'],
        nearby_valid_nodes=[],
        is_negative=True,
        target_visible=True,
        timeout_s=30.0,
    )

    assert case['exact_valid_nodes'] == []
    assert case['target_visible'] is False
    assert workspace.ground_truth['negative_queries'] == ['negative_pool_es']


def test_campaign_issues_require_ground_truth_for_positive_query(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.queries['cases'].append({
        'case_id': 'find_sofa',
        'query_id': 'find_sofa_es',
        'query_text': 'el sofa',
        'is_negative': False,
        'target_visible': True,
        'exact_valid_nodes': [],
        'nearby_valid_nodes': [],
        'timeout_s': 30.0,
    })

    codes = {issue.code for issue in campaign_issues(workspace)}

    assert 'positive_without_exact' in codes
    assert 'node_without_room' in codes


def test_confirm_object_ground_truth_requires_explicit_node_validity(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.nodes[0].room_id = 'salon'

    changed = confirm_object_ground_truth(
        workspace,
        {'node_01': ['sofa', 'sofa']},
        exact_nodes={'node_01'},
        nearby_nodes=set(),
    )

    assert changed == 1
    assert workspace.ground_truth['objects'] == [{
        'category': 'sofa',
        'room_id': 'salon',
        'exact_valid_nodes': ['node_01'],
        'nearby_valid_nodes': [],
    }]
