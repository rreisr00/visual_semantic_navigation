import json
import sqlite3

import pytest
import yaml

from semantic_evaluation.core.campaign_designer import (
    OccupancyMap,
    load_workspace,
    save_workspace,
    validate_workspace,
)


def _graph_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE nodes(name TEXT PRIMARY KEY, type TEXT, properties TEXT);
        CREATE TABLE edges(
            type TEXT, source_node TEXT, target_node TEXT, properties TEXT
        );
        """
    )
    properties = {
        'pose_x': 1.0,
        'pose_y': 2.0,
        'pose_z': 0.0,
        'scene_id': 'test_scene',
    }
    connection.execute(
        'INSERT INTO nodes VALUES (?, ?, ?)',
        ('node_01', 'waypoint', json.dumps(properties)),
    )
    connection.commit()
    connection.close()


def _workspace(tmp_path):
    graph = tmp_path / 'graph.db'
    _graph_database(graph)
    image = tmp_path / 'map.pgm'
    image.write_text('P2\n1 1\n255\n254\n', encoding='ascii')
    map_file = tmp_path / 'map.yaml'
    map_file.write_text(
        yaml.safe_dump({
            'image': image.name,
            'resolution': 0.5,
            'origin': [-1.0, -2.0, 0.0],
        }),
        encoding='utf-8',
    )
    return load_workspace(
        scene_id='test_scene',
        graph_database=str(graph),
        map_file=str(map_file),
        queries_file=str(tmp_path / 'queries.yaml'),
        ground_truth_file=str(tmp_path / 'ground_truth.yaml'),
    )


def test_world_to_pixel_accounts_for_origin_scale_and_y_flip():
    metadata = OccupancyMap('map.yaml', 'map.pgm', 0.5, -1.0, -2.0)

    assert metadata.world_to_pixel(1.0, 2.0, 20) == (4.0, 12.0)


def test_pixel_to_world_is_inverse_with_rotated_map():
    metadata = OccupancyMap(
        'map.yaml', 'map.pgm', 0.2, -3.0, 1.5, origin_yaw=0.35
    )
    pixel = metadata.world_to_pixel(2.4, -0.7, 240)

    world = metadata.pixel_to_world(*pixel, image_height=240)

    assert world == pytest.approx((2.4, -0.7))


def test_occupancy_pixel_classification_obeys_negate_and_free_threshold():
    normal = OccupancyMap('map.yaml', 'map.pgm', 0.2, 0.0, 0.0)
    negated = OccupancyMap(
        'map.yaml', 'map.pgm', 0.2, 0.0, 0.0, negate=True
    )

    assert normal.pixel_is_free(255, 255, 255)
    assert not normal.pixel_is_free(0, 0, 0)
    assert negated.pixel_is_free(0, 0, 0)
    assert not negated.pixel_is_free(255, 255, 255)


def test_workspace_creates_compatible_query_and_ground_truth_files(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.queries['cases'].append({
        'case_id': 'find_node',
        'query_id': 'find_node_es',
        'query_text': 've al nodo',
        'query_type': 'object',
        'language': 'es',
        'start_pose_id': 'origin',
        'exact_valid_nodes': ['node_01'],
        'nearby_valid_nodes': [],
        'is_negative': False,
        'timeout_s': 30.0,
    })
    workspace.ground_truth['rooms'].append({
        'room_id': 'room_a',
        'valid_nodes': ['node_01'],
    })

    save_workspace(workspace)

    queries = yaml.safe_load((tmp_path / 'queries.yaml').read_text())
    truth = yaml.safe_load((tmp_path / 'ground_truth.yaml').read_text())
    assert queries['scene_id'] == 'test_scene'
    assert queries['cases'][0]['exact_valid_nodes'] == ['node_01']
    assert truth['rooms'][0] == {
        'room_id': 'room_a', 'valid_nodes': ['node_01']
    }


def test_workspace_rejects_references_to_unknown_nodes(tmp_path):
    workspace = _workspace(tmp_path)
    workspace.ground_truth['relations'].append({
        'subject': 'cup',
        'predicate': 'ON',
        'object': 'table',
        'valid_nodes': ['missing_node'],
    })

    with pytest.raises(ValueError, match='unknown nodes'):
        validate_workspace(workspace)
