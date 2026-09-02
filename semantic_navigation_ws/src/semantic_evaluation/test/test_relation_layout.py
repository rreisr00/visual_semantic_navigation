from semantic_evaluation.core.relation_layout import (
    non_overlapping_label_rects,
    radial_node_positions,
    relation_lane_offsets,
)


def _overlaps(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return not (
        ax + aw <= bx
        or bx + bw <= ax
        or ay + ah <= by
        or by + bh <= ay
    )


def test_radial_layout_assigns_distinct_positions():
    positions = radial_node_positions(['chair', 'table', 'lamp', 'sofa'])

    assert set(positions) == {'chair', 'table', 'lamp', 'sofa'}
    assert len(set(positions.values())) == 4


def test_parallel_relations_receive_separate_lanes():
    offsets = relation_lane_offsets([
        ('chair', 'table'),
        ('chair', 'table'),
        ('table', 'chair'),
    ])

    assert offsets == [-34.0, 0.0, 34.0]


def test_relation_labels_do_not_overlap_nodes_or_each_other():
    occupied = [(75.0, 75.0, 50.0, 50.0)]
    rectangles = non_overlapping_label_rects(
        [(100.0, 100.0)] * 7,
        [(120.0, 28.0)] * 7,
        occupied=occupied,
    )

    assert len(rectangles) == 7
    assert all(not _overlaps(rectangle, occupied[0]) for rectangle in rectangles)
    assert all(
        not _overlaps(first, second)
        for index, first in enumerate(rectangles)
        for second in rectangles[index + 1:]
    )
