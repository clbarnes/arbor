import itertools
from collections import Counter

import numpy as np
import pytest

from arbor import ArborClassic, ArborNX
from arbor.arbor import FlowCentrality
from tests.fixtures import arbor_class, simple_arbor


def test_instantiate(arbor_class):
    arbor = arbor_class()


def test_add_edge_pairs(arbor_class):
    arbor = arbor_class()
    arbor.add_edge_pairs((2, 1), (3, 2), (4, 3))
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_add_edges(arbor_class):
    arbor = arbor_class()
    arbor.add_edges([2, 1, 3, 2, 4, 3])
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_add_path(arbor_class):
    arbor = arbor_class()
    arbor.add_path([1, 2, 3, 4])
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_reroot(simple_arbor):
    simple_arbor.reroot(8)
    assert simple_arbor.root == 8
    assert simple_arbor.edges == {7: 8, 6: 7, 3: 6, 5: 4, 4: 3, 2: 3, 1: 2}


def test_all_successors(simple_arbor):
    assert simple_arbor.all_successors() == {
        1: [2],
        2: [3],
        3: [4, 6],
        4: [5],
        5: [],
        6: [7],
        7: [8],
        8: [],
    }


def assert_same_members(item1, item2):
    assert Counter(iter(item1)) == Counter(iter(item2))


def test_children_list(simple_arbor):
    assert_same_members(simple_arbor.children_list(), simple_arbor.edges)


def test_find_branch_end_nodes(simple_arbor):
    branch_end_nodes = simple_arbor.find_branch_and_end_nodes()
    assert branch_end_nodes.branches == {3: 2}
    assert_same_members(branch_end_nodes.ends, [8, 5])


def test_nodes_distance_to_simple(simple_arbor):
    nodes_distance_to = simple_arbor.nodes_distance_to()
    assert nodes_distance_to.distances == {
        8: 5,
        7: 4,
        6: 3,
        3: 2,
        2: 1,
        1: 0,
        5: 4,
        4: 3,
    }


def assert_distances_close(real_dict, expected_dict):
    assert real_dict.keys() == expected_dict.keys()
    for key, real_value in real_dict.items():
        assert real_value == pytest.approx(expected_dict[key])


def test_nodes_distance_to_euclidean(simple_arbor):
    nodes_distance_to = simple_arbor.nodes_distance_to(
        location_dict={idx: np.array([idx, idx, idx]) for idx in range(1, 9)}
    )
    unit_len = np.linalg.norm([1, 1, 1])
    assert_distances_close(
        nodes_distance_to.distances, {idx: (idx - 1) * unit_len for idx in range(1, 9)}
    )


def starts_with(sequence, subsequence):
    return sequence[: len(subsequence)] == subsequence


def test_partition(simple_arbor):
    partitions = simple_arbor.partition()
    # 2 partitions; , are of length
    assert len(partitions) == 2
    # repeat the branch point and nothing else
    assert dict(Counter(itertools.chain(*partitions))) == {
        1: 1,
        2: 1,
        3: 2,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
    }
    # which start with with 5,4,3 and 8,7,6,3
    assert all(
        starts_with(partition, [5, 4, 3]) or starts_with(partition, [8, 7, 6, 3])
        for partition in partitions
    )
    # one of which ends with 3, 2, 1
    assert any(starts_with(partition[::-1], [1, 2, 3]) for partition in partitions)


def test_flow_centrality(simple_arbor):
    sources = {4: 2}
    targets = {7: 3, 4: 1}
    fc = simple_arbor.flow_centrality(targets, sources)
    assert fc == {  # tested with JS implementation
        5: FlowCentrality(centrifugal=0, centripetal=0),
        4: FlowCentrality(centrifugal=6, centripetal=0),
        3: FlowCentrality(centrifugal=0, centripetal=0),
        8: FlowCentrality(centrifugal=0, centripetal=0),
        7: FlowCentrality(centrifugal=0, centripetal=6),
        6: FlowCentrality(centrifugal=0, centripetal=6),
        2: FlowCentrality(centrifugal=0, centripetal=0),
        1: FlowCentrality(centrifugal=0, centripetal=0),
    }
