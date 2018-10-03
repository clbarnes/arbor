from collections import Counter

import numpy as np
import pytest

from arbor import Arbor


@pytest.fixture
def simple_arbor():
    arbor = Arbor()
    arbor.add_path([1, 2, 3, 4, 5])
    arbor.add_path([3, 6, 7, 8])
    return arbor


def test_instantiate():
    arbor = Arbor()


def test_add_edge_pairs():
    arbor = Arbor()
    arbor.add_edge_pairs((2, 1), (3, 2), (4, 3))
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_add_edges():
    arbor = Arbor()
    arbor.add_edges([2, 1, 3, 2, 4, 3])
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_add_path():
    arbor = Arbor()
    arbor.add_path([1, 2, 3, 4])
    assert arbor.edges == {2: 1, 3: 2, 4: 3}
    assert arbor.root == 1


def test_reroot(simple_arbor):
    simple_arbor.reroot(8)
    assert simple_arbor.root == 8
    assert simple_arbor.edges == {
        7: 8,
        6: 7,
        3: 6,
        5: 4,
        4: 3,
        2: 3,
        1: 2
    }


def test_all_successors(simple_arbor):
    assert simple_arbor.all_successors() == {
        1: [2], 2: [3], 3: [4, 6], 4: [5], 5: [], 6: [7], 7: [8], 8: []
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
    nodes_distance_to = simple_arbor.nodes_distance_to(location_dict={
        idx: np.array([idx, idx, idx]) for idx in range(1, 9)
    })
    unit_len = np.linalg.norm([1, 1, 1])
    assert_distances_close(nodes_distance_to.distances, {idx: (idx-1) * unit_len for idx in range(1, 9)})


def test_partition(simple_arbor):
    partition = simple_arbor.partition()
    assert partition == [
        [5, 4, 3],
        [8, 7, 6, 3, 2, 1],
    ]
