import numpy as np
import pytest

from arbor import SynapseClustering, RelationType, ArborClassic

from tests.test_arbor import simple_arbor, arbor_class


@pytest.fixture
def simple_syn_clus(simple_arbor):
    if isinstance(simple_arbor, ArborClassic):
        pytest.xfail("Clustering not implemented for ArborClassic")

    locations = {idx: np.array([idx, 0, 0]) for idx in simple_arbor.edges}
    locations[simple_arbor.root] = np.array([simple_arbor.root, 0, 0])

    synapses = {
        5: (RelationType.PRESYNAPTIC, 10),
        7: (RelationType.POSTSYNAPTIC, 11)
    }
    lambda_ = 1

    return SynapseClustering(simple_arbor, locations, synapses, lambda_)


def test_instantiate(simple_syn_clus):
    # arbor partitioning, arbor nodes_distance_to, synapse_clustering distance_map
    distances = simple_syn_clus.distances
    assert distances == {
        5: [0, 6.0],
        4: [1.0, 5.0],
        3: [2.0, 4.0],
        2: [3.0, 5.0],
        1: [4.0, 6.0],
        6: [5.0, 1.0],
        7: [6.0, 0],
        8: [7.0, 1.0]
    }


def test_density_hill_map(simple_syn_clus):
    density_hill_map = simple_syn_clus.density_hill_map()
    assert density_hill_map == {1: 0, 2: 0, 4: 1, 6: 2, 3: 1, 7: 2, 8: 2, 5: 1}
