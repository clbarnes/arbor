import numpy as np
import pytest

from arbor import SynapseClustering, RelationType

from tests.test_arbor import simple_arbor


@pytest.fixture
def simple_syn_clus(simple_arbor):
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

