from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from arbor import SynapseClustering, RelationType
from arbor.arborparser import ArborParser

from tests.constants import LAMBDA, FRACTION
from tests.utils import assert_same


FIXTURE_DIR = "synapse_clustering"


@pytest.fixture
def simple_syn_clus(simple_arbor):
    locations = {idx: np.array([idx, 0, 0]) for idx in simple_arbor.edges}
    locations[simple_arbor.root] = np.array([simple_arbor.root, 0, 0])

    synapses = {5: (RelationType.PRESYNAPTIC, 10), 7: (RelationType.POSTSYNAPTIC, 11)}
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
        8: [7.0, 1.0],
    }


def test_density_hill_map(simple_syn_clus):
    density_hill_map = simple_syn_clus.density_hill_map()
    assert density_hill_map == {1: 0, 2: 0, 4: 1, 6: 2, 3: 1, 7: 2, 8: 2, 5: 1}


def combine_dicts(*dicts):
    out = defaultdict(lambda: 0)
    for d in dicts:
        for key, value in d.items():
            out[key] += value
    return dict(out)


@pytest.fixture(scope="session")
def _real_syn_clus(real_arbor_parser, pytestconfig):
    if pytestconfig.getoption("--skipslow"):
        pytest.skip("synapse clustering tests are slow, and --skipslow was used")

    synapses = combine_dicts(real_arbor_parser.inputs, real_arbor_parser.outputs)

    return SynapseClustering(
        real_arbor_parser.arbor, real_arbor_parser.positions, synapses, LAMBDA
    )


@pytest.fixture
def real_syn_clus(_real_syn_clus):
    """Hack to get around the (temporarily) slow instantiation of SynapseClustering"""
    return deepcopy(_real_syn_clus)


def test_instantiate_real(real_syn_clus: SynapseClustering):
    """Checks that distance_map works, among other things"""
    assert_same(real_syn_clus, FIXTURE_DIR, "synapseclustering")


def test_density_hill_map_real(real_syn_clus: SynapseClustering):
    """depends on test_instantiate_real"""
    dhm = real_syn_clus.density_hill_map()
    assert_same(dhm, FIXTURE_DIR, "density_hill_map")


def test_cluster_sizes_real(real_syn_clus: SynapseClustering):
    """depends on test_density_hill_map_real"""
    dhm = real_syn_clus.density_hill_map()
    csize = real_syn_clus.cluster_sizes(dhm)
    assert_same(csize, FIXTURE_DIR, "cluster_sizes")


def test_clusters_real(real_syn_clus: SynapseClustering):
    """depends on test_density_hill_map_real"""
    dhm = real_syn_clus.density_hill_map()
    clusters = real_syn_clus.clusters(dhm)
    assert_same(clusters, FIXTURE_DIR, "clusters")


def test_cluster_maps_real(real_syn_clus: SynapseClustering):
    """depends on test_density_hill_map_real"""
    dhm = real_syn_clus.density_hill_map()
    cluster_maps = real_syn_clus.cluster_maps(dhm)
    assert_same(cluster_maps, FIXTURE_DIR, "cluster_maps")


def test_segregation_index_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering
):
    """depends on test_clusters_real"""
    dhm = real_syn_clus.density_hill_map()
    clusters = real_syn_clus.clusters(dhm)
    seg_ind = real_syn_clus.segregation_index(
        clusters, real_arbor_parser.outputs, real_arbor_parser.inputs
    )
    assert_same(seg_ind, FIXTURE_DIR, "segregation_index")


def test_find_arbor_regions_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering
):
    """depends on test_arbor.test_flow_centralities"""
    arbor = real_arbor_parser.arbor
    fcs = arbor.flow_centrality(real_arbor_parser.outputs, real_arbor_parser.inputs)
    arbor_regions = real_syn_clus.find_arbor_regions(fcs, FRACTION, arbor)
    assert_same(arbor_regions, FIXTURE_DIR, "find_arbor_regions")


def test_find_axon_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering
):
    axon = real_syn_clus.find_axon(
        real_arbor_parser, FRACTION, real_arbor_parser.positions
    )
    assert_same(axon, FIXTURE_DIR, "find_axon")


def test_find_axon_cut_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering
):
    """depends on test_find_arbor_regions_real"""
    arbor = real_arbor_parser.arbor
    fcs = arbor.flow_centrality(real_arbor_parser.outputs, real_arbor_parser.inputs)
    arbor_regions = real_syn_clus.find_arbor_regions(fcs, FRACTION, arbor)

    axon_cut = real_syn_clus.find_axon_cut(
        real_arbor_parser.outputs,
        above=arbor_regions.above,
        positions=real_arbor_parser.positions,
        arbor=arbor,
    )

    assert_same(axon_cut, FIXTURE_DIR, "find_axon_cut")
