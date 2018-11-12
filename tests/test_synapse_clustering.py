from collections import defaultdict

import numpy as np
import pytest

from arbor import SynapseClustering
from arbor.arborparser import ArborParser

from tests.constants import LAMBDA, FRACTION
from tests.utils import to_jso_like

FIXTURE_DIR = "synapse_clustering"


@pytest.fixture
def simple_syn_clus(simple_arbor):
    locations = {idx: np.array([idx, 0, 0]) for idx in simple_arbor.edges}
    locations[simple_arbor.root] = np.array([simple_arbor.root, 0, 0])

    synapses = {5: 10, 7: 11}
    lambda_ = 1

    return SynapseClustering(simple_arbor, locations, synapses, lambda_)


def combine_dicts(*dicts):
    out = defaultdict(lambda: 0)
    for d in dicts:
        for key, value in d.items():
            out[key] += value
    return dict(out)


@pytest.fixture
def real_syn_clus(real_arbor_parser, pytestconfig):
    if pytestconfig.getoption("--skipslow"):
        pytest.skip("synapse clustering tests are slow, and --skipslow was used")

    synapses = combine_dicts(real_arbor_parser.inputs, real_arbor_parser.outputs)

    return SynapseClustering(
        real_arbor_parser.arbor, real_arbor_parser.positions, synapses, LAMBDA
    )


def test_instantiate_real(real_syn_clus: SynapseClustering, result_factory):
    """Checks that distance_map works, among other things"""
    expected = result_factory(FIXTURE_DIR, "synapse_clustering").result
    assert to_jso_like(real_syn_clus) == expected


def test_density_hill_map_real(real_syn_clus: SynapseClustering, result_factory):
    """depends on test_instantiate_real"""
    dhm = real_syn_clus.density_hill_map()
    expected = result_factory(FIXTURE_DIR, "density_hill_map").result
    assert dhm == expected


def test_cluster_sizes_real(real_syn_clus: SynapseClustering, result_factory):
    """depends on test_density_hill_map_real"""
    dhm = real_syn_clus.density_hill_map()
    csize = real_syn_clus.cluster_sizes(dhm)

    expected = result_factory(FIXTURE_DIR, "cluster_sizes").result
    assert csize == expected


def test_clusters_real(real_syn_clus: SynapseClustering, result_factory):
    """depends on test_density_hill_map_real"""
    dhm = real_syn_clus.density_hill_map()
    clusters = real_syn_clus.clusters(dhm)
    expected = result_factory(FIXTURE_DIR, "clusters").result
    expected_sets = {k: set(v) for k, v in expected.items()}
    assert expected_sets == clusters


def test_segregation_index_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering, result_factory
):
    """depends on test_clusters_real"""
    dhm = real_syn_clus.density_hill_map()
    clusters = real_syn_clus.clusters(dhm)
    seg_ind = real_syn_clus.segregation_index(
        clusters, real_arbor_parser.outputs, real_arbor_parser.inputs
    )
    expected = result_factory(FIXTURE_DIR, "segregation_index")
    assert seg_ind == pytest.approx(expected)


def test_find_arbor_regions_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering, result_factory
):
    """depends on test_arbor.test_flow_centralities"""
    arbor = real_arbor_parser.arbor
    fcs = arbor.flow_centrality(real_arbor_parser.outputs, real_arbor_parser.inputs)
    arbor_regions = real_syn_clus.find_arbor_regions(fcs, FRACTION, arbor)
    expected = result_factory(FIXTURE_DIR, "find_arbor_regions").result
    assert to_jso_like(arbor_regions) == expected


def test_find_axon_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering, result_factory
):
    axon = real_syn_clus.find_axon(
        real_arbor_parser, FRACTION, real_arbor_parser.positions
    )
    expected = result_factory(FIXTURE_DIR, "find_axon").result

    assert to_jso_like(axon) == expected


def test_find_axon_cut_real(
    real_arbor_parser: ArborParser, real_syn_clus: SynapseClustering, result_factory
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

    expected = result_factory(FIXTURE_DIR, "find_axon_cut").result
    assert to_jso_like(axon_cut) == expected
