from arbor.arborparser import ArborParser

from tests.fixtures import (
    compact_arbor, compact_skeleton, get_expected,
    arbor_class, real_arbor_parser, compact_arbor
)


FIXTURE_DIR = 'arbor_parser'


def test_can_instantiate():
    ArborParser()


def test_connectors(compact_skeleton):
    ap = ArborParser()
    ap.connectors(compact_skeleton[1])


def test_synapses(compact_arbor):
    ap = ArborParser()
    ap.synapses(compact_arbor[1])


def test_init_skeleton(compact_skeleton):
    ap = ArborParser()
    ap.init("compact-skeleton", compact_skeleton)


def test_init_arbor(compact_arbor):
    ap = ArborParser()
    ap.init("compact-arbor", compact_arbor)


def test_real(real_arbor_parser):
    expected = get_expected(FIXTURE_DIR, 'arborparser')
    assert real_arbor_parser.inputs == expected['inputs']
    assert real_arbor_parser.outputs == expected['outputs']


def test_create_synapse_map(real_arbor_parser):
    real = real_arbor_parser.create_synapse_map()
    expected = get_expected(FIXTURE_DIR, 'create_synapse_map')
    assert real == expected
