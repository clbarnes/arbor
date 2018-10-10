from arbor.arborparser import ArborParser
from tests.fixtures import compact_arbor, compact_skeleton


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
