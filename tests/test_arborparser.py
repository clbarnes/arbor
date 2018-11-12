from arbor.arborparser import ArborParser

FIXTURE_DIR = "arbor_parser"


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


def test_real(real_arbor_parser, result_factory):
    expected = result_factory(FIXTURE_DIR, "from_compact-arbor").result
    assert real_arbor_parser.inputs == expected["inputs"]
    assert real_arbor_parser.outputs == expected["outputs"]


def test_create_synapse_map(real_arbor_parser, result_factory):
    real = real_arbor_parser.create_synapse_map()
    expected = result_factory(FIXTURE_DIR, "create_synapse_map").result
    assert real == expected
