import pytest

from arbor.arbor import ArborClassic, ArborNX  # noqa
from arbor.arborparser import ArborParser

from tests.utils import load_json
from tests.constants import TEST_SKELETON


def pytest_addoption(parser):
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skipslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="--skipslow was used")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def compact_arbor():
    return load_json(str(TEST_SKELETON), "compact-arbor.json")


@pytest.fixture
def compact_skeleton():
    return load_json(str(TEST_SKELETON), "compact-skeleton.json")


# @pytest.fixture(params=[ArborNX, ArborClassic])  # ArborClassic not fully implemented
@pytest.fixture(params=[ArborNX])
def arbor_class(request):
    return request.param


@pytest.fixture
def simple_arbor(arbor_class):
    arbor = arbor_class()
    arbor.add_path([1, 2, 3, 4, 5])
    arbor.add_path([3, 6, 7, 8])
    return arbor


@pytest.fixture
def real_arbor_parser(arbor_class, compact_arbor):
    parser = ArborParser()
    parser.arbor_class = arbor_class
    parser.init("compact-arbor", compact_arbor)
    return parser


@pytest.fixture
def real_arbor(real_arbor_parser):
    return real_arbor_parser.arbor
