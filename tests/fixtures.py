import json
import os

from pathlib import Path
import pytest

from arbor.arbor import ArborClassic, ArborNX
from arbor.arborparser import ArborParser

from tests.utils import convert_json

TEST_SKELETON = 3034133
DATA_ROOT = Path(__file__).absolute().parent / "fixture_data"


def fixture_path(path_item, *path_items):
    return os.path.join(DATA_ROOT, path_item, *path_items)


def load_json(path_item, *path_items, parse_keys=True, parse_strings=True):
    """Get JSON data from fixture file"""
    with open(fixture_path(path_item, *path_items)) as f:
        obj = json.load(f)

    if parse_keys:
        return convert_json(obj, parse_strings=parse_strings)
    else:
        return obj


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


@pytest.fixture
def real_arbor(real_arbor_parser):
    return real_arbor_parser.arbor
