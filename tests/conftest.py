import os
from pathlib import Path

import pytest

from arbor.arbor import ArborClassic, ArborNX  # noqa
from arbor.arborparser import ArborParser

from tests.utils import load_json, node_version
from tests.constants import TEST_SKELETON, TEST_ROOT


HARNESS_ROOT = TEST_ROOT / "arbor-harness"


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
def reference_dir():
    data_root = TEST_ROOT / "arbor-harness" / "data"
    if not data_root.is_dir():
        pytest.xfail(
            "arbor-harness submodule has not been fetched, try\n\t"
            "git submodule update --init --recursive"
        )
    skel_root = data_root / str(TEST_SKELETON)
    if not skel_root.is_dir():
        pytest.xfail(
            f"arbor-harness does not contain arbor data for skeleton {TEST_SKELETON}"
        )
    return skel_root


@pytest.fixture
def compact_arbor(reference_dir):
    return load_json(reference_dir, "compact-arbor")


@pytest.fixture
def compact_skeleton(reference_dir):
    return load_json(reference_dir, "compact-skeleton")


def check_node_version():
    os.chdir(HARNESS_ROOT)
    try:
        node_ver = node_version()
    except RuntimeError:
        pytest.xfail(
            f"node.js is not installed in {HARNESS_ROOT}; install node.js v11+"
        )
    else:
        if node_ver < (11, 0, 0):
            pytest.xfail(
                f"node.js version installed in {HARNESS_ROOT} may be too old: install v11+"
            )


@pytest.fixture
def result_dir(reference_dir):
    result_dir = reference_dir / "results"
    if not result_dir.is_dir():
        check_node_version()
        pytest.xfail(
            "arbor-harness results have not been populated, try\n\t"
            f"cd {HARNESS_ROOT}\n\t"
            "npm install\n\t"
            "npm start"
        )
    return result_dir


class HarnessResult:
    def __init__(self, root: Path, category: os.PathLike, name: os.PathLike):
        self.category = category
        self.name = name
        cat_dir = root / category
        if not cat_dir.is_dir():
            pytest.xfail(
                f"Category {category} does not exist in directory {result_dir}"
            )
        self.base = str(cat_dir / name)
        self._result = None
        self._bench = None

    @property
    def result(self):
        if self._result is None:
            path = self.base + ".result.json"
            if not os.path.isfile(path):
                pytest.xfail(
                    f"Result JSON does not exist for test {self.category}/{self.name}"
                )

            self._result = load_json(path)
        return self._result

    @property
    def bench(self):
        if self._bench is None:
            path = self.base + ".bench.json"
            if not os.path.isfile(path):
                pytest.xfail(
                    f"Benchmark JSON does not exist for test {self.category}/{self.name}"
                )

            self._result = load_json(path)

        return self._bench


@pytest.fixture
def result_factory(result_dir):
    def fn(category: os.PathLike, name: os.PathLike) -> HarnessResult:
        return HarnessResult(result_dir, category, name)

    return fn


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
