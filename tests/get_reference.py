import subprocess

from pathlib import Path
from urllib.request import urlopen

from tests.fixtures import DATA_ROOT, TEST_SKELETON
from tests.constants import LAMBDA, FRACTION

BRANCH = "master"
REPO_URL = f"https://raw.githubusercontent.com/catmaid/CATMAID/{BRANCH}"
CATMAID_LIB_URL = REPO_URL + "/django/applications/catmaid/static/libs/catmaid"

ARBOR_URL = CATMAID_LIB_URL + "/Arbor.js"
ARBOR_PARSER_URL = CATMAID_LIB_URL + "/arbor_parser.js"
SYNAPSE_CLUSTERING_URL = CATMAID_LIB_URL + "/synapse_clustering.js"


def get_file(url):
    response = urlopen(url)
    b = response.read()
    return b.decode()


template_path = DATA_ROOT / 'template.js'
skel_root = DATA_ROOT / str(TEST_SKELETON)
arbor_path = skel_root / "compact-arbor.json"
skeleton_path = skel_root / "compact-skeleton.json"
reference_path = skel_root / "reference"

template = template_path.read_text()


def main(force=False):
    if not force and reference_path.is_file():
        return

    arbor_ref_path = reference_path / 'arbor'
    parser_ref_path = reference_path / 'arbor_parser'
    clustering_ref_path = reference_path / 'synapse_clustering'

    s = template.format(
        arbor_js=get_file(ARBOR_URL),
        arbor_parser_js=get_file(ARBOR_PARSER_URL),
        synapse_clustering_js=get_file(SYNAPSE_CLUSTERING_URL),
        LAMBDA=LAMBDA,
        FRACTION=FRACTION,
        ref_path=reference_path,
        arbor_ref_path=arbor_ref_path,
        parser_ref_path=parser_ref_path,
        clustering_ref_path=clustering_ref_path,
        arbor_path=arbor_path,
        skeleton_path=skeleton_path,
    )
    for path in [reference_path, arbor_ref_path, parser_ref_path, clustering_ref_path]:
        path.mkdir(exist_ok=True)

    script_path = reference_path / "impl.js"

    with open(script_path, "w") as f:
        f.write(s)

    subprocess.run(['node', str(script_path)])


if __name__ == '__main__':
    main(True)
