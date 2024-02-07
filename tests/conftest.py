# content of conftest.py
import os
import gzip
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="mini",
        help="How many SMILES to test on",
        choices=("mini", "full"),
    )

@pytest.fixture
def TEST_SMILES(request):
    if request.config.getoption("--input") == 'mini':
        with open(os.path.join(os.path.dirname(__file__), "test_smiles.smi")) as f:
            TEST_SMILES = [smi.strip() for smi in f.readlines()]
    else:
        with gzip.open(os.path.join(os.path.dirname(__file__), "guacamol_train.smi.gz")) as gf:
            TEST_SMILES = [smi.decode().strip("\n") for smi in gf.readlines()]
    return TEST_SMILES

@pytest.fixture
def TEST_SCAFFOLDS():
    with open(os.path.join(os.path.dirname(__file__), "scaffolds.smi")) as f:
        return [smi.strip() for smi in f.readlines()]