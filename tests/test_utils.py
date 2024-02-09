# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import os
import sys
import gzip
import pytest
from promptsmiles import utils

# --------------------- Load test files ---------------------
# PyTest was doing my head in so I've done it this way
TEST_SMILES = []
TEST_SCAFFOLD = []
# Load supplied files
for arg in sys.argv:
    if arg.endswith(".smi"):
        with open(arg) as f:
            TEST_SMILES.extend([smi.strip() for smi in f.readlines()])
    if arg.endswith(".smi.gz"):
        with gzip.open(arg) as gf:
            TEST_SMILES.extend([smi.decode().strip() for smi in gf.readlines()])
    if arg.endswith(".scaff"):
        with open(arg) as f:
            TEST_SCAFFOLDS.extend([smi.strip() for smi in f.readlines()])
# Load defaults
if not TEST_SMILES:
    with open(os.path.join(os.path.dirname(__file__), "test_smiles.smi")) as f:
        TEST_SMILES.extend([smi.strip() for smi in f.readlines()])
if not TEST_SCAFFOLD:
    with open(os.path.join(os.path.dirname(__file__), "test_scaffolds.scaff")) as f:
        TEST_SCAFFOLD.extend([smi.strip() for smi in f.readlines()])

# TODO test fragments
# TODO test randomize with random root?

# --------------------- Broad tests ------------------------
@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_reverse(smiles):
    """Make an assumption that if we reverse a SMILES string twice, we get the same thing back."""
    rsmiles = utils.reverse_smiles(smiles, renumber_rings=False)
    nsmiles = utils.reverse_smiles(rsmiles, renumber_rings=False)
    assert utils.smiles_eq(smiles, nsmiles)[0], "SMILES are not chemically equivalent"
    assert nsmiles == smiles, "SMILES chemically equivalent but not exactly the same"

@pytest.mark.skip(reason="This is not implemented anywhere as it fails")
@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_ring_numbers(smiles):
    """Check to see that we can safely reindex ring numbers by chronological opening order... this fixes an RDKit Error."""
    csmiles = utils._check_ring_numbers(smiles)
    eq, err = utils.smiles_eq(smiles, csmiles)
    assert eq, f"\nInput {i}: {smiles}\nError: {err}"

@pytest.mark.parametrize("scaffold", TEST_SCAFFOLD)
def test_attachment_points(scaffold):
    """If we get extract attachment points, and put them back in, we should get the same thing back."""
    stripped, at_pts = utils.strip_attachment_points(scaffold) # This calls get_attachment_points
    recycled_smiles = utils.insert_attachment_points(stripped, at_pts)
    eq, error = utils.smiles_eq(scaffold, recycled_smiles)
    assert eq, error # Check if they are the same molecule
    assert scaffold == recycled_smiles # Check if the are the exact same arrangement

@pytest.mark.parametrize("scaffold", TEST_SCAFFOLD)
def test_randomize(scaffold):
    """If we pick every attachment point, randomize it and reverse it, we should get the same molecule back."""
    at_pts = utils.get_attachment_points(scaffold)
    for at_pt in at_pts:
        c_pt = utils.correct_attachment_point(scaffold, at_pt)
        rev_rand_smiles = utils.randomize_smiles(scaffold, rootAtom=c_pt, reverse=True)
        for rev_rand_smi in rev_rand_smiles:
            eq, error = utils.smiles_eq(utils.strip_attachment_points(scaffold)[0], utils.strip_attachment_points(rev_rand_smi)[0])
            assert eq, error


# --------------------- Specific tests ---------------------
@pytest.mark.parametrize("smiles,at_pt,expected",[
    ("c1(*)cc(*)ccc1", 0, 1),
    ("c1(*)cc(*)ccc1", 2, 4)
])
def test_correct_attachment(smiles, at_pt, expected):
    """Check if we can correct attachment points to the actual wildcard"""
    assert utils.correct_attachment_point(smiles, at_pt) == expected