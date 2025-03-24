# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import gzip
import os
import sys

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
            TEST_SCAFFOLD.extend([smi.strip() for smi in f.readlines()])
# Load defaults
if not TEST_SMILES:
    with open(os.path.join(os.path.dirname(__file__), "test_smiles.smi")) as f:
        TEST_SMILES.extend([smi.strip() for smi in f.readlines()])
if not TEST_SCAFFOLD:
    with open(os.path.join(os.path.dirname(__file__), "test_scaffolds.scaff")) as f:
        TEST_SCAFFOLD.extend([smi.strip() for smi in f.readlines()])


# --------------------- SMILES tests ------------------------
@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_reverse(smiles):
    """Make an assumption that if we reverse a SMILES string twice, we get the same thing back."""
    rsmiles = utils.reverse_smiles(smiles, renumber_rings=False)
    eq, err = utils.smiles_eq(smiles, rsmiles)
    assert eq, "Reverse SMILES are not chemically equivalent"
    nsmiles = utils.reverse_smiles(rsmiles, renumber_rings=False)
    assert utils.smiles_eq(smiles, nsmiles)[
        0
    ], "Double reverse SMILES are not chemically equivalent"
    assert nsmiles == smiles, "Double reverse SMILES are not exactly the same"


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_ring_numbers(smiles):
    """Check to see that we can safely reindex ring numbers by chronological opening order... this fixes an RDKit Error."""
    csmiles = utils._check_ring_numbers(smiles)
    eq, err = utils.smiles_eq(smiles, csmiles)
    assert eq, "SMILES with corrected ring numbers is not equivalent"
    
@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_superstructure(smiles):
    """Check to see that we can convert smiles to superstructures without throwing an error."""
    assert utils.superstructure_smiles(smiles), "Superstructure conversion failed"


# --------------------- Scaffold tests ----------------------
@pytest.mark.parametrize("scaffold", TEST_SCAFFOLD)
def test_attachment_points(scaffold):
    """If we get extract attachment points, and put them back in, we should get the same thing back."""
    stripped, at_pts = utils.strip_attachment_points(
        scaffold
    )  # This calls get_attachment_points
    recycled_smiles = utils.insert_attachment_points(stripped, at_pts)
    eq, error = utils.smiles_eq(scaffold, recycled_smiles)
    assert eq, error  # Check if they are the same molecule
    assert scaffold == recycled_smiles  # Check if the are the exact same arrangement


@pytest.mark.parametrize("scaffold", TEST_SCAFFOLD)
def test_root(scaffold):
    """If we pick every attachment point, randomize it and reverse it, we should get the same molecule back and an attachment point should be on the end."""
    at_pts = utils.get_attachment_points(scaffold)
    for at_pt in at_pts:
        c_pt = utils.correct_attachment_point(scaffold, at_pt)
        rev_rand_smi = utils.root_smiles(scaffold, rootAtom=c_pt, reverse=True)
        eq, error = utils.smiles_eq(
            utils.strip_attachment_points(scaffold)[0],
            utils.strip_attachment_points(rev_rand_smi)[0],
        )
        assert eq, error
        assert rev_rand_smi.endswith(
            "(*)"
        ), f"Attachment point not at the end of the molecule: {rev_rand_smi}"


@pytest.mark.parametrize("scaffold", TEST_SCAFFOLD)
def test_randomize(scaffold):
    """If we pick every attachment point, randomize it and reverse it, we should get the same molecule back and an attachment point should be on the end."""
    at_pts = utils.get_attachment_points(scaffold)
    for at_pt in at_pts:
        c_pt = utils.correct_attachment_point(scaffold, at_pt)
        rev_rand_smiles = utils.randomize_smiles(scaffold, rootAtom=c_pt, reverse=True)
        for rev_rand_smi in rev_rand_smiles:
            eq, error = utils.smiles_eq(
                utils.strip_attachment_points(scaffold)[0],
                utils.strip_attachment_points(rev_rand_smi)[0],
            )
            assert eq, error
            assert rev_rand_smi.endswith(
                "(*)"
            ), f"Attachment point not at the end of the molecule: {rev_rand_smi}"


# --------------------- Specific tests ---------------------
# smi, attachment_point, RDKit_atom_index
scaffold_parameters = [
    ("c1(*)cc(*)ccc1", 0, 1),
    ("c1(*)cc(*)ccc1", 2, 4),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        0,
        1,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        3,
        5,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        8,
        11,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        9,
        13,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        17,
        22,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        18,
        24,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        19,
        26,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        20,
        28,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        21,
        30,
    ),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        22,
        32,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        0,
        1,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        3,
        5,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        5,
        8,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        6,
        10,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        9,
        14,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        10,
        16,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        11,
        18,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        16,
        24,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        18,
        27,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        19,
        29,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        21,
        32,
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        23,
        35,
    ),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 0, 1),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 1, 3),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 3, 6),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 6, 10),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 7, 12),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 9, 15),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 10, 17),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 12, 20),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 13, 22),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 14, 24),
    ("C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", 15, 26),
    (
        "c1(*)c(*)c(-c2c(OC(*))c(*)nc(*)c2(*))c(*)c(NC(C(c2c(*)c(*)n(C(*))n2)(*))=O)c1(*)",
        15,
        32,
    ),
    (
        "c1(*)c(-c2c(OC(*))c(*)nc(*)c2(*))c(*)c(NC(=O)C(c2c(*)c(*)n(C(*))n2)(*))c(*)c1(*))",
        15,
        31,
    ),
    ("c1ccXc(*)c1", 4, 5),
    ("c1cXXc(*)c1", 4, 5)
]


@pytest.mark.parametrize("smiles,at_pt,rd_pt", scaffold_parameters)
def test_correct_attachment(smiles, at_pt, rd_pt):
    """Check if we can correct attachment points to the actual wildcard"""
    assert utils.correct_attachment_point(smiles, at_pt) == rd_pt


@pytest.mark.parametrize("smiles,at_pt,rd_pt", scaffold_parameters)
def test_get_attachment(smiles, at_pt, rd_pt):
    """Check if we can correct attachment points to the actual wildcard"""
    at_pts = utils.get_attachment_points(smiles)
    assert at_pt in at_pts


# TODO test fragments utils
