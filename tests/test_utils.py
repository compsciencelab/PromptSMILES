import pytest
from promptsmiles import utils

# TODO test fragments
# TODO test randomize with random root?

# --------------------- Broad SMILES tests ------------------------
def test_reverse(TEST_SMILES):
    """Make an assumption that if we reverse a SMILES string twice, we get the same thing back."""
    for smiles in TEST_SMILES:
        rsmiles = utils.reverse_smiles(smiles, renumber_rings=False)
        nsmiles = utils.reverse_smiles(rsmiles, renumber_rings=False)
        assert utils.smiles_eq(smiles, nsmiles)[0], "SMILES are not chemically equivalent"
        assert nsmiles == smiles, "SMILES chemically equivalent but not exactly the same"


# --------------------- Broad Scaffold tests ---------------------

#@pytest.mark.parametrize("smiles", TEST_SCAFFOLDS)
def test_attachment_points(TEST_SCAFFOLDS):
    """If we get extract attachment points, and put them back in, we should get the same thing back."""
    for scaffold in TEST_SCAFFOLDS:
        stripped, at_pts = utils.strip_attachment_points(scaffold) # This calls get_attachment_points
        recycled_smiles = utils.insert_attachment_points(stripped, at_pts)
        eq, error = utils.smiles_eq(scaffold, recycled_smiles)
        assert eq, error # Check if they are the same molecule
        assert scaffold == recycled_smiles # Check if the are the exact same arrangement


# --------------------- Specific tests ---------------------
@pytest.mark.parametrize("smiles,at_pt,expected",[
    ("c1(*)cc(*)ccc1", 0, 1),
    ("c1(*)cc(*)ccc1", 2, 4)
])
def test_correct_attachment(smiles, at_pt, expected):
    """Check if we can correct attachment points to the actual wildcard"""
    assert utils.correct_attachment_point(smiles, at_pt) == expected