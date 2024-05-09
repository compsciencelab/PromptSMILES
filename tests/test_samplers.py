# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import random

import pytest

from promptsmiles import samplers, utils


def evaluate_fn(smiles: list):
    return [random.randint(0, 40) for i in range(len(smiles))]


def sample_scaffold_fn(prompt, batch_size):
    """Dummy sampler that adds a F"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] + "F" for i in range(batch_size)]
    else:
        return [prompt + "F" for _ in range(batch_size)]


def sample_scaffold_fn2(prompt, batch_size):
    """Dummy sampler that adds nothing"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] for i in range(batch_size)]
    else:
        return [prompt for _ in range(batch_size)]


def sample_linker_fn(prompt, batch_size):
    """Dummy sampler that adds a PEG monomer"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] + "CCOCC" for i in range(batch_size)]
    else:
        return [prompt + "CCOCC" for _ in range(batch_size)]


def sample_linker_fn2(prompt, batch_size):
    """Dummy sampler that adds nothing"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] for i in range(batch_size)]
    else:
        return [prompt for _ in range(batch_size)]


# --------------------- Scaffold decoration ---------------------
scaffold_params = [
    ("c1c(Cl)c(*)c(*)cc1", "c1c(Cl)c(F)c(F)cc1"),
    ("C(*)C1=C(*)C(*)N=C(c2nccs2)N1", "C(F)C1=C(F)C(F)N=C(c2nccs2)N1"),
    (
        "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
        "C(F)c1nc(F)n2[nH]c(-c3c(F)c(F)c(F)c(S(=O)(=O)N4C(F)C(F)N(F)C(F)C4(F))c3(F))nc(=O)c12",
    ),
    (
        "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
        "C(F)Oc1c(F)nc(F)c(F)c1-c1c(F)c(F)c(F)c(NC(=O)C(F)c2c(F)c(F)n(C(F))n2)c1(F)",
    ),
    (
        "C(*)C(*)c1c(*)nc(-c2c(*)c(*)c(*)c(*)c2(*))c(C(*)(C(*)c2c(*)c(*)c(*)c(*)c2(*))NC(=O)C(*)n2nc(C(*)(F))c3c2C(C(*))(F)C2(*)C(*)C32(*))n1",
        "C(F)C(F)c1c(F)nc(-c2c(F)c(F)c(F)c(F)c2(F))c(C(F)(C(F)c2c(F)c(F)c(F)c(F)c2(F))NC(=O)C(F)n2nc(C(F)(F))c3c2C(C(F))(F)C2(F)C(F)C32(F))n1",
    ),
    (
        "C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O",
        "C(F)C(F)N(C(F)=O)c1c(F)c(F)c(F)c(F)c1(F)N1C(F)C(F)C(F)C(F)C1=O",
    ),
    (
        "N(*)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O",
        "N(F)c1c(F)c(F)c(F)c(F)c1(F)N1C(F)C(F)C(F)C(F)C1=O",
    ),
]


@pytest.mark.parametrize("scaffold,expected", scaffold_params)
def test_decoration_canonical(scaffold, expected):
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffold,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("scaffold,expected", scaffold_params)
def test_decoration_shuffle(scaffold, expected):
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffold,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=True,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("scaffold,expected", scaffold_params)
def test_decoration_batch_canonical(scaffold, expected):
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffold,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("scaffold,expected", scaffold_params)
def test_decoration_batch_shuffle(scaffold, expected):
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffold,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=True,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


# --------------------- Fragment linking ---------------------
fragment_params1 = [  # frag1, frag2, expected
    ("C1CC1(*)", "c1cc(*)cc(Cl)c1", "C1CC1CCOCCc1cc(Cl)ccc1")
]


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_canonical(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=False,
        scan=False,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_shuffle(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=True,
        scan=False,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_batch_canonical(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=False,
        scan=False,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_batch_shuffle(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=True,
        scan=False,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles in denovo])


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_scan(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=True,
        scan=True,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    # Extract linker, and check it's what we expect
    for smiles in denovo:
        linker = utils.extract_linker(smiles, [frag1, frag2])
        if linker:
            assert utils.strip_attachment_points(linker)[0] == "CCOCC"


@pytest.mark.parametrize("frag1,frag2,expected", fragment_params1)
def test_linking_batch_scan(frag1, frag2, expected):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=True,
        scan=True,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    # Extract linker, and check it's what we expect
    for smiles in denovo:
        linker = utils.extract_linker(smiles, [frag1, frag2])
        if linker:
            assert utils.strip_attachment_points(linker)[0] == "CCOCC"


# --------------------- Fragment linking more than two fragments -------------------

fragment_params2 = [  # frag1, frag2, frag3
    ("C1CC1(*)", "c1cc(*)cc(Cl)c1", "N1(*)CCC1")
]


@pytest.mark.parametrize("frag1,frag2,frag3", fragment_params2)
def test_linking_scan_multi(frag1, frag2, frag3):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2, frag3],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=True,
        scan=True,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    # Extract linker, and check it's what we expect
    for smiles in denovo:
        linker = utils.extract_linker(smiles, [frag1, frag2, frag3])
        if linker:
            assert utils.smiles_eq(utils.strip_attachment_points(linker)[0], "CCOCC")[0]


@pytest.mark.parametrize("frag1,frag2,frag3", fragment_params2)
def test_linking_batch_scan_multi(frag1, frag2, frag3):
    SD = samplers.FragmentLinker(
        fragments=[frag1, frag2, frag3],
        batch_size=10,
        sample_fn=sample_linker_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=True,
        scan=True,
        detect_existing=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    # Extract linker, and check it's what we expect
    for smiles in denovo:
        linker = utils.extract_linker(smiles, [frag1, frag2, frag3])
        if linker:
            assert utils.smiles_eq(utils.strip_attachment_points(linker)[0], "CCOCC")[0]
