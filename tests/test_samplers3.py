# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import random

import pytest

from promptsmiles import samplers, utils


def evaluate_fn(smiles: list):
    return [random.randint(0, 40) for i in range(len(smiles))]


def sample_scaffold_fn(prompt, batch_size):
    """Dummy sampler that adds nothing"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] for i in range(batch_size)]
    else:
        return [prompt for _ in range(batch_size)]


def sample_linker_fn(prompt, batch_size):
    """Dummy sampler that adds nothing"""
    if isinstance(prompt, list):
        assert (
            len(prompt) == batch_size
        ), "Prompts provided is not the same as batch size requested"
        return [prompt[i] for i in range(batch_size)]
    else:
        return [prompt for _ in range(batch_size)]


# --------------------- Scaffold decoration ---------------------

@pytest.fixture
def scaffold_params():
    return [
        ("c1c(Cl)c(*)c(*)cc1", "c1c(Cl)cccc1"),
        ("C(*)C1=C(*)C(*)N=C(c2nccs2)N1", "CC1=CCN=C(c2nccs2)N1"),
        (
            "C(*)c1nc(*)n2[nH]c(-c3c(*)c(*)c(F)c(S(=O)(=O)N4C(*)C(*)N(*)C(*)C4(*))c3(*))nc(=O)c12",
            "Cc1ncn2[nH]c(-c3ccc(F)c(S(=O)(=O)N4CCNCC4)c3)nc(=O)c12",
        ),
        (
            "C(*)Oc1c(*)nc(*)c(*)c1-c1c(*)c(*)c(*)c(NC(=O)C(*)c2c(*)c(*)n(C(*))n2)c1(*)",
            "COc1cnccc1-c1cccc(NC(=O)Cc2ccn(C)n2)c1",
        ),
        (
            "C(*)C(*)c1c(*)nc(-c2c(*)c(*)c(*)c(*)c2(*))c(C(*)(C(*)c2c(*)c(*)c(*)c(*)c2(*))NC(=O)C(*)n2nc(C(*)(F))c3c2C(C(*))(F)C2(*)C(*)C32(*))n1",
            "CCc1cnc(-c2ccccc2)c(C(Cc2ccccc2)NC(=O)Cn2nc(C)c3c2C(C)C2CC32)n1",
        ),
        (
            "C(*)C(*)N(C(*)=O)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O",
            "CCN(C=O)c1ccccc1N1CCCCC1=O",
        ),
        ("N(*)c1c(*)c(*)c(c(*)c1(*))N1C(*)C(*)C(*)C(*)C1=O", "Nc1ccccc1N1CCCCC1=O"),
    ]

def test_decoration_canonical(scaffold_params):
    scaffolds = [i[0] for i in scaffold_params]
    expecteds = [i[1] for i in scaffold_params]
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffolds,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles, expected in zip(denovo, expecteds)])


def test_decoration_shuffle(scaffold_params):
    scaffolds = [i[0] for i in scaffold_params]
    expecteds = [i[1] for i in scaffold_params]
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffolds,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=False,
        shuffle=True,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    for smiles in denovo:
        assert any([utils.smiles_eq(smiles, expected) for expected in expecteds])


def test_decoration_batch_canonical(scaffold_params):
    scaffolds = [i[0] for i in scaffold_params]
    expecteds = [i[1] for i in scaffold_params]
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffolds,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=False,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    assert all([utils.smiles_eq(smiles, expected) for smiles, expected in zip(denovo, expecteds)])


def test_decoration_batch_shuffle(scaffold_params):
    scaffolds = [i[0] for i in scaffold_params]
    expecteds = [i[1] for i in scaffold_params]
    SD = samplers.ScaffoldDecorator(
        scaffold=scaffolds,
        batch_size=10,
        sample_fn=sample_scaffold_fn,
        evaluate_fn=evaluate_fn,
        batch_prompts=True,
        shuffle=True,
        return_all=False,
    )
    denovo = SD.sample()
    assert len(denovo) == 10
    for smiles in denovo:
        assert any([utils.smiles_eq(smiles, expected) for expected in expecteds])
