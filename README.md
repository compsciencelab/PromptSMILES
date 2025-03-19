

[![DOI](https://zenodo.org/badge/757912118.svg)](https://zenodo.org/doi/10.5281/zenodo.11161563)


# PromptSMILES: Prompting for scaffold decoration and fragment linking in chemical language models

[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w) |
[Tutorial](https://github.com/Acellera/acegen-open/blob/main/tutorials/using_promptsmiles.md) |
[ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895)


This library contains code to manipulate SMILES strings to facilitate iterative prompting to be coupled with a trained chemical language model (CLM) that uses SMILES notation.

# Installation
The libary can be installed via pip
```
pip install promptsmiles
```
Or via obtaining a copy of this repo, promptsmiles requires RDKit.
```
git clone https://github.com/compsciencelab/PromptSMILES.git
cd PromptSMILES
pip install ./
```

# Use
PromptSMILES is designed as a wrapper to CLM sampling that can accept a prompt (i.e., an initial string to begin autoregressive token generation). Therefore, it requires two callable functions, described later. PromptSMILES has 3 main classes, DeNovo (a dummy wrapper to make code consistent), ScaffoldDecorator, and FragmentLinker.

## Scaffold Decoration
```python
from promptsmiles import ScaffoldDecorator, FragmentLinker

SD = ScaffoldDecorator(
    scaffold="N1(*)CCN(CC1)CCCCN(*)", # Or list of SMILES
    batch_size=64,
    sample_fn=CLM.sampler,
    evaluate_fn=CLM.evaluater,
    batch_prompts=False, # CLM.sampler accepts a list of prompts or not
    optimize_prompts=True,
    shuffle=True, # Randomly select attachment points within a batch or not
    return_all=False,
    )
smiles = SD.sample(batch_size=3, return_all=True) # Parameters can be overriden here if desired
```
![alt text](https://github.com/MorganCThomas/PromptSMILES/blob/main/images/scaff_dec_example.png)

## Superstructure generation
```python
from promptsmiles import ScaffoldDecorator, FragmentLinker

SD = ScaffoldDecorator(
    scaffold="CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C", # Or list of SMILES
    batch_size=64,
    sample_fn=CLM.sampler,
    evaluate_fn=CLM.evaluater,
    batch_prompts=False, # CLM.sampler accepts a list of prompts or not
    optimize_prompts=False,
    shuffle=False, # Randomly select attachment points within a batch or not
    return_all=False,
    )
smiles = SD.sample(batch_size=3, return_all=True) # Parameters can be overriden here if desired
```
![alt text](https://github.com/MorganCThomas/PromptSMILES/blob/main/images/scaff_super_example.png)

## Fragment linking / scaffold hopping
```python
FL = FragmentLinker(
    fragments=["N1(*)CCNCC1", "C1CC1(*)"],
    batch_size=64,
    sample_fn=CLM.sampler,
    evaluate_fn=CLM.evaluater,
    batch_prompts=False,
    optimize_prompts=True,
    shuffle=True,
    scan=False, # Optional when combining 2 fragments, otherwise is set to true
    return_all=False,
)
smiles = FL.sample(batch_size=3)
```
![alt text](https://github.com/MorganCThomas/PromptSMILES/blob/main/images/frag_link_example.png)
## Required chemical language model functions
Notice the callable functions required CLM.sampler and CLM.evaluater. The first is a function that samples from the CLM given a prompt.

```python
def CLM_sampler(prompt: Union[str, list[str]], batch_size: int):
    """
    Input: Must have a prompt and batch_size argument.
    Output: SMILES [list]
    """
    # Encode prompt and sample as per model implementation
    return smiles
```
**Note**: For a more efficient implementation, prompt should accept a list of prompts equal to batch_size and `batch_prompts` should be set to `True` in the promptsmiles class used.

The second is a function that evaluates the NLL of a list of SMILES
```python
def CLM_evaluater(smiles: list[str]):
    """
    Input: A list of SMILES
    Output: NLLs [list, np.array, torch.tensor](CPU w.o. gradient)
    """
    return nlls
```
