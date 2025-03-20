# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import logging
import random
import typing

logger = logging.getLogger("promptsmiles")
from collections import namedtuple
from copy import deepcopy
from typing import Callable

from promptsmiles import utils


class BaseSampler:
    def __init__(
        self,
        sample_fn: Callable,
        evaluate_fn: Callable,
        optimize_prompts: bool,
        **kwargs,
    ):
        self.sample_fn = sample_fn
        self.evaluate_fn = evaluate_fn
        self.optimize_prompts = optimize_prompts
        self.tokenizer = utils.SMILESTokenizer()

    def rearrange_prompt(
        self, smiles: str, at_idx: int, reverse: bool, n_rand: int = 10, **kwargs
    ):
        # ---- Randomize ----
        at_idx = utils.correct_attachment_point(
            smiles, at_idx
        )  # NOTE correcting attachment index to atom index
        if self.optimize_prompts:
            rand_smiles = utils.randomize_smiles(
                smiles,
                n_rand=n_rand,
                random_type="restricted",
                rootAtom=at_idx,
                reverse=reverse,
            )
            if rand_smiles:
                # ---- Evaluate ----
                try:
                    nlls = self.evaluate_fn(
                        [utils.strip_attachment_points(smi)[0] for smi in rand_smiles]
                    )
                    # ---- Sort ----
                    opt_smi, opt_nll = sorted(
                        zip(rand_smiles, nlls), key=lambda x: x[1]
                    )[0]
                    return opt_smi, opt_nll

                except KeyError:
                    # NOTE RDKit sometimes inserts a token that may not have been present in the vocabulary
                    logger.debug(
                        f"SMILES evaluation failed for {smiles} at {at_idx}, rearranging instead..."
                    )

            else:
                # NOTE RDKit sometimes creates duplicate ring indexes for different rings causing an error upon reversal
                logger.debug(
                    f"SMILES randomization failed for {smiles} at {at_idx}, rearranging instead..."
                )

        return utils.root_smiles(smiles, at_idx, reverse=reverse), None


class DeNovo:
    def __init__(
        self, batch_size: int, sample_fn: Callable, sample_fn_kwargs={}, **kwargs
    ):
        """
        A de novo sampling class to generate molecules from scratch i.e., dummy wrap that just calls the sample_fn.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate. Passed to sample_fn.
        sample_fn : Callable
            The sampling function to use in the following format.
            sample_fn(batch_size: int):
                return smiles: list, nlls: Union[list, np.array, torch.tensor] (on CPU without gradients)
        """
        self.batch_size = batch_size
        self.sample_fn = sample_fn
        self.sample_fn_kwargs = sample_fn_kwargs

    def sample(self, batch_size: int = None, **kwargs):
        """
        Sample de novo molecules, see init docstring for more details.
        """
        # Set parameters
        if not batch_size:
            batch_size = self.batch_size

        # Sample
        results = self.sample_fn(batch_size=batch_size, **self.sample_fn_kwargs)

        return results


class ScaffoldDecorator(BaseSampler):
    def __init__(
        self,
        scaffold: typing.Union[str, list],
        batch_size: int,
        sample_fn: Callable,
        evaluate_fn: Callable,
        sample_fn_kwargs: dict = {},
        batch_prompts: bool = False,
        optimize_prompts: bool = True,
        shuffle=True,
        return_all: bool = False,
        random_seed=123,
        force_first: bool = False,
        rdkit_logging=False,
    ):
        """
        A scaffold decorator class to sample from a scaffold constraint via iterative prompting.

        Parameters
        ----------
        scaffold : str | list
            The scaffold SMILES or list of scaffold SMILES to decorate.
        batch_size : int
            The number of samples to generate. Passed to sample_fn.
        sample_fn : Callable
            The sampling function to use in the following format.
            sample_fn(prompt: Union[str, list], batch_size: int):
                return smiles: list, nlls: Union[list, np.array, torch.tensor] (on CPU without gradients)
        evaluate_fn : Callable
            The evaluation function to use in the following format.
            evaluate_fn(smiles: Union[str, list]):
                return nlls: Union[list, np.array, torch.tensor] (on CPU without gradients)
        batch_prompts : bool, optional
            Whether the sample_fn can accept a list of prompts equal to batch_size, by default False
        optimize_prompts : bool, optional
            Whether to optimize the SMILES string for the attachment point, by default True
        shuffle : bool, optional
            Whether to shuffle the attachment points, by default True
        return_all : bool, optional
            Whether to return all intermediate samples, by default False
        random_seed : int, optional
            The random seed to use, by default 123, only for the wrapper and not e.g., torch.
        force_first : bool, optional
            Whether to force the first attachment point to be the first prompt, by default False

        Returns
        -------
        smiles : list
            A list of generated SMILES with a scaffold decorated.
        """
        super().__init__(sample_fn, evaluate_fn, optimize_prompts)
        self.batch_size = batch_size
        self.batch_prompts = batch_prompts
        self.shuffle = shuffle
        self.sample_fn = sample_fn
        self.sample_fn_kwargs = sample_fn_kwargs
        self.evaluate_fn = evaluate_fn
        self.return_all = return_all
        self.force_first = force_first
        self.variant = namedtuple(
            "variant", ["orig_smiles", "opt_smiles", "strip_smiles", "at_pts", "nll"]
        )
        self.seed = random_seed
        random.seed(self.seed)
        if not rdkit_logging:
            utils.disable_rdkit_logging()

        # Prepare scaffold SMILES
        self.scaffolds = []
        self.scaff_idx = 0
        assert isinstance(scaffold, (str, list)), "Scaffold must be a SMILES string or list of SMILES strings."
        scaffold = [scaffold] if isinstance(scaffold, str) else scaffold
        for scaff in scaffold:
            # Check if we need to make it a superstructure
            if "*" not in scaff:
                scaff = utils.superstructure_smiles(scaff)
            # Get attachment points and process variants
            at_pts = utils.get_attachment_points(scaff)
            n_pts = len(at_pts)
            variants = []
            # Optionally force initial configuration as the first prompt
            if self.force_first:
                opt_smi = scaff
                opt_nll = self.evaluate_fn([opt_smi])[0]
                strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                rem_pts.pop(-1)
                at_pts.pop(-1)  # remove last one
                variants.append(
                    self.variant(
                        orig_smiles=scaff,
                        opt_smiles=opt_smi,
                        strip_smiles=strip_smi,
                        at_pts=rem_pts,
                        nll=opt_nll,
                    )
                )
            # Optimize all other attachment points
            opt_variants = []
            for aidx in at_pts:
                opt_smi, opt_nll = self.rearrange_prompt(scaff, aidx, reverse=True)
                strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                rem_pts.pop(-1)
                opt_variants.append(
                    self.variant(
                        orig_smiles=scaff,
                        opt_smiles=opt_smi,
                        strip_smiles=strip_smi,
                        at_pts=rem_pts,
                        nll=opt_nll,
                    )
                )
            if self.optimize_prompts:
                opt_variants.sort(key=lambda x: x.nll)
            variants.extend(opt_variants)
            self.scaffolds.append({'scaff': scaff, 'n_pts': n_pts, 'variants': variants})

    def _single_sample(self, shuffle: bool = None, return_all: bool = None):
        # Set parameters
        if not shuffle:
            shuffle = self.shuffle
        if not return_all:
            return_all = self.return_all

        # Select initial attachment point
        if shuffle:
            # Random scaffold and variant
            scaff_idx = random.randint(0, len(self.scaffolds) - 1)
            var_idx = random.randint(0, self.scaffolds[scaff_idx]['n_pts'] - 1)
        else:
            # Next scaffold and variant
            self.scaff_idx += 1
            scaff_idx = self.scaff_idx % len(self.scaffolds)
            var_idx = 0
        if self.force_first:
            var_idx = 0
        variant = deepcopy(self.scaffolds[scaff_idx]['variants'][var_idx])

        # Sample
        n_rem = self.scaffolds[scaff_idx]['n_pts']
        batch_smiles = []
        while n_rem:
            prompt = variant.strip_smiles
            smiles = self.sample_fn(
                prompt=prompt, batch_size=1, **self.sample_fn_kwargs
            )
            smiles = smiles[0]
            if not smiles.startswith(prompt):
                logger.error(
                    f"Sampled SMILES {smiles} does not start with prompt {prompt}, why not?"
                )
            batch_smiles.append(smiles)
            n_rem -= 1

            if n_rem:
                # Insert remaining attachment points
                smi_w_at = utils.insert_attachment_points(smiles, variant.at_pts)
                # Select another
                if shuffle:
                    i = random.randint(0, n_rem - 1)
                else:
                    i = 0
                sel_pt = variant.at_pts[i]
                # Optimize & strip
                opt_smi, opt_nll = self.rearrange_prompt(smi_w_at, sel_pt, reverse=True)
                if opt_smi:
                    strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                    rem_pts.pop(-1)  # remove the last one
                    variant = self.variant(
                        orig_smiles=smi_w_at,
                        opt_smiles=opt_smi,
                        strip_smiles=strip_smi,
                        at_pts=rem_pts,
                        nll=opt_nll,
                    )
                else:
                    logger.debug(
                        f"SMILES optimization failed for {smiles}, reverting to previous prompt."
                    )
                    # Skip position
                    smi_w_at = utils.insert_attachment_points(prompt, variant.at_pts)
                    opt_smi, opt_nll = self.rearrange_prompt(
                        smi_w_at, sel_pt, reverse=True
                    )
                    if opt_smi:
                        strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                        rem_pts.pop(-1)
                        variant = self.variant(
                            orig_smiles=smi_w_at,
                            opt_smiles=opt_smi,
                            strip_smiles=strip_smi,
                            at_pts=rem_pts,
                            nll=opt_nll,
                        )
                    else:
                        logger.debug(
                            f"SMILES optimization failed for {smiles}, stopping here."
                        )
                        # Stop here
                        variant = self.variant(
                            orig_smiles=smi_w_at,
                            opt_smiles=smiles,
                            strip_smiles=smiles,
                            at_pts=deepcopy(variant.at_pts),
                            nll=None,
                        )

        if return_all:
            return batch_smiles
        else:
            return batch_smiles[-1]

    def _batch_sample(
        self, batch_size: int = None, shuffle: bool = None, return_all: bool = None
    ):
        """More efficient sampling assuming the sample_fn can accept a batch of prompts"""
        # Set parameters
        if not batch_size:
            batch_size = self.batch_size
        if not shuffle:
            shuffle = self.shuffle
        if not return_all:
            return_all = self.return_all

        # Select initial attachment points
        batch_variants = []
        for _ in range(batch_size):
            if shuffle:
                scaff_idx = random.randint(0, len(self.scaffolds) - 1)
                var_idx = random.randint(0, self.scaffolds[scaff_idx]['n_pts'] - 1)
            else:
                self.scaff_idx += 1
                scaff_idx = self.scaff_idx % len(self.scaffolds)
                var_idx = 0
            if self.force_first:
                var_idx = 0
            batch_variants.append(deepcopy(self.scaffolds[scaff_idx]['variants'][var_idx]))

        # Sample
        not_finished = True
        batch_smiles = []
        while not_finished:
            # Sample based on initial prompt
            prompts = [v.strip_smiles for v in batch_variants]
            smiles = self.sample_fn(
                prompt=prompts, batch_size=batch_size, **self.sample_fn_kwargs
            )
            batch_smiles.append(smiles)
            not_finished = any([len(variant.at_pts) for variant in batch_variants])

            if not_finished:
                new_variants = []
                for smi, variant in zip(smiles, batch_variants):
                    if not smi.startswith(variant.strip_smiles):
                        logger.error(
                            f"Sampled SMILES {smi} does not start with prompt {variant.strip_smiles}, why not?"
                        )
                    # If already completed, re-do previous step until all completed
                    if not variant.at_pts:
                        new_variants.append(
                            variant
                        )
                        continue
                    # Insert remaining attachment points
                    smi_w_at = utils.insert_attachment_points(smi, variant.at_pts)
                    # Select another
                    if shuffle:
                        i = random.randint(0, len(variant.at_pts) - 1)
                    else:
                        i = 0
                    sel_pt = variant.at_pts[i]
                    # Optimize & strip
                    opt_smi, opt_nll = self.rearrange_prompt(
                        smi_w_at, sel_pt, reverse=True
                    )
                    if opt_smi:
                        strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                        rem_pts.pop(-1)  # remove the last one
                        # Create new batch_variants?
                        new_variants.append(
                            self.variant(
                                orig_smiles=smi_w_at,
                                opt_smiles=opt_smi,
                                strip_smiles=strip_smi,
                                at_pts=rem_pts,
                                nll=opt_nll,
                            )
                        )
                    else:
                        logger.debug(
                            f"SMILES optimization failed for {smi}, reverting to previous prompt."
                        )
                        # Skip position (variant.strip_smiles = previous_)
                        smi_w_at = utils.insert_attachment_points(
                            variant.strip_smiles, variant.at_pts
                        )
                        opt_smi, opt_nll = self.rearrange_prompt(
                            smi_w_at, sel_pt, reverse=True
                        )
                        if opt_smi:
                            strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                            rem_pts.pop(-1)
                            new_variants.append(
                                self.variant(
                                    orig_smiles=smi_w_at,
                                    opt_smiles=opt_smi,
                                    strip_smiles=strip_smi,
                                    at_pts=rem_pts,
                                    nll=opt_nll,
                                )
                            )
                        else:
                            logger.debug(
                                f"SMILES optimization failed for {smi}, stopping here."
                            )
                            # Stop here
                            new_variants.append(
                                self.variant(
                                    orig_smiles=smi_w_at,
                                    opt_smiles=smi,
                                    strip_smiles=smi,
                                    at_pts=deepcopy(variant.at_pts),
                                    nll=None,
                                )
                            )
                batch_variants = new_variants

        if return_all:
            return batch_smiles
        else:
            return batch_smiles[-1]

    def sample(
        self,
        batch_size: int = None,
        batch_prompts: bool = None,
        return_all: bool = None,
        shuffle: bool = None,
    ):
        """
        Sample de novo molecules, see init docstring for more details.
        
        Parameters
        ----------
        batch_size : int, optional
            The number of samples to generate. Passed to sample_fn, by default None
        batch_prompts : bool, optional
            Whether the sample_fn can accept a list of prompts equal to batch_size, by default None
        return_all : bool, optional
            Whether to return all intermediate samples, by default None
        shuffle : bool, optional
            Whether to shuffle the attachment points, by default None
        
        Returns
        -------
        smiles : list
        """
        # Set parameters
        if not batch_size:
            batch_size = self.batch_size
        if not batch_prompts:
            batch_prompts = self.batch_prompts
        if not shuffle:
            shuffle = self.shuffle
        if not return_all:
            return_all = self.return_all

        # Choose _single_sample, iterative _single_sample, or _batch_sample
        if batch_prompts:
            return self._batch_sample(
                batch_size=batch_size, shuffle=shuffle, return_all=return_all
            )
        else:
            batch_smiles = []
            for i in range(batch_size):
                smiles = self._single_sample(shuffle=shuffle, return_all=return_all)
                batch_smiles.append(smiles)
            # Reformat to be the same as batch_prompts
            if return_all:
                batch_smiles = list(map(list, zip(*batch_smiles)))
            return batch_smiles


class FragmentLinker(BaseSampler):
    def __init__(
        self,
        fragments: list,
        batch_size: int,
        sample_fn: Callable,
        evaluate_fn: Callable,
        sample_fn_kwargs: dict = {},
        batch_prompts: bool = False,
        optimize_prompts: bool = True,
        shuffle: bool = True,
        scan: bool = False,
        detect_existing: bool = True,
        return_all: bool = False,
        random_seed=123,
        rdkit_logging=False,
    ):
        """
        A Fragment linker class to combine different fragments together via iterative prompting.

        Parameters
        ----------
        fragments : list
            The fragmnet SMILES to link together with one specified attachment point each.
        batch_size : int
            The number of samples to generate. Passed to sample_fn.
        sample_fn : Callable
            The sampling function to use in the following format.
            sample_fn(prompt: Union[str, list], batch_size: int):
                return smiles: list, nlls: Union[list, np.array, torch.tensor] (on CPU without gradients)
        evaluate_fn : Callable
            The evaluation function to use in the following format.
            evaluate_fn(smiles: Union[str, list]):
                return nlls: Union[list, np.array, torch.tensor] (on CPU without gradients)
        batch_prompts : bool, optional
            Whether the sample_fn can accept a list of prompts equal to batch_size, by default False
        optimize_prompts : bool, optional
            Whether to optimize the SMILES string for the attachment point, by default True
        shuffle : bool, optional
            Whether to shuffle the attachment points, by default True
        scan : bool, optional
            Whether to evaluate fragment attachment through the whole linker as oppose to simple concatenation, by default False, automatically enabled for more than two fragments
        return_all : bool, optional
            Whether to return all intermediate samples, by default False
        random_seed : int, optional
            The random seed to use, by default 123, only for the wrapper and not e.g., torch.

        Returns
        -------
        smiles : list
            A list of generated SMILES with a fragments linked.
        """
        super().__init__(sample_fn, evaluate_fn, optimize_prompts)
        self.batch_size = batch_size
        self.batch_prompts = batch_prompts
        self.shuffle = shuffle
        self.scan = scan
        self.detect_existing = detect_existing
        self.sample_fn = sample_fn
        self.sample_fn_kwargs = sample_fn_kwargs
        self.evaluate_fn = evaluate_fn
        self.return_all = return_all
        self.fragment = namedtuple(
            "fragment", ["for_smiles", "for_nll", "rev_smiles", "rev_nll"]
        )
        self.seed = random_seed
        random.seed(self.seed)
        if not rdkit_logging:
            utils.disable_rdkit_logging()
            
        # Check fragments
        if any(["X" in f for f in fragments]):
            raise NotImplementedError("FragmentLinker does not support X substitution yet.")

        # Correct scan
        if len(fragments) > 2 and not self.scan:
            logger.warn(
                "Scan must be used for more than two fragments, Scan will be enabled."
            )
            self.scan = True

        # Prepare fragments
        self.n_fgs = len(fragments)
        self.fragments = []
        for frag in fragments:
            # Get attachment index
            aidx = utils.get_attachment_points(frag)
            assert (
                len(aidx) == 1
            ), f"Fragment {frag} should only have one attachment point"
            # Optimize forward direction
            for_smi, for_nll = self.rearrange_prompt(frag, aidx[0], reverse=False)
            # Optimize reverse direction
            rev_smi, rev_nll = self.rearrange_prompt(frag, aidx[0], reverse=True)
            # Append
            self.fragments.append(
                self.fragment(
                    for_smiles=utils.strip_attachment_points(for_smi)[0],
                    for_nll=for_nll,
                    rev_smiles=utils.strip_attachment_points(rev_smi)[0],
                    rev_nll=rev_nll,
                )
            )
        if self.optimize_prompts:
            self.fragments.sort(key=lambda x: x.for_nll)

    def _single_sample(
        self,
        shuffle: bool = None,
        scan: bool = None,
        detect_existing: bool = None,
        return_all: bool = None,
    ):
        # Set parameters
        if not shuffle:
            shuffle = self.shuffle
        if not scan:
            scan = self.scan
        if not detect_existing:
            detect_existing = self.detect_existing
        if not return_all:
            return_all = self.return_all
        fragments = deepcopy(self.fragments)

        # Randomly select starting attachment points
        if shuffle:
            i = random.randint(0, self.n_fgs - 1)
        else:
            i = 0
        f0 = fragments.pop(i)

        # Sample
        n_rem = self.n_fgs
        batch_smiles = []
        prompt = f0.rev_smiles
        prompt_tokens = self.tokenizer.tokenize(prompt, with_begin_and_end=False)
        frag_indexes = list(range(len(prompt_tokens)))
        smiles = self.sample_fn(prompt=prompt, batch_size=1, **self.sample_fn_kwargs)
        smiles = smiles[0]
        assert smiles.startswith(
            prompt
        ), f"Sampled SMILES {smiles} does not start with prompt {prompt}, why not?"
        smiles_tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=False)
        batch_smiles.append(smiles)
        n_rem -= 1

        if self.scan:
            while n_rem:
                # Select another fragment
                if shuffle:
                    i = random.randint(0, n_rem - 1)
                else:
                    i = 0
                fi = fragments.pop(i)
                # Detect existing fragments
                if detect_existing:
                    exists = False
                    existing_atoms = utils.detect_existing_fragment(
                        smiles, fi.for_smiles
                    )
                    # Get an atom map between atoms and tokens
                    atom_map = self.tokenizer._token2atom_map(smiles_tokens)
                    # Check it's not generated linker
                    for match in existing_atoms:
                        if not any([atom_map[aidx] in frag_indexes for aidx in match]):
                            # If so, insert fragment indexes, append etc.
                            frag_indexes.extend([atom_map[aidx] for aidx in match])
                            batch_smiles.append(smiles)
                            n_rem -= 1
                            exists = True
                            break
                    if exists:
                        continue
                # Correct rings
                fi_smi = utils.correct_fragment_ring_numbers(smiles, fi.for_smiles)
                # Insert fragment at different positions
                temp_smiles = []
                for i in range(len(prompt_tokens) - 1, len(smiles_tokens)):
                    if i in frag_indexes:
                        continue
                    else:
                        # NOTE operate in token space to reduce errors
                        tsmi = "".join(
                            smiles_tokens[: i + 1]
                            + ["("]
                            + [fi_smi]
                            + [")"]
                            + smiles_tokens[i + 1 :]
                        )
                        tidx = list(range(i + 1, i + len(fi_smi) + 2))
                        temp_smiles.append((tsmi, tidx))
                if temp_smiles:
                    # Select best position
                    temp_nlls = self.evaluate_fn([smi for smi, _ in temp_smiles])
                    (smiles, fidxs), nll = sorted(
                        zip(temp_smiles, temp_nlls), key=lambda x: x[1]
                    )[0]
                    smiles_tokens = self.tokenizer.tokenize(
                        smiles, with_begin_and_end=False
                    )
                    batch_smiles.append(smiles)
                    frag_indexes.extend(fidxs)
                else:
                    # Don't add fragment
                    batch_smiles.append(smiles)
                n_rem -= 1
        else:
            fi = fragments.pop(0)
            # Detect existing fragments
            exists = False
            if detect_existing:
                existing_atoms = utils.detect_existing_fragment(smiles, fi.for_smiles)
                # Get an atom map between atoms and tokens
                atom_map = self.tokenizer._token2atom_map(smiles_tokens)
                for match in existing_atoms:
                    # Check it's not generated linker
                    if not any([atom_map[aidx] in frag_indexes for aidx in match]):
                        # If so, insert fragment indexes, append etc.
                        batch_smiles.append(smiles)
                        n_rem -= 1
                        exists = True
                        break
            if not exists:
                # Correct rings
                fi_smi = utils.correct_fragment_ring_numbers(smiles, fi.for_smiles)
                # Append fragment
                smiles = smiles + fi_smi
                # Evaluate
                batch_smiles.append(smiles)
                n_rem -= 1

        if return_all:
            return batch_smiles
        else:
            return batch_smiles[-1]

    def _batch_sample(
        self,
        batch_size: int = None,
        shuffle: bool = None,
        scan: bool = None,
        detect_existing: bool = None,
        return_all: bool = None,
    ):
        """More efficient sampling assuming the sample_fn can accept a batch of prompts"""
        # Set parameters
        if not batch_size:
            batch_size = self.batch_size
        if not shuffle:
            shuffle = self.shuffle
        if not scan:
            scan = self.scan
        if not detect_existing:
            detect_existing = self.detect_existing
        if not return_all:
            return_all = self.return_all
        batch_fragments = [
            deepcopy(self.fragments) for _ in range(batch_size)
        ]  # NOTE repeating with itertools or by *x leads to abberant behaviour with pop removing element from all sublists

        # Select initial fragments
        prompts = []
        frag_indexes = []
        for bi in range(batch_size):
            if shuffle:
                i = random.randint(0, self.n_fgs - 1)
            else:
                i = 0
            f0 = batch_fragments[bi].pop(i)
            f0_tokens = self.tokenizer.tokenize(f0.rev_smiles, with_begin_and_end=False)
            prompts.append(f0.rev_smiles)
            frag_indexes.append(list(range(len(f0_tokens))))

        # Sample
        n_rem = self.n_fgs
        batch_smiles = []
        smiles = self.sample_fn(
            prompt=prompts, batch_size=batch_size, **self.sample_fn_kwargs
        )
        batch_smiles.append(smiles)
        n_rem -= 1

        if self.scan:
            while n_rem:
                n_smiles = []
                for bi, (smiles, fragments, existing_indexes) in enumerate(
                    zip(batch_smiles[-1], batch_fragments, frag_indexes)
                ):
                    assert smiles.startswith(
                        prompts[bi]
                    ), f"Sampled SMILES {smiles} does not start with prompt {prompts[bi]}, why not?"
                    # Select another fragment
                    if shuffle:
                        i = random.randint(0, n_rem - 1)
                    else:
                        i = 0
                    fi = fragments.pop(i)
                    # Detect existing fragments
                    if detect_existing:
                        exists = False
                        smiles_tokens = self.tokenizer.tokenize(
                            smiles, with_begin_and_end=False
                        )
                        existing_atoms = utils.detect_existing_fragment(
                            smiles, fi.for_smiles
                        )
                        # Get an atom map between atoms and tokens
                        atom_map = self.tokenizer._token2atom_map(smiles_tokens)
                        # Check it's not generated linker
                        for match in existing_atoms:
                            if not any(
                                [atom_map[aidx] in frag_indexes for aidx in match]
                            ):
                                # If so, insert fragment indexes, append etc.
                                frag_indexes[bi].extend(
                                    [atom_map[aidx] for aidx in match]
                                )
                                n_smiles.append(smiles)
                                exists = True
                                break
                        if exists:
                            continue
                    # Correct rings
                    fi_smi = utils.correct_fragment_ring_numbers(smiles, fi.for_smiles)
                    # Insert fragment at different positions
                    smiles_tokens = self.tokenizer.tokenize(
                        smiles, with_begin_and_end=False
                    )
                    prompt_tokens = self.tokenizer.tokenize(
                        prompts[bi], with_begin_and_end=False
                    )
                    temp_smiles = []
                    for i in range(len(prompt_tokens) - 1, len(smiles_tokens)):
                        if i in existing_indexes:
                            continue
                        else:
                            tsmi = "".join(
                                smiles_tokens[: i + 1]
                                + ["("]
                                + [fi_smi]
                                + [")"]
                                + smiles_tokens[i + 1 :]
                            )
                            tidx = list(range(i + 1, i + len(fi_smi) + 2))
                            temp_smiles.append((tsmi, tidx))
                    if temp_smiles:
                        # Select best position
                        temp_nlls = self.evaluate_fn([smi for smi, _ in temp_smiles])
                        (smiles, fidxs), nll = sorted(
                            zip(temp_smiles, temp_nlls), key=lambda x: x[1]
                        )[0]
                        n_smiles.append(smiles)
                        frag_indexes[bi].extend(fidxs)
                    else:
                        # Don't add fragment
                        n_smiles.append(smiles)
                batch_smiles.append(n_smiles)
                n_rem -= 1
        else:
            concat_smiles = []
            for bi, (smiles, fragments) in enumerate(
                zip(batch_smiles[0], batch_fragments)
            ):
                assert smiles.startswith(
                    prompts[bi]
                ), f"Sampled SMILES {smiles} does not start with prompt {prompts[bi]}, why not?"
                fi = fragments.pop(0)
                # Detect existing fragments
                exists = False
                if detect_existing:
                    smiles_tokens = self.tokenizer.tokenize(
                        smiles, with_begin_and_end=False
                    )
                    existing_atoms = utils.detect_existing_fragment(
                        smiles, fi.for_smiles
                    )
                    # Get an atom map between atoms and tokens
                    atom_map = self.tokenizer._token2atom_map(smiles_tokens)
                    for match in existing_atoms:
                        # Check it's not generated linker
                        if not any([atom_map[aidx] in frag_indexes for aidx in match]):
                            # If so, insert fragment indexes, append etc.
                            concat_smiles.append(smiles)
                            exists = True
                            break
                if not exists:
                    # Correct rings
                    fi_smi = utils.correct_fragment_ring_numbers(smiles, fi.for_smiles)
                    # Append fragment
                    smiles = smiles + fi_smi
                    concat_smiles.append(smiles)
            # Evaluate
            batch_smiles.append(concat_smiles)
            n_rem -= 1

        if return_all:
            return batch_smiles
        else:
            return batch_smiles[-1]

    def sample(
        self,
        batch_size: int = None,
        batch_prompts: bool = None,
        return_all: bool = None,
        shuffle: bool = None,
        scan: bool = None,
        **kwargs,
    ):
        """
        Sample de novo molecules, see init docstring for more details.
        """
        # Set parameters
        if not batch_size:
            batch_size = self.batch_size
        if not batch_prompts:
            batch_prompts = self.batch_prompts
        if not shuffle:
            shuffle = self.shuffle
        if not scan:
            scan = self.scan
        if not return_all:
            return_all = self.return_all
        # Correct scan
        if self.n_fgs > 2 and not self.scan:
            logger.warn(
                "Scan must be used for more than two fragments, Scan will be enabled."
            )
            self.scan = True

        # Choose _single_sample, iterative _single_sample, or _batch_sample
        if batch_prompts:
            return self._batch_sample(
                batch_size=batch_size, shuffle=shuffle, return_all=return_all
            )
        else:
            batch_smiles = []
            for i in range(batch_size):
                smiles = self._single_sample(shuffle=shuffle, return_all=return_all)
                batch_smiles.append(smiles)
            # Reformat to be the same as batch_prompts
            if return_all:
                batch_smiles = list(map(list, zip(*batch_smiles)))
            return batch_smiles
