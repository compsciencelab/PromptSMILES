import random
import warnings
from copy import deepcopy
from typing import Callable
from collections import namedtuple

from promptsmiles import utils


class BaseSampler:
    def __init__(self, sample_fn: Callable, evaluate_fn: Callable, **kwargs):
        self.sample_fn = sample_fn
        self.evaluate_fn = evaluate_fn
        self.tokenizer = utils.SMILESTokenizer()

    def optimize_prompt(self, smiles: str, at_idx: int, reverse: bool, n_rand: int = 10, **kwargs):
        # ---- Randomize ----
        # NOTE need to correct attachment index from of attachment point to index of "*"
        at_idx = utils.correct_attachment_idx(smiles, at_idx)
        rand_smi = utils.randomize_smiles(smiles, n_rand=n_rand, random_type='restricted', rootAtom=at_idx, reverse=reverse)
        if rand_smi is None:
            warnings.warn(f"SMILES randomization failed for {smiles}, rearranging instead...")
            return utils.root_smiles(smiles, at_idx, reverse=reverse), None
        # ---- Evaluate ----
        try:
            nlls = self.evaluate_fn([utils.strip_attachment_points(smi)[0] for smi in rand_smi])
        except KeyError:
            # NOTE RDKit sometimes inserts a token that may not have been present in the vocabulary
            warnings.warn(f"SMILES randomization failed for {smiles}, rearranging instead...")
            return utils.root_smiles(smiles, at_idx, reverse=reverse), None
        # ---- Sort ----
        opt_smi, opt_nll = sorted(zip(rand_smi, nlls), key=lambda x: x[1])[0]
        return opt_smi, opt_nll


class DeNovo:
    def __init__(self, batch_size: int, sample_fn: Callable, **kwargs):
        """
        A de novo sampling class to generate molecules from scratch i.e., dummy wrap that just calls the sample_fn.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate. Passed to sample_fn.
        sample_fn : Callable
            The sampling function to use in the following format.
            sample_fn(batch_size: int):
                return smiles: list, nlls: Union[list, np.array, torch.tensor]
        """
        self.batch_size = batch_size
        self.sample_fn = sample_fn

    def sample(self, batch_size: int = None, **kwargs):
        """
        Sample de novo molecules, see init docstring for more details.
        """
        # Set parameters
        if not batch_size: batch_size = self.batch_size
        
        # Sample
        results = self.sample_fn(batch_size=batch_size)
        
        return results


class ScaffoldDecorator(BaseSampler):
    def __init__(self, scaffold: str, batch_size: int, sample_fn: Callable, evaluate_fn: Callable, batch_prompts: bool = False, shuffle=True, return_all: bool = False, random_seed = 123, force_first: bool = False):
        """
        A scaffold decorator class to sample from a scaffold constraint via iterative prompting.

        Parameters
        ----------
        scaffold : str
            The scaffold SMILES to decorate.
        batch_size : int
            The number of samples to generate. Passed to sample_fn.
        sample_fn : Callable
            The sampling function to use in the following format.
            sample_fn(prompt: Union[str, list], batch_size: int):
                return smiles: list, nlls: Union[list, np.array, torch.tensor]
        evaluate_fn : Callable
            The evaluation function to use in the following format.
            evaluate_fn(smiles: Union[str, list]):
                return nlls: Union[list, np.array, torch.tensor]
        batch_prompts : bool, optional
            Whether the sample_fn can accept a list of prompts equal to batch_size, by default False
        shuffle : bool, optional
            Whether to shuffle the attachment points, by default True
        return_all : bool, optional
            Whether to return all intermediate samples, by default False
        random_seed : int, optional
            The random seed to use, by default 123
        force_first : bool, optional
            Whether to force the first attachment point to be the first prompt, by default False
        
        Returns
        -------
        smiles : list
            A list of generated SMILES with a scaffold decorated.
        """
        super().__init__(sample_fn, evaluate_fn)
        self.batch_size = batch_size
        self.batch_prompts = batch_prompts
        self.shuffle = shuffle
        self.sample_fn = sample_fn
        self.evaluate_fn = evaluate_fn
        self.return_all = return_all
        self.scaffold = scaffold
        self.force_first = force_first
        self.variant = namedtuple('variant', ['orig_smiles', 'opt_smiles', 'strip_smiles', 'at_pts', 'nll'])
        self.seed = random_seed
        random.seed(self.seed)

        # Prepare scaffold SMILES
        self.at_pts = utils.get_attachment_indexes(self.scaffold)
        self.n_pts = len(self.at_pts)
        self.variants = []
        # Optionally force initial configuration as the first prompt
        if self.force_first:
            opt_smi = self.scaffold
            opt_nll = self.evaluate_fn([opt_smi])[0]
            strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
            rem_pts.pop(-1)
            self.at_pts.pop(-1) # remove last one
            self.variants.append(
                self.variant(
                    orig_smiles=self.scaffold,
                    opt_smiles=opt_smi,
                    strip_smiles=strip_smi,
                    at_pts=rem_pts,
                    nll=opt_nll
                    )
                )
        # Optimize all other attachment points
        variants = []
        for aidx in self.at_pts:
            opt_smi, opt_nll = self.optimize_prompt(self.scaffold, aidx, reverse=True)
            strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
            rem_pts.pop(-1)
            variants.append(
                self.variant(
                    orig_smiles=self.scaffold,
                    opt_smiles=opt_smi,
                    strip_smiles=strip_smi,
                    at_pts=rem_pts,
                    nll=opt_nll
                    )
                )
        variants.sort(key=lambda x: x.nll)
        self.variants.extend(variants)

    def _single_sample(self, shuffle: bool = None, return_all: bool = None):
        # Set parameters
        if not shuffle: shuffle = self.shuffle
        if not return_all: return_all = self.return_all

        # Select initial attachment point
        if shuffle: i = random.randint(0, self.n_pts-1)
        else: i = 0
        if self.force_first: i = 0
        variant = deepcopy(self.variants[i])

        # Sample
        n_rem = self.n_pts
        batch_smiles = []
        batch_nlls = []
        while n_rem:
            prompt = variant.strip_smiles
            smiles, nll = self.sample_fn(prompt=prompt, batch_size=1)
            smiles, nll = smiles[0], nll[0]
            batch_smiles.append(smiles)
            batch_nlls.append(nll)
            n_rem -= 1

            if n_rem:
                # Insert remaining attachment points
                smi_w_at, _ = utils.insert_attachment_points(smiles, variant.at_pts)
                # Select another
                if shuffle: i = random.randint(0, n_rem-1)
                else: i = 0
                sel_pt = variant.at_pts.pop(i)
                # Optimize & strip
                opt_smi, opt_nll = self.optimize_prompt(smi_w_at, sel_pt, reverse=True)
                if opt_smi:
                    strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                    rem_pts.pop(-1) # remove the last one
                    # Update variant
                    variant = self.variant(
                        orig_smiles=smi_w_at,
                        opt_smiles=opt_smi,
                        strip_smiles=strip_smi,
                        at_pts=rem_pts,
                        nll=opt_nll
                        )
                else:
                    warnings.warn(f"SMILES optimization failed for {smiles}, stopping here.")
                    # Update variant
                    variant = self.variant(
                        orig_smiles=smi_w_at,
                        opt_smiles=smiles,
                        strip_smiles=smiles,
                        at_pts=variant.at_pts,
                        nll=None
                        )


        if return_all:
            return batch_smiles, batch_nlls
        else:
            return batch_smiles[-1], batch_nlls[-1]
    
    def _batch_sample(self, batch_size: int = None, shuffle: bool = None, return_all: bool = None):
        """More efficient sampling assuming the sample_fn can accept a batch of prompts"""
        # Set parameters
        if not batch_size: batch_size = self.batch_size
        if not shuffle: shuffle = self.shuffle
        if not return_all: return_all = self.return_all

        # Select initial attachment points
        batch_variants = []
        for _ in range(batch_size):
            if shuffle: i = random.randint(0, self.n_pts-1)
            else: i = 0
            if self.force_first: i = 0
            batch_variants.append(deepcopy(self.variants[i]))

        # Sample
        n_rem = self.n_pts
        batch_smiles = []
        batch_nlls = []
        while n_rem:
            # Sample based on initial prompt
            prompts = [v.strip_smiles for v in batch_variants]
            smiles, nlls = self.sample_fn(prompt=prompts, batch_size=batch_size)
            batch_smiles.append(smiles)
            batch_nlls.append(nlls)
            n_rem -= 1

            if n_rem:
                new_variants = []
                for smi, variant in zip(smiles, batch_variants):
                    # Insert remaining attachment points
                    smi_w_at, _ = utils.insert_attachment_points(smi, variant.at_pts)
                    # Select another
                    if shuffle: i = random.randint(0, n_rem-1)
                    else: i = 0
                    sel_pt = variant.at_pts.pop(i)
                    # Optimize & strip
                    opt_smi, opt_nll = self.optimize_prompt(smi_w_at, sel_pt, reverse=True)
                    if opt_smi:
                        strip_smi, rem_pts = utils.strip_attachment_points(opt_smi)
                        rem_pts.pop(-1) # remove the last one
                        # Create new batch_variants?
                        new_variants.append(
                            self.variant(
                                orig_smiles=smi_w_at,
                                opt_smiles=opt_smi,
                                strip_smiles=strip_smi,
                                at_pts=rem_pts,
                                nll=opt_nll
                                )
                            )
                    else:
                        warnings.warn(f"SMILES optimization failed for {smiles}, stopping here.")
                        new_variants.append(
                            self.variant(
                                orig_smiles=smi_w_at,
                                opt_smiles=smi,
                                strip_smiles=smi,
                                at_pts=variant.at_pts,
                                nll=None
                                )
                            )
                batch_variants = new_variants

        if return_all:
            return batch_smiles, batch_nlls
        else:
            return batch_smiles[-1], batch_nlls[-1]

    def sample(self, batch_size: int = None, batch_prompts: bool = None, return_all: bool = None, shuffle: bool = None):
        """
        Sample de novo molecules, see init docstring for more details.
        """
        # Set parameters
        if not batch_size: batch_size = self.batch_size
        if not batch_prompts: batch_prompts = self.batch_prompts
        if not shuffle: shuffle = self.shuffle
        if not return_all: return_all = self.return_all

        # Choose _single_sample, iterative _single_sample, or _batch_sample
        if batch_prompts:
            return self._batch_sample(batch_size=batch_size, shuffle=shuffle, return_all=return_all)
        else:
            batch_smiles = []
            batch_nlls = []
            for i in range(batch_size):
                smiles, nlls = self._single_sample(shuffle=shuffle, return_all=return_all)
                batch_smiles.append(smiles)
                batch_nlls.append(nlls)
            # Reformat to be the same as batch_prompts
            if return_all:
                batch_smiles = list(map(list, zip(*batch_smiles)))
                batch_nlls = list(map(list, zip(*batch_nlls)))
            return batch_smiles, batch_nlls


class FragmentLinker(BaseSampler):
    def __init__(self, fragments: list, batch_size: int, sample_fn: Callable, evaluate_fn: Callable, batch_prompts: bool = False, shuffle=True, scan=False, return_all: bool = False, random_seed = 123):
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
                return smiles: list, nlls: Union[list, np.array, torch.tensor]
        evaluate_fn : Callable
            The evaluation function to use in the following format.
            evaluate_fn(smiles: Union[str, list]):
                return nlls: Union[list, np.array, torch.tensor]
        batch_prompts : bool, optional
            Whether the sample_fn can accept a list of prompts equal to batch_size, by default False
        shuffle : bool, optional
            Whether to shuffle the attachment points, by default True
        scan : bool, optional
            Whether to evaluate fragment attachment through the whole linker as oppose to simple concatenation, by default False, automatically enabled for more than two fragments 
        return_all : bool, optional
            Whether to return all intermediate samples, by default False
        random_seed : int, optional
            The random seed to use, by default 123
        
        Returns
        -------
        smiles : list
            A list of generated SMILES with a fragments linked.
        """
        super().__init__(sample_fn, evaluate_fn)
        self.batch_size = batch_size
        self.batch_prompts = batch_prompts
        self.shuffle = shuffle
        self.scan = scan
        self.sample_fn = sample_fn
        self.evaluate_fn = evaluate_fn
        self.return_all = return_all
        self.fragment = namedtuple('fragment', ['for_smiles', 'for_nll', 'rev_smiles', 'rev_nll'])
        self.seed = random_seed
        random.seed(self.seed)

        # Correct scan
        if len(fragments) > 2 and not self.scan:
            warnings.warn(f"Scan must be used for more than two fragments, Scan will be enabled.")
            self.scan = True

        # Prepare fragments
        self.n_fgs = len(fragments)
        self.fragments = []
        for frag in fragments:
            # Get attachment index
            aidx = utils.get_attachment_indexes(frag)
            assert len(aidx) == 1, f"Fragment {frag} should only have one attachment point"
            # Optimize forward direction
            for_smi, for_nll = self.optimize_prompt(frag, aidx[0], reverse=False)
            # Optimize reverse direction
            rev_smi, rev_nll = self.optimize_prompt(frag, aidx[0], reverse=True)
            # Append
            self.fragments.append(
                self.fragment(
                    for_smiles=utils.strip_attachment_points(for_smi)[0],
                    for_nll=for_nll,
                    rev_smiles=utils.strip_attachment_points(rev_smi)[0],
                    rev_nll=rev_nll
                    )
            )
        self.fragments.sort(key=lambda x: x.for_nll)

    def _single_sample(self, shuffle: bool = None, scan: bool = None, return_all: bool = None):
        # Set parameters
        if not shuffle: shuffle = self.shuffle
        if not scan: scan = self.scan
        if not return_all: return_all = self.return_all
        fragments = deepcopy(self.fragments)

        # Randomly select starting attachment points
        if shuffle: i = random.randint(0, self.n_fgs-1)
        else: i = 0
        f0 = fragments.pop(i)

        # Sample
        n_rem = self.n_fgs
        batch_smiles = []
        batch_nlls = []
        prompt = f0.rev_smiles
        prompt_tokens = self.tokenizer.tokenize(prompt, with_begin_and_end=False)
        frag_indexes = list(range(len(prompt_tokens)))
        smiles, nll = self.sample_fn(prompt=prompt, batch_size=1)
        smiles, nll = smiles[0], nll[0]
        smiles_tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=False)
        batch_smiles.append(smiles)
        batch_nlls.append(nll)
        n_rem -= 1
        
        if self.scan:
            while n_rem:
                # Select another fragment
                if shuffle: i = random.randint(0, n_rem-1)
                else: i = 0
                fi = fragments.pop(i)
                # Correct rings
                fi_smi = utils.correct_ring_numbers(smiles, fi.for_smiles)
                # Insert fragment at different positions
                temp_smiles = []
                for i in range(max(frag_indexes), len(smiles_tokens)):
                    if i in frag_indexes: 
                        continue 
                    if i == len(smiles_tokens)-1: 
                        tsmi = smiles + "(" + fi_smi + ")"
                    else: 
                        # NOTE operate in token space to minimize errors
                        tsmi = "".join(smiles_tokens[:i+1] + ["("] + [fi_smi] + [")"] + smiles_tokens[i+1:])
                    tidx = list(range(i+1, i+len(fi_smi)+2))
                    temp_smiles.append((tsmi, tidx))
                temp_nlls = self.evaluate_fn([smi for smi, _ in temp_smiles])
                # Select best
                (smiles, fidxs), nll = sorted(zip(temp_smiles, temp_nlls), key=lambda x: x[1])[0]
                batch_smiles.append(smiles)
                batch_nlls.append(nll)
                frag_indexes.extend(fidxs)
                n_rem -= 1
        else:
            fi = fragments.pop(0)
            # Correct rings
            fi_smi = utils.correct_ring_numbers(smiles, fi.for_smiles)
            # Append fragment
            smiles = smiles + fi_smi
            # Evaluate
            nll = self.evaluate_fn([smiles])[0]
            batch_smiles.append(smiles)
            batch_nlls.append(nll)
            n_rem -= 1

        if return_all:
            return batch_smiles, batch_nlls
        else:
            return batch_smiles[-1], batch_nlls[-1]

    def _batch_sample(self, batch_size: int = None, shuffle: bool = None, scan: bool = None, return_all: bool = None):
        """More efficient sampling assuming the sample_fn can accept a batch of prompts"""
        # Set parameters
        if not batch_size: batch_size = self.batch_size
        if not shuffle: shuffle = self.shuffle
        if not scan: scan = self.scan
        if not return_all: return_all = self.return_all
        batch_fragments = [deepcopy(self.fragments) for _ in range(batch_size)] # NOTE repeating with itertools or by *x leads to abberant behaviour with pop removing element from all sublists

        # Select initial fragments
        prompts = []
        frag_indexes = []
        for bi in range(batch_size):
            if shuffle: i = random.randint(0, self.n_fgs-1)
            else: i = 0
            f0 = batch_fragments[bi].pop(i)
            f0_tokens = self.tokenizer.tokenize(f0.rev_smiles, with_begin_and_end=False)
            prompts.append(f0.rev_smiles)
            frag_indexes.append(list(range(len(f0_tokens))))

        # Sample
        n_rem = self.n_fgs
        batch_smiles = []
        batch_nlls = []
        smiles, nlls = self.sample_fn(prompt=prompts, batch_size=batch_size)
        batch_smiles.append(smiles)
        batch_nlls.append(nlls)
        n_rem -= 1

        if self.scan:
            while n_rem:
                n_smiles = []
                n_nlls = []
                n_idxs = []
                for bi, (smiles, fragments, existing_indexes) in enumerate(zip(batch_smiles[-1], batch_fragments, frag_indexes)):
                    # Select another fragment
                    if shuffle: i = random.randint(0, n_rem-1)
                    else: i = 0
                    fi = fragments.pop(i)
                    # Correct rings
                    fi_smi = utils.correct_ring_numbers(smiles, fi.for_smiles)
                    # Insert fragment at different positions
                    smiles_tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=False)
                    temp_smiles = []
                    for i in range(max(existing_indexes), len(smiles_tokens)):
                        if i in existing_indexes: 
                            continue 
                        if i == len(smiles_tokens)-1: 
                            tsmi = smiles + "(" + fi_smi + ")"
                        else: 
                            tsmi = "".join(smiles_tokens[:i+1] + ["("] + [fi_smi] + [")"] + smiles_tokens[i+1:])
                        tidx = list(range(i+1, i+len(fi_smi)+2))
                        temp_smiles.append((tsmi, tidx))
                    # Evaluate
                    temp_nlls = self.evaluate_fn([smi for smi, _ in temp_smiles])
                    # Select best
                    (smiles, fidxs), nll = sorted(zip(temp_smiles, temp_nlls), key=lambda x: x[1])[0]
                    n_smiles.append(smiles)
                    n_nlls.append(nll)
                    frag_indexes[bi].extend(fidxs)
                batch_smiles.append(n_smiles)
                batch_nlls.append(n_nlls)
                n_rem -= 1
        else:
            concat_smiles = []
            for smiles, fragments in zip(batch_smiles[0], batch_fragments):
                fi = fragments.pop(0)
                # Correct rings
                fi_smi = utils.correct_ring_numbers(smiles, fi.for_smiles)
                # Append fragment
                smiles = smiles + fi_smi
                concat_smiles.append(smiles)
            # Evaluate
            batch_smiles.append(concat_smiles)
            batch_nlls.append(self.evaluate_fn(concat_smiles))
            n_rem -= 1

        if return_all:
            return batch_smiles, batch_nlls
        else:
            return batch_smiles[-1], batch_nlls[-1]

    def sample(self, batch_size: int = None, batch_prompts: bool = None, return_all: bool = None, shuffle: bool = None, scan: bool = None, **kwargs):
        """
        Sample de novo molecules, see init docstring for more details.
        """
        # Set parameters
        if not batch_size: batch_size = self.batch_size
        if not batch_prompts: batch_prompts = self.batch_prompts
        if not shuffle: shuffle = self.shuffle
        if not scan: scan = self.scan
        if not return_all: return_all = self.return_all
        # Correct scan
        if self.n_fgs > 2 and not self.scan:
            warnings.warn(f"Scan must be used for more than two fragments, Scan will be enabled.")
            self.scan = True

        # Choose _single_sample, iterative _single_sample, or _batch_sample
        if batch_prompts:
            return self._batch_sample(batch_size=batch_size, shuffle=shuffle, return_all=return_all)
        else:
            batch_smiles = []
            batch_nlls = []
            for i in range(batch_size):
                smiles, nlls = self._single_sample(shuffle=shuffle, return_all=return_all)
                batch_smiles.append(smiles)
                batch_nlls.append(nlls)
            # Reformat to be the same as batch_prompts
            if return_all:
                batch_smiles = list(map(list, zip(*batch_smiles)))
                batch_nlls = list(map(list, zip(*batch_nlls)))
            return batch_smiles, batch_nlls