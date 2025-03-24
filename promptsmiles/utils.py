# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)

import copy
import logging
import random
import re
from collections import defaultdict

logger = logging.getLogger("promptsmiles")

from rdkit import Chem, RDLogger
from rdkit.Chem import rdqueries

# TODO Canonicalize to ensure ring atoms have the ring number afterwards (before any branches or )

# REGEX
SQUARE_BRACKET = re.compile(r"(\[[^\]]*\])")
SQUARE_BRACKET_noH = re.compile(r"(\[(?:(?!H)[^\]]*)\])")
BRCL = re.compile(r"(Br|Cl)")
ATOM = re.compile(r"([a-zA-Z])")
# ---- RING_ATOM needs some explaining ----
# An atom followed by ring connections (0-9) or double ring connections (%0-9)
# The atom may also have an explicit bond to the ring connection -/=/#
# The atom may also be wrapped in square brackets []
# The atom should not be *within* square brackets, e.g., "[NH2+]2" is a ring atom but not the "H2" inside it
RING_ATOM = re.compile(
    r"([a-zA-Z][%0-9]+(?![^[]*\])|[a-zA-Z][-=#:][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9]+(?![^[]*\])|\[[^\]]*\][-=#:][%0-9]+(?![^[]*\]))"
)
# The following regex identifies ring numbers seperated by an explicit bond e.g., C2-3, whereas above only recognizes C-2
RING_ATOM_2 = re.compile(
    r"([a-zA-Z][%0-9][-=#][%0-9]+(?![^[]*\])|[a-zA-Z][%0-9]+(?![^[]*\])|[a-zA-Z][-=#][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9][-=#][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9]+(?![^[]*\])|\[[^\]]*\][-=#][%0-9]+(?![^[]*\]))"
)
SINGLE_RING = re.compile(r"([0-9]{1})")
DOUBLE_RING = re.compile(r"(%[0-9]{2})")
BR_ATTCH = re.compile(r"(\(\*\))")
ATTCH = re.compile(r"(\*(?![^[]*\]))") # Not * in square brackets
SUB_ATTCH = re.compile(r"(\[99\*\])")
BR_OPEN = re.compile(r"(\()")
BR_CLOSE = re.compile(r"(\))")
BR = re.compile(r"(\(|\))")


def disable_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")


def disable_rdkit_logging_dec(func):
    def wrapper(*args, **kwargs):
        RDLogger.DisableLog("rdApp.*")
        out = func(*args, **kwargs)
        RDLogger.EnableLog("rdApp.*")
        return out

    return wrapper


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SMILES"
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
        "atom": re.compile(r"[a-zA-Z]"),
        "ring_atom": re.compile(
            r"([a-zA-Z][%0-9]+(?![^[]*\])|[a-zA-Z][-=#][%0-9]+(?![^[]*\])|\[[^\]]*\][%0-9]+(?![^[]*\])|\[[^\]]*\][-=#][%0-9]+(?![^[]*\]))"
        ),
    }
    REGEXP_ORDER = [
        "ring_atom",
        "brackets",
        "brcl",
    ]  # ["brackets", "2_ring_nums", "brcl"]

    def __init__(self):
        self.GRAMMAR = copy.deepcopy(self.GRAMMAR)
        self.REGEXPS = copy.deepcopy(self.REGEXPS)
        self.REGEXP_ORDER = copy.deepcopy(self.REGEXP_ORDER)

    def _token2atom_map(self, tokens):
        """Given a list of tokens that tokenizes at least by atom e.g., Br / Cl / [NH2+]"""
        atom_map = {}
        atom_counter = 0
        for i, t in enumerate(tokens):
            if any(
                [
                    regex.fullmatch(t)
                    for regex in [
                        self.REGEXPS["ring_atom"],
                        self.REGEXPS["brackets"],
                        self.REGEXPS["brcl"],
                        self.REGEXPS["atom"],
                    ]
                ]
            ):
                atom_map[atom_counter] = i
                atom_counter += 1
        return atom_map

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens, **kwargs):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def split_by_regex(data: str, regexes: list):
    if not regexes:
        return list(data)
    regexp = regexes[0]
    splitted = regexp.split(data)
    tokens = []
    for i, split in enumerate(splitted):
        if i % 2 == 0:
            tokens += split_by_regex(split, regexes[1:])
        else:
            tokens.append(split)
    return tokens


def int2ring_number(x):
    if x < 10:
        return str(x)
    else:
        return "%" + str(x)


def root_smiles(smi, rootAtom=None, reverse=False):
    """Rearrange SMILES to start at rootAtom"""
    # Convert leading wildcard out of parenthesis if presented that way
    if smi.startswith("(*)"):
        smi = re.sub(r"\(\*\)", "*", smi, count=1)

    smi = xsmiles2smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    new_smi = None
    if mol:
        new_smi = Chem.MolToSmiles(mol, rootedAtAtom=rootAtom)
        if reverse:
            # NOTE sometimes RDKit assigns the same ring index to different rings, causing an error upon reversing, so let's re-index if necessary
            try:
                
                new_smi = _check_ring_numbers(new_smi)
            except Exception as e:
                logger.error(e)
                pass
            new_smi = reverse_smiles(new_smi)
        # Convert back to (*)
        new_smi = smiles2xsmiles(new_smi)
        new_smi = bracket_attachments(new_smi)
    return new_smi


def randomize_smiles(
    smi, n_rand=10, random_type="restricted", rootAtom=None, reverse=False
):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smi: A SMILES string
    :param n_rand: Number of randomized smiles per molecule
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :param rootAtom: Root smiles generation to begin with this atom, -1 denotes the last atom)
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    assert random_type in [
        "restricted",
        "unrestricted",
    ], f"Type {random_type} is not valid"

    # Convert leading wildcard out of parenthesis if presented that way
    if smi.startswith("(*)"):
        smi = re.sub(r"\(\*\)", "*", smi, count=1)
    
    smi = xsmiles2smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    if random_type == "unrestricted":
        rand_smiles = []
        for i in range(n_rand):
            if rootAtom is not None:
                if rootAtom == -1:
                    rootAtom = mol.GetNumAtoms() - 1
                random_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=False,
                    doRandom=True,
                    isomericSmiles=False,
                    rootedAtAtom=rootAtom,
                )
            else:
                random_smiles = Chem.MolToSmiles(
                    mol, canonical=False, doRandom=True, isomericSmiles=False
                )

            if reverse:
                assert (
                    "*" not in smi
                ), "Unexpected behaviour when smiles contain a wildcard character (*), please use restricted randomization"
                random_smiles = reverse_smiles(random_smiles)

            # Convert back to (*)
            random_smiles = smiles2xsmiles(random_smiles)
            random_smiles = bracket_attachments(random_smiles)

            rand_smiles.append(random_smiles)

        return list(set(rand_smiles))

    if random_type == "restricted":
        rand_smiles = []
        i = 0
        attempts = 0
        while (i < n_rand) and (attempts < 50):
            attempts += 1
            if rootAtom is not None:
                new_atom_order = list(range(mol.GetNumAtoms()))
                root_atom = new_atom_order.pop(rootAtom)  # -1
                random.shuffle(new_atom_order)
                new_atom_order = [root_atom] + new_atom_order
                random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
                random_smiles = Chem.MolToSmiles(
                    random_mol, canonical=False, isomericSmiles=True
                )
            else:
                new_atom_order = list(range(mol.GetNumAtoms()))
                random.shuffle(new_atom_order)
                random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
                random_smiles = Chem.MolToSmiles(
                    random_mol, canonical=False, isomericSmiles=True
                )

            if reverse:
                # NOTE sometimes RDKit assigns the same ring index to different rings, causing an error upon reversing, so let's re-index if necessary
                try:
                    random_smiles = _check_ring_numbers(random_smiles)
                except Exception as e:
                    logger.error(e)
                    pass
                random_smiles = reverse_smiles(random_smiles)

            # Convert back to (*)
            random_smiles = smiles2xsmiles(random_smiles)
            random_smiles = bracket_attachments(random_smiles)

            rand_smiles.append(random_smiles)
            i += 1

        return list(set(rand_smiles))


def reverse_smiles(smiles, renumber_rings=True, v=False):
    """
    Reverse a SMILES string
    """
    if v:
        print(f"Reversing: {smiles}")

    # Tokenize
    tokens = split_by_regex(smiles, [RING_ATOM_2, SQUARE_BRACKET, BRCL, ATTCH, ATOM])
    if v:
        print(f"Tokenized:\n\t{tokens}")

    # Find parenthesis
    branching_idxs = _seek_parenthesis(tokens)
    if v:
        print(f"Branches identified:\n\t{branching_idxs}")

    # Merge branches with source atom
    new_tokens = []
    i = 0
    while i < len(tokens):
        # Check if we need to combine branches into one token
        if i + 1 in branching_idxs:
            t = "".join(tokens[i : branching_idxs[i + 1] + 1])
            new_tokens.append(t)
            i = branching_idxs[i + 1] + 1
        else:
            new_tokens.append(tokens[i])
            i += 1
    if v:
        print(f"Tokens corrected by branch:\n\t{new_tokens}")

    # Reverse
    rev_tokens = list(reversed(new_tokens))
    rsmiles = "".join(rev_tokens)
    if v:
        print(f"Tokens reversed:\n\t{rev_tokens}")
    if v:
        print(f"SMILES reversed: {rsmiles}")

    # Re-number rings
    if renumber_rings:
        rsmiles = _reverse_ring_numbers(rsmiles)
        if v:
            print(f"Rings reindexed:\n\t {rsmiles}")

    return rsmiles


def _reverse_ring_numbers(smi: str) -> str:
    """Given ring numbers in smi, reindex the rings in smi from left to right"""
    ring_map = {}
    ring_count = 1
    new_smiles = []
    for c in split_by_regex(smi, [SQUARE_BRACKET, BRCL, DOUBLE_RING, SINGLE_RING]):
        if SINGLE_RING.fullmatch(c) or DOUBLE_RING.fullmatch(c):
            # Add new ring to map
            if c not in ring_map.keys():
                ring_map[c] = int2ring_number(ring_count)
                ring_count += 1
            # Update ring close
            c = ring_map[c]
        # Add token
        new_smiles.append(c)
    smiles = "".join(new_smiles)
    return smiles


def _check_ring_numbers(smiles, debug=False, v=False):
    """Check and re-index ring numbers sequentially if needed"""
    smiles = xsmiles2smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    ringinfo = mol.GetRingInfo()
    N_rings = ringinfo.NumRings()
    L_rings = [
        int(t.strip("%"))
        for t in DOUBLE_RING.findall(smiles) + SINGLE_RING.findall(smiles)
    ]
    if L_rings:
        L_rings = max(L_rings)

    # ---- Check rings
    if (not L_rings) or (L_rings == N_rings):
        # Assume they're labelled correctly
        return smiles

    # ---- Otherwise let's re-index
    rings = list(ringinfo.AtomRings())
    if v:
        print(rings)
    tokens = split_by_regex(smiles, [SQUARE_BRACKET, DOUBLE_RING, BRCL])

    ring_count = 0
    atom_count = -1  # Counting what the last atom was
    ring_map = {}  # {old_ring -> new_ring}
    new_tokens = []
    for i, t in enumerate(tokens):
        # If it's a ring token -> update
        if any([regex.fullmatch(t) for regex in [DOUBLE_RING, SINGLE_RING]]):
            if debug:
                import pdb

                pdb.set_trace()
            # Check to see if it's already mapped
            if t in ring_map.keys():
                new_tokens.append(ring_map[t])
                if v:
                    print(f"Relabelled {t} -> {ring_map[t]} at atom {atom_count}")
                ring_map.pop(t)
            else:
                # Add it
                ring_count += 1
                new_ring_id = int2ring_number(ring_count)
                ring_map[t] = new_ring_id
                new_tokens.append(new_ring_id)
                if v:
                    print(f"Relabelled {t} -> {new_ring_id} at atom {atom_count}")
        # If it's an atom -> count
        elif any(
            [
                regex.fullmatch(t)
                for regex in [SQUARE_BRACKET_noH, BRCL, ATOM, BR_ATTCH, ATTCH]
            ]
        ):
            atom_count += 1
            new_tokens.append(t)
        else:
            new_tokens.append(t)
    if v:
        print(ring_map)
    new_smiles = "".join(new_tokens)
    new_smiles = smiles2xsmiles(new_smiles)
    logger.debug(f"Re-indexed SMILES rings from {smiles} -> {new_smiles}")
    return new_smiles


def _seek_parenthesis(smiles_or_tokens):
    """
    Return the indices of top level parenthesis only as a dict map
    """
    br_open = 0
    multiple = False
    open_close_idxs = {}  # {idx1: idx2}
    for i, t in enumerate(smiles_or_tokens):
        # Open
        if BR_OPEN.fullmatch(t):
            # If first open, save index
            if (br_open == 0) and not multiple:
                br_open_idx = i
            br_open += 1
            multiple = False
        # Close
        if BR_CLOSE.fullmatch(t):
            br_open -= 1
            # If last close, save index
            if br_open == 0:
                # Check to see we aren't immediately branching afterwards
                if (i < len(smiles_or_tokens) - 1) and BR_OPEN.fullmatch(
                    smiles_or_tokens[i + 1]
                ):
                    multiple = True
                else:
                    br_close_idx = i
                    open_close_idxs[br_open_idx] = br_close_idx
    return open_close_idxs


def _seek_source_atom(smiles_or_tokens, idx):
    found = False
    br_open = 0
    while not found:
        idx = idx - 1
        t = smiles_or_tokens[idx]
        # NOTE opposite because in reverse
        if BR_CLOSE.fullmatch(t):
            br_open += 1
        if BR_OPEN.fullmatch(t):
            br_open += -1
        # If there's no branches open and we're not opening another, or it's another attch point
        if (br_open == 0) and not BR.fullmatch(t) and not (BR_ATTCH.fullmatch(t) or ATTCH.fullmatch(t)):
            found = True
    return idx


def get_attachment_points(smi: str, return_map: bool = False) -> list:
    tokens = split_by_regex(
        smi, [RING_ATOM, SQUARE_BRACKET, BRCL, BR_ATTCH, ATTCH, ATOM]
    )
    atom_counter = 0
    all_counter = 0
    token2atom_map = {}
    attch2dummy_map = defaultdict(list)
    for ti, t in enumerate(tokens):
        if ATTCH.fullmatch(t) or BR_ATTCH.fullmatch(t):
            # If it's the first atom
            if ti == 0:
                # Seek next atom...
                attch2dummy_map[atom_counter].append(all_counter)
            # NOTE correcting for preceeding branches i.e., see previous atom...
            elif (ti > 0) and BR_CLOSE.fullmatch(tokens[ti - 1]):
                source_ti = _seek_source_atom(tokens, ti)
                attch2dummy_map[token2atom_map[source_ti]].append(all_counter)
            # Otherwise it's the previous atom
            else:
                attch2dummy_map[atom_counter - 1].append(all_counter)
            all_counter += 1

        if any(
            [
                regex.fullmatch(t)
                for regex in [RING_ATOM, SQUARE_BRACKET_noH, BRCL, ATOM]
            ]
        ):
            token2atom_map[ti] = atom_counter
            atom_counter += 1
            all_counter += 1

    if return_map:
        return attch2dummy_map
    else:
        return [k for k in attch2dummy_map for _ in attch2dummy_map[k]]


def insert_attachment_points(smi: str, at_pts: list):
    tokens = split_by_regex(
        smi, [RING_ATOM, SQUARE_BRACKET, BRCL, BR_ATTCH, ATTCH, ATOM]
    )
    atom_counter = 0
    new_tokens = []
    for t in tokens:
        new_tokens.append(t)
        if any(
            [
                regex.fullmatch(t)
                for regex in [RING_ATOM, SQUARE_BRACKET_noH, BRCL, ATOM]
            ]
        ):
            if atom_counter in at_pts:
                for _ in range(sum([at_pt == atom_counter for at_pt in at_pts])):
                    new_tokens.append("(*)")
            atom_counter += 1
    smi = "".join(new_tokens)
    return smi


def correct_attachment_point(smi: str, at_pt: int) -> int:
    """Switch attachment point index to wildcard index in atom"""
    attch2dummy_map = get_attachment_points(smi, return_map=True)
    return attch2dummy_map[at_pt][0]


def strip_attachment_points(smi: str):
    """
    Remove * and provide canonical SMILES
    :param smi: SMILES with (*)
    :return: SMILES without (*), Atom index of attachment points
    """
    at_pts = get_attachment_points(smi)
    smi = smi.replace("(*)", "").replace("*", "")
    return smi, at_pts


def bracket_attachments(smi):
    """
    Convert all * to ensure they are branching points (*)
    """
    # Replace start
    nsmi = re.sub(r"^(\*)([a-zA-Z][0-9]?)", "\\2(\\1)", smi)
    # Replace end
    nsmi = re.sub(r"(\*)$", "(\\1)", nsmi)
    # Replace un-isolated
    nsmi = re.sub(r"([^(])(\*)", "\\1(\\2)", nsmi)
    nsmi = re.sub(r"(\*)([^)])", "(\\1)\\2", nsmi)
    return nsmi


# ----- Useful functions for fragment linking specifically -----


def extract_linker(smiles, fragments=[], return_all=False):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return None

    # Sort frags by largest SMILES first
    fragments = sorted(fragments, key=lambda x: len(x), reverse=True)

    for frag in fragments:
        # Get frag
        sfrag, _ = strip_attachment_points(frag)  ###
        # Remove explicit Hs as no substructure match otherwise (RDKit seems to have a bug, doesn't work)
        sfrag = re.sub(r"\[([a-zA-Z])H\]", "\\1", sfrag)
        fmol = Chem.MolFromSmiles(sfrag)
        # Get attachment point
        for match in mol.GetSubstructMatches(fmol):
            fragment_point = set()
            attachment_points = set()
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                neighbour_atoms = atom.GetNeighbors()
                for natom in neighbour_atoms:
                    nidx = natom.GetIdx()
                    if nidx not in match:
                        fragment_point.add(idx)
                        attachment_points.add(nidx)
            if len(fragment_point) == 1:
                break

        # An end fragment should have exactly one fragment atom attached
        # This may cause an error if the RNN has added the exact same fragment in the middle of the linker
        if not mol.HasSubstructMatch(fmol) or (len(fragment_point) != 1):
            return None

        # Add attachment points
        mol = Chem.RWMol(mol)
        fp = fragment_point.pop()
        mol.AddAtom(Chem.AtomFromSmiles("*"))
        for ap in attachment_points:
            # Get bond type first
            fatom_bonds = mol.GetAtomWithIdx(fp).GetBonds()
            fl_bond_type = [
                bond.GetBondType()
                for bond in fatom_bonds
                if (bond.GetBeginAtomIdx() == ap) or (bond.GetEndAtomIdx() == ap)
            ][0]
            # Add bond
            mol.AddBond(ap, mol.GetNumAtoms() - 1, fl_bond_type)
        # Delete substructure match
        mol.BeginBatchEdit()
        for aid in match:
            mol.RemoveAtom(aid)
        mol.CommitBatchEdit()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    linker = Chem.MolToSmiles(mol)
    if "." in linker:
        linker = sorted(linker.split("."), key=lambda x: x.count("*"))[-1]

    return linker


def _smiles2smarts(smiles):
    """Convert a SMILES sequence to a more specific SMARTS query including degree, aromaticity and ring membership"""
    smiles = smiles.replace("(*)", "*")
    mol = Chem.MolFromSmiles(smiles)
    mol2 = Chem.RWMol(mol)
    for i, at in enumerate(mol.GetAtoms()):
        if at.GetSymbol() == "*":
            continue
        num = at.GetAtomicNum()
        degree = at.GetDegree()
        aromatic = at.GetIsAromatic()
        ring = at.IsInRing()
        q = rdqueries.AtomNumEqualsQueryAtom(num)
        q.ExpandQuery(rdqueries.ExplicitDegreeEqualsQueryAtom(degree))
        if aromatic:
            q.ExpandQuery(rdqueries.IsAromaticQueryAtom())
        if ring:
            q.ExpandQuery(rdqueries.IsInRingQueryAtom())
        mol2.ReplaceAtom(i, q)
    smarts = Chem.MolToSmarts(mol2)
    smarts = smarts.replace("[#0]", "[*]")
    return smarts


def correct_fragment_ring_numbers(smi1: str, smi2: str) -> str:
    """Given the rings in smi1, reindex the rings in smi2"""
    # Count rings in smi1
    ring_count = 0
    for c in split_by_regex(smi1, [SQUARE_BRACKET, BRCL, DOUBLE_RING, SINGLE_RING]):
        # Check for number
        if SINGLE_RING.fullmatch(c) or DOUBLE_RING.fullmatch(c):
            # Count max ring index
            ring_count = max(ring_count, int(c.strip("%")))

    # Reindex smi2
    ring_map = {}
    ring_count += 1  # Start from next index
    new_smi2 = []
    for c in split_by_regex(smi2, [SQUARE_BRACKET, BRCL, DOUBLE_RING, SINGLE_RING]):
        # Check for number
        if SINGLE_RING.fullmatch(c) or DOUBLE_RING.fullmatch(c):
            # Add new ring to map
            if c not in ring_map.keys():
                ring_map[c] = int2ring_number(ring_count)
                ring_count += 1
            # Update c
            c = ring_map[c]
        # Add token
        new_smi2.append(c)
    return "".join(new_smi2)


def detect_existing_fragment(smiles, frag_smiles):
    """Get substructure match for frag smiles (with attachment index first)"""
    frag_indexes = []
    mol = Chem.MolFromSmiles(smiles)
    # Fragment must start with attachment point
    if not frag_smiles.startswith("(*)"):
        frag_smiles = "(*)" + frag_smiles
    frag_smarts = _smiles2smarts(frag_smiles)
    frag_patt = Chem.MolFromSmarts(frag_smarts)
    if mol:
        if mol.HasSubstructMatch(frag_patt):
            frag_indexes = list(mol.GetSubstructMatches(frag_patt))
    # NOTE that now we ignore the first atom i.e., the attachment point
    if frag_indexes:
        frag_indexes = [list(match)[1:] for match in frag_indexes]
    return frag_indexes


# ----- Superstructure utility -----
def superstructure_smiles(smiles: str) -> str:
    """
    Adds a dummy atom (*) to every heavy atom in the molecule with available valence.
    
    :param smiles: Input SMILES string.
    :return: Modified SMILES string with dummy atoms added.
    """
    smiles = xsmiles2smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.RWMol(mol)
    dummy = Chem.Atom(0)
    at_pts = []
    for ai in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(ai)
        if atom.GetAtomicNum() > 1:
            for _ in range(atom.GetNumImplicitHs()):
                at_pts.append(ai)

    mol.BeginBatchEdit()
    for ai in at_pts: 
        dummy_idx = mol.AddAtom(dummy)
        mol.AddBond(ai, dummy_idx, Chem.BondType.SINGLE)
    
    Chem.SanitizeMol(mol)
    super_smiles = Chem.MolToSmiles(mol)
    super_smiles = smiles2xsmiles(super_smiles)
    super_smiles = bracket_attachments(super_smiles)
    
    assert ATTCH.search(super_smiles) or BR_ATTCH.search(super_smiles), f"No attachment points found for superstructure SMILES: {super_smiles}"
    
    return super_smiles


# ----- Useful functions for testing -----
def smiles_eq(smi1, smi2):
    smi1 = xsmiles2smiles(smi1)
    smi2 = xsmiles2smiles(smi2)
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    # Parse them
    if not mol1:
        return False, f"Parsing error: {smi1}"
    if not mol2:
        return False, f"Parsing error: {smi2}"
    # Remove atom map
    for mol in [mol1, mol2]:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    # Check smiles are the same
    nsmi1 = Chem.MolToSmiles(mol1)
    nsmi2 = Chem.MolToSmiles(mol2)
    if nsmi1 != nsmi2:
        return False, f"Inequivalent SMILES: {nsmi1} vs {nsmi2}"
    # Check InChi
    inchi1 = Chem.MolToInchi(mol1)
    inchi2 = Chem.MolToInchi(mol2)
    if inchi1 != inchi2:
        return False, "Inequivalent InChi's"
    return True, ""


def mol_eq(mol1, mol2):
    mols = []
    # Remove atom map
    for mol in [mol1, mol2]:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mols.append(mol)
    # Check InChi
    inchi1 = Chem.MolToInchi(mols[0])
    inchi2 = Chem.MolToInchi(mols[1])
    if inchi1 != inchi2:
        return False, "Inequivalent InChi's"
    return True, ""


def xsmiles2smiles(smiles):
    smiles = smiles.replace("X", "[99*]")
    # NOTE: Remove explicit aromatic bonds, likely not in many vocabularies
    aromatic_bond = r"(\:(?![^[]*\]))"
    smiles = re.sub(aromatic_bond, "", smiles)
    return smiles

def smiles2xsmiles(smiles):
    return smiles.replace("[99*]", "X")
