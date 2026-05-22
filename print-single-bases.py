#!/usr/bin/env python3

import argparse

import numpy as np
from seamless import Buffer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print residue numbers marked positive in allpdb-detect-single-bases.npy "
            "for one PDB code and RNA chain."
        )
    )
    parser.add_argument("pdb", help="PDB code, upper or lower case")
    parser.add_argument("chain", help="RNA chain ID")
    parser.add_argument(
        "--mask",
        default="allpdb-detect-single-bases.npy",
        help="Boolean per-residue mask produced by allpdb-detect-single-bases.py.",
    )
    return parser.parse_args()


def iter_residues(rna_struc):
    previous = None
    for resid in rna_struc["resid"]:
        resid = int(resid)
        if resid != previous:
            yield resid
            previous = resid


def main():
    args = parse_args()
    pdb = args.pdb[:4].lower()
    chain = args.chain
    rna_code = pdb + chain

    rna_struc_index, rna_strucs_data = Buffer.load(
        "input/allpdb-rna-aareduce.mixed"
    ).get_value("mixed")
    if rna_code not in rna_struc_index:
        available_chains = sorted(
            code[4:] for code in rna_struc_index if code.startswith(pdb)
        )
        if available_chains:
            raise SystemExit(
                f"RNA chain not found: {rna_code}\n"
                f"PDB code exists: {pdb}\n"
                f"Available RNA chains: {', '.join(available_chains)}"
            )
        raise SystemExit(
            f"RNA chain not found: {rna_code}\n"
            f"PDB code exists: no\n"
            "Available RNA chains: none"
        )

    mask = np.load(args.mask)

    offset = 0
    target_residues = None
    for code, (start, size) in rna_struc_index.items():
        rna_struc = rna_strucs_data[start : start + size]
        residues = list(iter_residues(rna_struc))
        next_offset = offset + len(residues)

        if code == rna_code:
            target_residues = residues
            target_mask = mask[offset:next_offset]
            break

        offset = next_offset

    if target_residues is None:
        raise SystemExit(f"RNA chain not found: {rna_code}")

    if len(target_mask) != len(target_residues):
        raise SystemExit(
            f"Mask length mismatch for {rna_code}: "
            f"{len(target_mask)} mask values, {len(target_residues)} residues"
        )

    positives = [
        resid for resid, is_single_base in zip(target_residues, target_mask) if is_single_base
    ]
    for resid in positives:
        print(resid)


if __name__ == "__main__":
    main()
