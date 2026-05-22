#!/usr/bin/env python3

# Find Pfam domains in contact with at least "min_nt" consecutive ssRNA
# nucleotides, using precomputed all-PDB per-nucleotide files.

import argparse
from collections import defaultdict
from itertools import zip_longest
import json

import numpy as np


def parse_nuc_line(line):
    pdb, chain, resid = line.split()
    return pdb, chain, int(resid)


def iter_aligned_rows(nuc_path, single_base_path, pfam_contacts_path):
    single_base_mask = np.load(single_base_path)

    with open(nuc_path) as nuc_file, open(pfam_contacts_path) as pfam_file:
        for index, (nuc_line, pfam_line) in enumerate(
            zip_longest(nuc_file, pfam_file)
        ):
            if nuc_line is None:
                raise ValueError(f"{pfam_contacts_path} has more rows than {nuc_path}")
            if pfam_line is None:
                raise ValueError(f"{nuc_path} has more rows than {pfam_contacts_path}")
            if index >= len(single_base_mask):
                raise ValueError(
                    f"{single_base_path} has fewer rows than {nuc_path} "
                    f"and {pfam_contacts_path}"
                )

            pdb, chain, resid = parse_nuc_line(nuc_line)
            pfams = pfam_line.split()
            yield pdb, chain, resid, bool(single_base_mask[index]), pfams
    row_count = index + 1 if "index" in locals() else 0
    if row_count != len(single_base_mask):
        raise ValueError(
            f"{single_base_path} has {len(single_base_mask)} rows, "
            f"but the text files have {row_count}"
        )


def get_consecutive_runs(resnums, min_nt):
    runs = []
    current = []

    for resid in sorted(set(resnums)):
        if current and resid == current[-1] + 1:
            current.append(resid)
        else:
            if len(current) >= min_nt:
                runs.append(current)
            current = [resid]

    if len(current) >= min_nt:
        runs.append(current)

    return runs


def find_pfam_contacts_with_ssrna_stretches(
    nuc_path,
    single_base_path,
    pfam_contacts_path,
    min_nt,
):
    contacted_ss_residues = defaultdict(list)

    for pdb, chain, resid, is_single_base, pfams in iter_aligned_rows(
        nuc_path,
        single_base_path,
        pfam_contacts_path,
    ):
        if not is_single_base:
            continue
        chain_key = f"chain_{chain}"
        for pfam in pfams:
            contacted_ss_residues[(pdb, pfam, chain_key)].append(resid)

    output_data = defaultdict(lambda: defaultdict(list))

    for (pdb, pfam, chain_key), resids in contacted_ss_residues.items():
        for run in get_consecutive_runs(resids, min_nt=min_nt):
            output_data[pdb][pfam].append({chain_key: [f"res_{x}" for x in run]})

    return {
        pdb: {pfam: runs for pfam, runs in sorted(pfam_runs.items())}
        for pdb, pfam_runs in sorted(output_data.items())
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find Pfam domains in contact with consecutive single-stranded RNA "
            "nucleotides from precomputed all-PDB per-nucleotide files."
        )
    )
    parser.add_argument("-o", "--output", required=True, help="pfam_interf_ssRNA.json")
    parser.add_argument(
        "--nucleotides",
        default="allpdb-print-nuc-auth.txt",
        help=(
            "Space-separated PDB/auth-chain/resid file produced by "
            "allpdb-print-nuc.py."
        ),
    )
    parser.add_argument(
        "--single-bases",
        default="allpdb-detect-single-bases.npy",
        help="Boolean per-residue mask produced by allpdb-detect-single-bases.py.",
    )
    parser.add_argument(
        "--pfam-contacts",
        default="allpdb-pfam-contacts.txt",
        help="Per-residue Pfam contact file produced by allpdb-pfam-contacts.py.",
    )
    parser.add_argument("--min_nt", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_data = find_pfam_contacts_with_ssrna_stretches(
        nuc_path=args.nucleotides,
        single_base_path=args.single_bases,
        pfam_contacts_path=args.pfam_contacts,
        min_nt=args.min_nt,
    )

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
