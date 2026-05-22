#!/usr/bin/env python3

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from collections import defaultdict

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"

RNA_RESNAMES = {"A", "C", "G", "U", "I"}

RNA_BASE_ATOMS = {
    "N1",
    "C2",
    "O2",
    "N3",
    "C4",
    "N4",
    "O4",
    "C5",
    "C6",
    "N6",
    "N7",
    "C8",
    "N9",
    "O6",
    "N2",
}

HYDROGEN_PREFIXES = ("H", "D")


def download_cif_if_missing(pdb, cif_dir):
    cif_dir = Path(cif_dir)
    cif_dir.mkdir(exist_ok=True)

    path = cif_dir / f"{pdb}.cif.gz"

    if path.exists():
        return path

    url = RCSB_CIF_URL.format(pdb=pdb.lower())
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)

    return path


def split_mmcif_line(line):
    return line.strip().split()


def load_rna_resnames(mutate_list_path):
    rna_resnames = set(RNA_RESNAMES)

    with open(mutate_list_path) as f:
        for line in f:
            fields = line.split()
            if fields:
                rna_resnames.add(fields[0])

    return rna_resnames


def parse_rna_base_atoms_from_cif_gz(cif_path, rna_resnames):
    with gzip.open(cif_path, "rt", errors="replace") as f:
        lines = f.readlines()

    atoms = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        tags = []

        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1

        if not tags or not any(t.startswith("_atom_site.") for t in tags):
            continue

        idx = {tag: n for n, tag in enumerate(tags)}

        required = [
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]

        if not all(r in idx for r in required):
            continue

        model_idx = idx.get("_atom_site.pdbx_PDB_model_num")

        while i < len(lines):
            line = lines[i].strip()

            if (
                not line
                or line.startswith("#")
                or line == "loop_"
                or line.startswith("_")
            ):
                break

            cols = split_mmcif_line(line)

            if len(cols) >= len(tags):
                if model_idx is not None and cols[model_idx].strip('"') != "1":
                    i += 1
                    continue

                atom_name = cols[idx["_atom_site.label_atom_id"]].strip('"')
                resname = cols[idx["_atom_site.label_comp_id"]].strip('"')
                chain = cols[idx["_atom_site.auth_asym_id"]].strip('"')
                seq = cols[idx["_atom_site.auth_seq_id"]].strip('"')

                if (
                    resname in rna_resnames
                    and atom_name in RNA_BASE_ATOMS
                    and seq not in {".", "?"}
                    and not atom_name.startswith(HYDROGEN_PREFIXES)
                ):
                    try:
                        atoms.append(
                            {
                                "chain": chain,
                                "resnum": int(float(seq)),
                                "resname": resname,
                                "atom": atom_name,
                                "x": float(cols[idx["_atom_site.Cartn_x"]]),
                                "y": float(cols[idx["_atom_site.Cartn_y"]]),
                                "z": float(cols[idx["_atom_site.Cartn_z"]]),
                            }
                        )
                    except ValueError:
                        pass

            i += 1

    return atoms


def squared_distance(a, b):
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2


def detect_S_residues(base_atoms, cutoff=3.0, min_contacts=2):
    cutoff2 = cutoff * cutoff

    atoms_by_residue = defaultdict(list)

    for atom in base_atoms:
        atoms_by_residue[(atom["chain"], atom["resnum"])].append(atom)

    residues = sorted(atoms_by_residue.keys())
    output = defaultdict(list)

    for residue in residues:
        atoms1 = atoms_by_residue[residue]
        has_non_neighbor_contact = False
        chain, resnum = residue

        for other_residue in residues:
            if other_residue == residue:
                continue

            other_chain, other_resnum = other_residue
            if chain == other_chain and abs(resnum - other_resnum) == 1:
                continue

            atoms2 = atoms_by_residue[other_residue]
            contacting_atoms = set()

            for a1 in atoms1:
                for a2 in atoms2:
                    if squared_distance(a1, a2) < cutoff2:
                        contacting_atoms.add(a1["atom"])

            if len(contacting_atoms) >= min_contacts:
                has_non_neighbor_contact = True
                break

        if not has_non_neighbor_contact:
            output[f"chain_{chain}"].append(resnum)

    return {chain: sorted(resnums) for chain, resnums in output.items()}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect single-stranded RNA bases in PDB/mmCIF structures. "
            "A base is marked as single-stranded if it does not have at least "
            "min_contacts heavy atoms within cutoff Å of at least one other "
            "non-neighbor base."
        )
    )
    parser.add_argument("pdb_list", help="Text file containing one PDB code per line")
    parser.add_argument("mutate_list", help="List of mutated RNA residue names")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--cif-dir", default="mmcif")
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--min-contacts", type=int, default=2)

    args = parser.parse_args()

    pdbs = [line.split()[0] for line in open(args.pdb_list).readlines() if line.split()]
    rna_resnames = load_rna_resnames(args.mutate_list)

    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            result = json.load(f)
    else:
        result = {}

    for pdb in pdbs:
        print(f"Processing {pdb}")

        cif_path = download_cif_if_missing(pdb, args.cif_dir)
        base_atoms = parse_rna_base_atoms_from_cif_gz(cif_path, rna_resnames)

        result[pdb] = detect_S_residues(
            base_atoms,
            cutoff=args.cutoff,
            min_contacts=args.min_contacts,
        )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
