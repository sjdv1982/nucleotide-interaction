import argparse

import numpy as np
from seamless import Buffer
from tqdm import tqdm


RNA_BASE_ATOMS = {
    b"N1",
    b"C2",
    b"O2",
    b"N3",
    b"C4",
    b"N4",
    b"O4",
    b"C5",
    b"C6",
    b"N6",
    b"N7",
    b"C8",
    b"N9",
    b"O6",
    b"N2",
}


def get_coor(struc):
    return np.stack((struc["x"], struc["y"], struc["z"]), axis=-1)


def is_same_chain_neighbor(residue1, residue2):
    chain1, resnum1 = residue1
    chain2, resnum2 = residue2
    return chain1 == chain2 and abs(int(resnum1) - int(resnum2)) == 1


def detect_single_base_residues(rna_strucs, cutoff=3.0, min_contacts=2):
    from scipy.spatial import KDTree

    base_coords = []
    base_atom_names = []
    base_residues = []

    for rna_code, rna_struc in rna_strucs:
        base_mask = np.isin(rna_struc["name"], list(RNA_BASE_ATOMS))
        base_struc = rna_struc[base_mask]
        base_coords.extend(get_coor(base_struc))
        base_atom_names.extend(base_struc["name"])
        base_residues.extend((rna_code, int(resid)) for resid in base_struc["resid"])

    residues = sorted(set(base_residues))
    if not residues:
        return set()

    tree = KDTree(np.asarray(base_coords))
    contacting_atoms = {}

    for atom1, atom2 in tree.query_pairs(cutoff):
        residue1 = base_residues[atom1]
        residue2 = base_residues[atom2]

        if residue1 == residue2 or is_same_chain_neighbor(residue1, residue2):
            continue

        contacting_atoms.setdefault((residue1, residue2), set()).add(
            base_atom_names[atom1]
        )
        contacting_atoms.setdefault((residue2, residue1), set()).add(
            base_atom_names[atom2]
        )

    paired_residues = set()
    for residue1, residue2 in contacting_atoms:
        if len(contacting_atoms[(residue1, residue2)]) >= min_contacts:
            paired_residues.add(residue1)

    return set(residues) - paired_residues


def iter_residues(rna_code, rna_struc):
    previous = None
    for resid in rna_struc["resid"]:
        residue = (rna_code, int(resid))
        if residue != previous:
            yield residue
            previous = residue


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect single-stranded RNA bases in all-PDB reduced RNA data. "
            "A base is marked as single-stranded if it does not have at least "
            "min_contacts heavy atoms within cutoff Å of at least one other "
            "non-neighbor base."
        )
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=3.0,
        help="Base-heavy-atom contact distance cutoff in Å.",
    )
    parser.add_argument(
        "--min-contacts",
        type=int,
        default=2,
        help="Minimum number of distinct base atoms needed to mark a residue paired.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("load RNA")
    rna_struc_index, rna_strucs_data = Buffer.load(
        "input/allpdb-rna-aareduce.mixed"
    ).get_value("mixed")

    print("index RNA residues")
    residue_index = {}
    for rna_code, (r_start, r_size) in rna_struc_index.items():
        rna_struc = rna_strucs_data[r_start : r_start + r_size]
        for residue in iter_residues(rna_code, rna_struc):
            residue_index[residue] = len(residue_index)

    single_base_mask = np.zeros(len(residue_index), bool)

    print("group RNA by PDB")
    pdb_rna_codes = {}
    for rna_code in rna_struc_index:
        pdb_rna_codes.setdefault(rna_code[:4], []).append(rna_code)

    print("detect single bases")
    for pdb, rna_codes in tqdm(pdb_rna_codes.items()):
        rna_strucs = []
        for rna_code in rna_codes:
            r_start, r_size = rna_struc_index[rna_code]
            rna_strucs.append((rna_code, rna_strucs_data[r_start : r_start + r_size]))

        for residue in detect_single_base_residues(
            rna_strucs,
            cutoff=args.cutoff,
            min_contacts=args.min_contacts,
        ):
            single_base_mask[residue_index[residue]] = True

    np.save("allpdb-detect-single-bases.npy", single_base_mask)


if __name__ == "__main__":
    main()
