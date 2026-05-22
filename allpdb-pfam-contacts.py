import csv

import numpy as np
from seamless import Buffer
from tqdm import tqdm
from scipy.spatial import KDTree


def get_coor(struc):
    return np.stack((struc["x"], struc["y"], struc["z"]), axis=-1)


def contacting_protein_atoms(rna_struc, prot_struc, distance=5):
    """For each RNA atom, the list of protein-atom indices within `distance`."""
    rna_tree = KDTree(get_coor(rna_struc))
    prot_tree = KDTree(get_coor(prot_struc))
    return rna_tree.query_ball_tree(prot_tree, distance)


def iter_residues(rna_code, rna_struc):
    previous = None
    for resid in rna_struc["resid"]:
        residue = (rna_code, int(resid))
        if residue != previous:
            yield residue
            previous = residue


# Pfam mapping: (pdb, AUTH_CHAIN) -> [(pfam_accession, pdb_start, pdb_end), ...]
# As in protseq/list-chains.py, the chain is the author chain, and the residue
# range uses the SIFTS sequential numbering (PDB_START/PDB_END), which matches
# the `resid` field of the struc -- not the author numbering (AUTH_PDBRES_*).
def load_pfam_mapping(path="pdb_pfam_mapping.csv"):
    pdb_chain_pfam = {}
    pfam_codes = set()
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "PDB":
                continue
            pdb, chain = row[0], row[1]
            pdb_start, pdb_end = int(row[2]), int(row[3])
            pfam_acc = row[4]
            pdb_chain_pfam.setdefault((pdb, chain), []).append(
                (pfam_acc, pdb_start, pdb_end)
            )
            pfam_codes.add(pdb)
    return pdb_chain_pfam, pfam_codes


def protein_atom_pfams(prot_struc, prot_chains, pdb, auth_map, pdb_chain_pfam):
    """Per protein-atom frozenset of Pfam accessions covering that residue."""
    pfam_sets = [frozenset()] * len(prot_struc)
    for chain in prot_chains:
        auth_chain = auth_map.get(chain.decode())
        if auth_chain is None:
            continue
        domains = pdb_chain_pfam.get((pdb, auth_chain))
        if not domains:
            continue
        chain_atoms = np.where(prot_struc["chain"] == chain)[0]
        chain_resids = prot_struc["resid"][chain_atoms]
        resid_to_pfam = {}
        for resid in np.unique(chain_resids):
            resid = int(resid)
            pfams = frozenset(
                acc for acc, start, end in domains if start <= resid <= end
            )
            if pfams:
                resid_to_pfam[resid] = pfams
        for atom_i, resid in zip(chain_atoms, chain_resids):
            pfams = resid_to_pfam.get(int(resid))
            if pfams:
                pfam_sets[atom_i] = pfams
    return pfam_sets


print("test")
struc = np.load("test-data/1b7f.npy")
test_contacts = contacting_protein_atoms(
    struc[struc["chain"] == b"A"], struc[struc["chain"] == b"C"]
)
assert len(test_contacts) == int(np.sum(struc["chain"] == b"A"))
print("/test")
print()

print("load Pfam mapping")
pdb_chain_pfam, pfam_codes = load_pfam_mapping()

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

pfam_per_residue = [set() for _ in range(len(residue_index))]

print("load interfaces")
interface_index, interface_data = Buffer.load(
    "input/allpdb-filtered-interfaces.mixed"
).get_value("mixed")
auth_chain = Buffer.load("input/allpdb-chain-auth.json").get_value("plain")

print("load complexes")
strucs = Buffer.load("input/allpdb-interface-struc.mixed").get_value("mixed")

for code in tqdm(interface_index):
    if_start, if_size = interface_index[code]
    if not if_size:
        continue
    pdb = code[:4]
    if pdb not in pfam_codes:
        continue

    interfaces = interface_data[if_start : if_start + if_size]
    prot_chains = set(iface["chain1"] for iface in interfaces)
    rna_chains = set(iface["chain2"] for iface in interfaces)

    struc0 = strucs[code]
    struc = struc0[struc0["model"] == 1]
    prot_mask = np.zeros(len(struc), bool)
    for chain in prot_chains:
        prot_mask |= struc["chain"] == chain
    prot_struc = struc[prot_mask]
    if not len(prot_struc):
        continue

    auth_map = dict(auth_chain[code])
    pfam_sets = protein_atom_pfams(
        prot_struc, prot_chains, pdb, auth_map, pdb_chain_pfam
    )
    if not any(pfam_sets):
        continue

    rna_struc_parts = []
    rna_residue_idx_parts = []
    for rna_chain in rna_chains:
        rna_code = pdb + rna_chain.decode()
        if rna_code not in rna_struc_index:
            continue
        r_start, r_size = rna_struc_index[rna_code]
        chunk = rna_strucs_data[r_start : r_start + r_size]
        rna_struc_parts.append(chunk)
        rna_residue_idx_parts.append(
            np.array(
                [residue_index[(rna_code, int(resid))] for resid in chunk["resid"]]
            )
        )
    if not rna_struc_parts:
        continue
    rna_struc = np.concatenate(rna_struc_parts)
    rna_res_idx = np.concatenate(rna_residue_idx_parts)

    contact_lists = contacting_protein_atoms(rna_struc, prot_struc)
    for rna_atom_i, prot_atom_list in enumerate(contact_lists):
        if not prot_atom_list:
            continue
        target = pfam_per_residue[rna_res_idx[rna_atom_i]]
        for prot_atom_i in prot_atom_list:
            pfams = pfam_sets[prot_atom_i]
            if pfams:
                target.update(pfams)

with open("allpdb-pfam-contacts.txt", "w") as f:
    for pfams in pfam_per_residue:
        f.write(" ".join(sorted(pfams)) + "\n")
