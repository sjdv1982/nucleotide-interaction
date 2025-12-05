import numpy as np
from seamless import Buffer
from tqdm import tqdm
from scipy.spatial import KDTree


def get_coor(struc):
    return np.stack((struc["x"], struc["y"], struc["z"]), axis=-1)


def count_contacts(rna_struc, prot_struc, distance=5):
    rna_coor = get_coor(rna_struc)
    rna_tree = KDTree(rna_coor)
    prot_coor = get_coor(prot_struc)
    prot_tree = KDTree(prot_coor)
    contact_lists = rna_tree.query_ball_tree(prot_tree, distance)
    return np.array([len(l) for l in contact_lists], np.uint)


struc = np.load("test-data/1b7f.npy")
prot_struc = struc[struc["chain"] == b"C"]
rna_struc = struc[struc["chain"] == b"A"]

print("test")
result = count_contacts(rna_struc, prot_struc)
assert len(result) == len(rna_struc)
print(np.histogram(result))
print(np.where(result)[0])
print("/test")
print()

interface_index, interface_data = Buffer.load(
    "input/allpdb-filtered-interfaces.mixed"
).deserialize("mixed")
allpdb_keyorder = Buffer.load("input/allpdb-keyorder.json").deserialize("plain")

print("load RNA")
rna_struc_index, rna_strucs_data = Buffer.load(
    "input/allpdb-rna-aareduce.mixed"
).deserialize("mixed")
rna_strucs_contacts = np.empty(len(rna_strucs_data), np.uint)
rna_strucs_contact_done = np.zeros(len(rna_strucs_data), bool)

print("load complexes")
strucs = Buffer.load("input/allpdb-interface-struc.mixed").deserialize("mixed")

for code in tqdm(interface_index):
    if_start, if_size = interface_index[code]
    if not if_size:
        continue

    interfaces = interface_data[if_start : if_start + if_size]
    s4 = np.dtype("|S4")
    prot_chains = set([iface["chain1"] for iface in interfaces])
    rna_chains = set([iface["chain2"] for iface in interfaces])
    struc0 = strucs[code]
    struc = struc0[struc0["model"] == 1]
    prot_mask = np.zeros(len(struc), bool)
    for chain in prot_chains:
        prot_mask |= struc["chain"] == chain
    prot_struc = struc[prot_mask]
    assert len(prot_struc), code
    rna_struc_offsets = []
    rna_struc = []
    for rna_chain in rna_chains:
        rna_code = code[:4] + rna_chain.decode()
        if rna_code not in rna_struc_index:
            ##print("SKIP", rna_code)
            continue
        ###print("CODE", rna_code)
        r_start, r_size = rna_struc_index[rna_code]
        chain_rna_struc = rna_strucs_data[r_start : r_start + r_size]
        rna_struc.append(chain_rna_struc)
        rna_struc_offsets.append((r_start, r_size))
    if not len(rna_struc):
        continue
    rna_struc = np.concatenate(rna_struc)
    rna_struc_contacts = count_contacts(rna_struc, prot_struc)

    pos = 0
    for r_start, r_size in rna_struc_offsets:
        newpos = pos + r_size
        contacts = rna_struc_contacts[pos:newpos]
        rna_strucs_contacts[r_start : r_start + r_size] = contacts
        rna_strucs_contact_done[r_start : r_start + r_size] = 1
        newpos = pos

assert np.all(rna_strucs_contact_done)
np.save("allpdb-count-contacts.npy", rna_strucs_contacts)
