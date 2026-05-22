#!/usr/bin/env python3

from seamless import Buffer


def iter_resids(rna_struc):
    previous_resid = None

    for resid in rna_struc["resid"]:
        if resid != previous_resid:
            yield int(resid)
            previous_resid = resid


rna_struc_index, rna_strucs_data = Buffer.load(
    "input/allpdb-rna-aareduce.mixed"
).get_value("mixed")
auth_chain = Buffer.load("input/allpdb-chain-auth.json").get_value("plain")

with open("allpdb-print-nuc.txt", "w") as f, open(
    "allpdb-print-nuc-auth.txt", "w"
) as f_auth:
    for rna_code, (r_start, r_size) in rna_struc_index.items():
        pdb = rna_code[:4]
        chain = rna_code[4:]
        auth_map = dict(auth_chain[pdb + ".cif"])
        auth = auth_map[chain]

        rna_struc = rna_strucs_data[r_start : r_start + r_size]
        for resid in iter_resids(rna_struc):
            f.write(f"{pdb} {chain} {resid}\n")
            f_auth.write(f"{pdb} {auth} {resid}\n")
