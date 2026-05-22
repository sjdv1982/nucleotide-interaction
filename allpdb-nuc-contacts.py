#!/usr/bin/env python3

import numpy as np
from seamless import Buffer


def iter_residue_slices(rna_struc):
    previous_resid = None
    residue_start = 0

    for pos, resid in enumerate(rna_struc["resid"]):
        if previous_resid is None:
            previous_resid = resid
            continue

        if resid != previous_resid:
            yield slice(residue_start, pos)
            residue_start = pos
            previous_resid = resid

    if previous_resid is not None:
        yield slice(residue_start, len(rna_struc))


rna_struc_index, rna_strucs_data = Buffer.load(
    "input/allpdb-rna-aareduce.mixed"
).get_value("mixed")
contact_counts = np.load("allpdb-count-contacts.npy")
assert len(contact_counts) == len(rna_strucs_data)

nuc_contact_counts = []
for r_start, r_size in rna_struc_index.values():
    rna_struc = rna_strucs_data[r_start : r_start + r_size]
    for residue_slice in iter_residue_slices(rna_struc):
        residue_start = r_start + residue_slice.start
        residue_stop = r_start + residue_slice.stop
        nuc_contact_counts.append(contact_counts[residue_start:residue_stop].sum())

np.save("allpdb-nuc-contacts.npy", np.asarray(nuc_contact_counts, dtype=np.uint16))
