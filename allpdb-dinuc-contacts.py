from seamless import Buffer
import numpy as np

fit_indices = np.load("input/allpdb-rna-fit-indices.npy")
data = Buffer.load("input/allpdb-rna-aareduce.mixed").deserialize("mixed")
contact_counts = np.load("allpdb-count-contacts.npy")
assert len(contact_counts) == len(data[1])

nucstart = np.where(data[1]["name"] == b"P")[0]

dinuc_starts = nucstart[fit_indices]
dinuc_ends = nucstart[2:][fit_indices]
slices = [slice(s, e) for s, e in zip(dinuc_starts, dinuc_ends)]

dinuc_contact_counts = np.array([contact_counts[sl].sum() for sl in slices])
np.save("allpdb-rna-fit-count-contacts.npy", dinuc_contact_counts.astype(np.uint16))
