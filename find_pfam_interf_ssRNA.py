#!/usr/bin/env python3

# usage: python3 find_pfam_interf_ssRNA.py single-bases.json -o pfam_ssRNA_interf.json --cutoff 5.0 --min_nt 4
# Find Pfam domains in contact with at least "min_nt" consecutive ssRNA nucleotides.

import argparse
import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path
from collections import defaultdict

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
PDBe_PFAM_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/pfam/{pdb}"

RNA_RESNAMES = {"A", "C", "G", "U", "I"}
PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
}

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


def download_if_missing(url, path):
    path = Path(path)
    if path.exists():
        return path
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def download_cif(pdb, cif_dir):
    cif_dir = Path(cif_dir)
    cif_dir.mkdir(exist_ok=True)
    path = cif_dir / f"{pdb}.cif.gz"
    return download_if_missing(RCSB_CIF_URL.format(pdb=pdb.lower()), path)


def split_mmcif_line(line):
    # Good enough for _atom_site records in most RCSB mmCIF files.
    # For stricter parsing, replace this with gemmi/Bio.PDB.MMCIFParser.
    return line.strip().split()


def parse_atoms_from_cif_gz(cif_path):
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
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                i += 1
            continue

        idx = {tag: n for n, tag in enumerate(tags)}

        required = [
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.label_asym_id",
            "_atom_site.label_seq_id",
        ]

        if not all(r in idx for r in required):
            continue

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

                atom = cols[idx["_atom_site.label_atom_id"]].strip('"')
                resname = cols[idx["_atom_site.label_comp_id"]].strip('"')

                auth_chain = cols[idx["_atom_site.auth_asym_id"]].strip('"')
                auth_seq = cols[idx["_atom_site.auth_seq_id"]].strip('"')

                label_chain = cols[idx["_atom_site.label_asym_id"]].strip('"')
                label_seq = cols[idx["_atom_site.label_seq_id"]].strip('"')

                if auth_seq not in {".", "?"} and not atom.startswith(
                    HYDROGEN_PREFIXES
                ):
                    try:
                        atoms.append(
                            {
                                "group": cols[idx["_atom_site.group_PDB"]],
                                "atom": atom,
                                "resname": resname,
                                "chain": auth_chain,
                                "resnum": int(float(auth_seq)),
                                "label_chain": label_chain,
                                "label_resnum": int(float(label_seq)),
                                "x": float(cols[idx["_atom_site.Cartn_x"]]),
                                "y": float(cols[idx["_atom_site.Cartn_y"]]),
                                "z": float(cols[idx["_atom_site.Cartn_z"]]),
                            }
                        )
                    except ValueError:
                        pass
            i += 1

    return atoms


def load_pfam_mapping_json(pdb, pfam_dir):
    pdb_l = pdb.lower()
    pfam_dir = Path(pfam_dir)
    pfam_dir.mkdir(exist_ok=True)
    path = pfam_dir / f"{pdb_l}.json"

    if path.exists():
        with open(path) as f:
            return json.load(f)

    url = PDBe_PFAM_URL.format(pdb=pdb_l)
    print(f"Downloading {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            print(f"Warning: no Pfam mapping found for {pdb} at PDBe")
            return {}
        if 500 <= error.code < 600:
            print(
                f"Warning: could not fetch Pfam mapping for {pdb} from PDBe: "
                f"HTTP {error.code} {error.reason}"
            )
            return {}
        raise
    except urllib.error.URLError as error:
        print(f"Warning: could not fetch Pfam mapping for {pdb} from PDBe: {error}")
        return {}
    except TimeoutError as error:
        print(f"Warning: timed out fetching Pfam mapping for {pdb} from PDBe: {error}")
        return {}

    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    return data


def pfam_regions_from_mapping_json(pdb, data):
    pdb_l = pdb.lower()
    regions = defaultdict(list)

    pfam_data = data.get(pdb_l, {}).get("Pfam", {})

    for pfam_acc, pfam_info in pfam_data.items():
        pfam = pfam_acc.split(".")[0]

        for mapping in pfam_info.get("mappings", []):
            chain = mapping.get("chain_id")
            start = mapping.get("start", {}).get("residue_number")
            end = mapping.get("end", {}).get("residue_number")

            if chain is None or start is None or end is None:
                continue

            if start > end:
                start, end = end, start

            regions[chain].append(
                {
                    "pfam": pfam,
                    "start": int(start),
                    "end": int(end),
                }
            )

    return regions


def load_pfam_regions(pdb, pfam_dir):
    data = load_pfam_mapping_json(pdb, pfam_dir)
    return pfam_regions_from_mapping_json(pdb, data)


def has_pfam_regions(pfam_regions_for_pdb):
    return any(pfam_regions_for_pdb.values())


def build_unknown_pfam_regions(atoms):
    residues_by_chain = defaultdict(list)

    for atom in atoms:
        if atom["resname"] in PROTEIN_RESNAMES:
            residues_by_chain[atom["chain"]].append(atom["resnum"])

    regions = defaultdict(list)
    for chain, resnums in residues_by_chain.items():
        regions[chain].append(
            {
                "pfam": "unknown",
                "start": min(resnums),
                "end": max(resnums),
            }
        )

    return regions


def pdb_ids_from_single_strand_data(single_strand_data):
    return sorted(str(pdb)[:4].upper() for pdb in single_strand_data)


def resnum_from_key(res_key):
    try:
        return (0, int(str(res_key).replace("res_", "", 1)))
    except ValueError:
        return (1, str(res_key))


def get_single_strand_stretches(single_strand_entry, min_nt):
    """
    single_strand_entry format:
    {
        "chain_R": [609, 610, 611, ...],
        "chain_S": [...]
    }
    """
    stretches = []

    for chain_key, resnums in single_strand_entry.items():
        nums = sorted(int(x) for x in resnums)

        current = []

        for n in nums:
            if current and n == current[-1] + 1:
                current.append(n)
            else:
                if len(current) >= min_nt:
                    stretches.append((chain_key, current.copy()))
                current = [n]

        if len(current) >= min_nt:
            stretches.append((chain_key, current.copy()))

    print("\nSS stretches retained:")
    for chain_key, stretch in stretches:
        print(chain_key, stretch[0], "->", stretch[-1], f"(n={len(stretch)})", stretch)

    return stretches


def squared_distance(a, b):
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2


def atom_in_pfam(atom, pfam_regions_for_pdb):
    chain = atom["chain"]
    resnum = atom["resnum"]

    for region in pfam_regions_for_pdb.get(chain, []):
        if region["start"] <= resnum <= region["end"]:
            yield region["pfam"].split(".")[0]


def find_pfam_contacts_with_ssRNA_stretches(
    single_strand_entry, atoms, pfam_regions_for_pdb, cutoff, min_nt
):
    cutoff2 = cutoff * cutoff

    protein_atoms = [a for a in atoms if a["resname"] in PROTEIN_RESNAMES]

    rna_atoms_by_chain_res = defaultdict(list)

    for a in atoms:
        if a["resname"] not in RNA_RESNAMES:
            continue
        rna_atoms_by_chain_res[(a["chain"], str(a["resnum"]))].append(a)

    pfam_to_contacted_ss_residues = defaultdict(set)
    pfam_to_runs = defaultdict(list)

    single_strand_stretches = get_single_strand_stretches(
        single_strand_entry, min_nt=min_nt
    )

    for chain_key, run_resnums in single_strand_stretches:
        rna_chain = chain_key.replace("chain_", "")

        for pdb_resnum_int in run_resnums:
            pdb_resnum = str(pdb_resnum_int)
            res_key = f"res_{pdb_resnum_int}"

            rna_atoms = rna_atoms_by_chain_res.get((rna_chain, pdb_resnum), [])
            if not rna_atoms:
                continue

            contacted_pfams = set()

            for pa in protein_atoms:
                pfams = list(atom_in_pfam(pa, pfam_regions_for_pdb))
                if not pfams:
                    continue

                for ra in rna_atoms:
                    if squared_distance(pa, ra) < cutoff2:
                        contacted_pfams.update(pfams)
                        break

            for pfam in contacted_pfams:
                pfam_to_contacted_ss_residues[(chain_key, pfam)].add(res_key)

    # Find maximal contacted runs of consecutive single-stranded nucleotides for each Pfam.
    pfams_ok = set()
    pfam_to_runs = defaultdict(list)

    for (chain_key, pfam), contacted_res in pfam_to_contacted_ss_residues.items():
        nums = sorted(int(r.replace("res_", "")) for r in contacted_res)

        runs = []
        current = []

        for n in nums:
            if current and n == current[-1] + 1:
                current.append(n)
            else:
                if len(current) >= min_nt:
                    runs.append(current)
                current = [n]

        if len(current) >= min_nt:
            runs.append(current)

        if runs:
            pfams_ok.add(pfam)
            for run in runs:
                pfam_to_runs[pfam].append(
                    {
                        chain_key: [f"res_{x}" for x in run],
                    }
                )
    print("pfam_to_contacted_ss_residues:")
    for (chain_key, pfam), residues in pfam_to_contacted_ss_residues.items():
        print(chain_key, pfam, sorted(residues, key=resnum_from_key))

    print("pfams_ok:", sorted(pfams_ok))
    print("pfam_to_runs:", dict(pfam_to_runs))

    return sorted(pfams_ok), pfam_to_runs


def main():
    parser = argparse.ArgumentParser(
        description="Find Pfam domains in contact with consecutive single-stranded RNA nucleotides."
    )
    parser.add_argument(
        "single_bases_json", help="JSON output from detect_single_bases.py"
    )
    parser.add_argument("-o", "--output", required=True, help="pfam_interf_ssRNA.json")
    parser.add_argument("--cif-dir", default="mmcif")
    parser.add_argument("--pfam-dir", default="pfam")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--min_nt", type=int, default=4)
    args = parser.parse_args()

    output_data = dict()
    with open(args.single_bases_json) as f:
        single_strand_data = json.load(f)

    for pdb in pdb_ids_from_single_strand_data(single_strand_data):
        output_data[pdb] = dict()

        print(f"Processing {pdb}")

        pfam_regions_for_pdb = load_pfam_regions(pdb, args.pfam_dir)
        cif_path = download_cif(pdb, args.cif_dir)
        atoms = parse_atoms_from_cif_gz(cif_path)

        if not has_pfam_regions(pfam_regions_for_pdb):
            print(f"Warning: no Pfam regions available for {pdb}; using 'unknown'")
            pfam_regions_for_pdb = build_unknown_pfam_regions(atoms)

        single_strand_entry = single_strand_data.get(pdb, {})

        pfams_ok, pfam_runs = find_pfam_contacts_with_ssRNA_stretches(
            single_strand_entry,
            atoms,
            pfam_regions_for_pdb,
            args.cutoff,
            args.min_nt,
        )

        output_data[pdb] = {pfam: pfam_runs.get(pfam, []) for pfam in pfams_ok}

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
