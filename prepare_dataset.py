"""
prepare_dataset.py
==================
Generates CoT-structured prompt datasets from the MassSpecGym benchmark for
MS-to-SMILES zero-shot evaluation.

Usage:
    python prepare_dataset.py [--split {train,val,test,full}] [--no-dedup]

Output:
    data/benchmark_dataset_<split>.jsonl
"""

import argparse
import os
import re
from datasets import Dataset, load_dataset
from tqdm import tqdm
from rdkit import Chem

# ------------------------------------------------------- #
# Templates
# ------------------------------------------------------- #

INSTRUCTION_TEMPLATE = """
You are an expert in analyzing Mass Spectra (MS) to deduce the corresponding SMILES. Given the MS m/z values: <mzs>, intensities: <intensities>, and molecular formula: <formula>, instrument: <instrument>, adduct (Ionization Method): <adduct>, and collision energy: <collision_energy> eV, your task is to analyze the spectrum and propose plausible SMILES representation candidates of the molecule. Please provide your analysis in <think> block and the most probable SMILES in <answer> block with the following template:
<think>
1. Formula and DBE Analysis:
   * Formula: <formula>
* Double Bond Equivalents (DBE): <DBE> (Calculated as C - H/2 + N/2 + 1)
* Initial structural implications from DBE: (e.g., presence of rings, double/triple bonds)
2. Key Peak Identification and Initial Fragmentation:
* Base Peak: m/z <base_peak_mz> with intensity <base_peak_intensity>. Potential stable fragment/substructure: <substructure_formula_from_base_peak>.
3. Neutral Loss Analysis from Significant Peaks:
* From M+ (or <source_peak_mz>): Loss of <mass_A> (m/z <molecular_ion_mz> -> m/z <fragment_A_mz>) suggests loss of <neutral_fragment_A_formula_or_structure>.
* From m/z <source_peak_B_mz>: Loss of <mass_B> (m/z <source_peak_B_mz> -> m/z <fragment_B_mz>) suggests loss of <neutral_fragment_B_formula_or_structure>.
* Common neutral losses observed (e.g., H2O, CO, NH3, C2H4): <list_common_losses_and_implications>.
4. Fragment Ion Relationship and Substructure Assembly:
* Peak at m/z <fragment_X_mz> likely arises from <parent_ion_mz_for_X> via <loss_or_rearrangement_for_X>. Proposed substructure for <fragment_X_mz>: <subformula_X>.
* Peak at m/z <fragment_Y_mz> and its relation to <fragment_X_mz> or other peaks.
* Hypothesize connections between identified substructures (<subformula_A>, <subformula_X>, etc.) consistent with <formula> and DBE.
5. Structure Elucidation and SMILES Hypothesis:
* Synthesize fragment information, neutral losses, and DBE to propose a consistent chemical structure.
* Consider plausible fragmentation mechanisms for the observed spectrum.
</think>
<answer>
Final 10 SMILES Proposals: <smiles_1>,<smiles_2>,...,<smiles_10>
</answer>
"""

INPUT_TEMPLATE = """MS m/z values: {mzs}
intensities: {intensities}
molecular formula: {formula} (DBE: {dbe:.1f})
instrument: {instrument}
adduct: {adduct}
collision energy: {collision_energy} eV"""


# ------------------------------------------------------- #
# Helpers
# ------------------------------------------------------- #


def calc_dbe(formula_str: str) -> float:
    """Double-bond equivalent = C - H/2 + N/2 + 1."""
    atom_counts = dict(re.findall(r"([A-Z][a-z]*)(\d*)", formula_str))
    atom_counts = {k: int(v) if v else 1 for k, v in atom_counts.items()}
    c = atom_counts.get("C", 0)
    h = atom_counts.get("H", 0)
    n = atom_counts.get("N", 0)
    return c - h / 2 + n / 2 + 1


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


def parse_peak_list(mzs_str: str, intensities_str: str):
    """Parse comma-separated m/z and intensity strings into float lists."""
    mzs = [round(float(v), 4) for v in mzs_str.split(",")]
    intensities = [round(float(v), 3) for v in intensities_str.split(",")]
    return mzs, intensities


def generate_prompt(row: dict) -> dict:
    """Convert a single MassSpecGym row into a prompt record."""
    formula = row["formula"]
    mzs, intensities = parse_peak_list(row["mzs"], row["intensities"])
    dbe = calc_dbe(formula)

    prompt_input = INPUT_TEMPLATE.format(
        mzs=mzs,
        intensities=intensities,
        formula=formula,
        dbe=dbe,
        instrument=row["instrument_type"],
        adduct=row["adduct"],
        collision_energy=row["collision_energy"],
    )

    canonical = canonicalize_smiles(row["smiles"])
    if not canonical:
        raise ValueError(f"Invalid SMILES: {row['smiles']!r}")

    return {
        "identifier": row["identifier"],
        "inchikey": row["inchikey"],
        "fold": row["fold"],
        "instruction": INSTRUCTION_TEMPLATE,
        "input": prompt_input,
        "output": canonical,
    }


# ------------------------------------------------------- #
# Main
# ------------------------------------------------------- #


def main(split: str = "test", unique_inchi: bool = True):
    """
    Args:
        split: one of "train", "val", "test", or "full" (all folds).
        unique_inchi: if True, keep only one spectrum per unique InChIKey
                      (selected in the order they appear in the dataset).
    """
    print(
        f"Loading MassSpecGym dataset (split='{split}', unique_inchi={unique_inchi}) ..."
    )

    # MassSpecGym is published as a single HuggingFace table under the "train" key.
    # The fold assignment (train/val/test) is stored in the "fold" column, following
    # the DiffMS split protocol (MCES dissimilarity >= 10 between train and test).
    hf_dataset = load_dataset(
        "roman-bushuiev/MassSpecGym",
        cache_dir="./data/MSGymDataset_HF",
    )[
        "train"
    ]  # "train" here is the HF table name, not the ML split

    if split != "full":
        hf_dataset = hf_dataset.filter(lambda x: x["fold"] == split)

    print(f"  Rows after fold filter: {len(hf_dataset)}")

    records = []
    seen_inchikeys: set = set()
    skipped_invalid = 0
    skipped_dedup = 0

    for row in tqdm(hf_dataset, desc="Preparing prompts"):
        inchikey = row["inchikey"]
        if unique_inchi and inchikey in seen_inchikeys:
            skipped_dedup += 1
            continue
        try:
            record = generate_prompt(row)
        except Exception as e:
            skipped_invalid += 1
            print(f"  Skipping {row.get('identifier', '?')}: {e}")
            continue
        records.append(record)
        seen_inchikeys.add(inchikey)

    print(f"  Records kept   : {len(records)}")
    print(f"  Skipped (dedup): {skipped_dedup}")
    print(f"  Skipped (error): {skipped_invalid}")

    os.makedirs("data", exist_ok=True)
    out_path = f"data/benchmark_dataset_{split}.jsonl"
    Dataset.from_list(records).to_json(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MassSpecGym CoT benchmark dataset"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "full"],
        default="test",
        help="Which MassSpecGym fold to output (default: test)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep all spectra, including duplicate InChIKeys",
    )
    args = parser.parse_args()
    main(split=args.split, unique_inchi=not args.no_dedup)
