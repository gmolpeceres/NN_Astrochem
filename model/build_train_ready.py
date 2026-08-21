import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def load_smiles_table(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["smiles", "name"])
    df["smiles"] = df["smiles"].astype(str)
    df["name"] = df["name"].astype(str)
    df["canonical_smiles"] = df["smiles"].map(canonicalize_smiles)
    return df

def load_density_table(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["smiles", "column_density"])
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["column_density"] = pd.to_numeric(df["column_density"], errors="coerce")
    df["canonical_smiles"] = df["smiles"].map(canonicalize_smiles)
    return df

def build_train_ready(smiles_path, density_path, output_path):
    smiles_df = load_smiles_table(smiles_path)
    density_df = load_density_table(density_path)

    smiles_df = smiles_df.reset_index().rename(columns={"index": "source_order"})

    merged = smiles_df.merge(
        density_df[["canonical_smiles", "column_density"]],
        on="canonical_smiles",
        how="left",
        validate="one_to_one",
    )
    merged = merged.sort_values("source_order").reset_index(drop=True)

    ready = pd.DataFrame(
        {
            "smiles": merged["canonical_smiles"],
            "name": merged["name"],
            "column_density": merged["column_density"].astype(float),
        }
    )
    ready["logn"] = np.log10(ready["column_density"])
    ready.to_csv(output_path, sep="\t", index=False, float_format="%.10e")
    return ready

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--densities")
    parser.add_argument("--outfile")
    return parser

def main():
    args = parser().parse_args()

    smiles_path = Path(args.smiles)
    density_path = Path(args.densities)
    out_path = Path(args.outfile)

    ready = build_train_ready(smiles_path=smiles_path, density_path=density_path, output_path=out_path)
    print(f"Testing isolated train_ready script")
    print(f"{out_path} with {len(ready)} rows.")

if __name__ == "__main__":
    main()
