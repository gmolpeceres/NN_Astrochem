import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

molecular_descriptors = [ # using markus dablander list
    # size  
    "ExactMolWt",
    "MolWt",
    "HeavyAtomMolWt",
    "HeavyAtomCount",
    # electrons
    "NumValenceElectrons",
    "NumRadicalElectrons",
    # heteroatoms 
    "NumHeteroatoms",
    "NOCount",
    "NHOHCount",
    # polarity and surface 
    "TPSA",
    "LabuteASA",          # surface 
    "MolMR",              # polarizability 
    "MolLogP",
    # topology / structure 
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "NumRotatableBonds",
    # H bonding 
    "NumHAcceptors",
    "NumHDonors",
#    following four give Nan for diatomics like NO or CO, does not work!
#    "MaxAbsPartialCharge",
#    "MinAbsPartialCharge",
#    "MaxPartialCharge",
#    "MinPartialCharge",
    "FormalCharge",
    # connectivity density not sure if useful
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BertzCT",       
    "BalabanJ",      
    "HallKierAlpha",
    "Ipc",          
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",   
    # nitrile
    "fr_nitrile",
    # carbon chains
    "fr_term_acetylene",
    "fr_alkyne",   
    # oxygen 
    "fr_ether",
    "fr_ketone",
    "fr_aldehyde",
    "fr_ester",
    "fr_C_O",
    "fr_C_O_noCOO",   
    # nitrogen take out non detected groups, but chaos creates some
    "fr_amine",
    "fr_imine",
    "fr_amide",
#    "fr_nitro",
#    "fr_azide",
#    "fr_diazo",   
    # aromatic / rings
    "fr_benzene",
    "fr_heterocycle",
    "fr_aromatic", #seems that is not wroking very well for partial aromaticity
    # sulfur 
    "fr_sulfide",
    "fr_thiophene",
    "fr_SH",
]

def rdkit_descriptors(smiles, descriptor_names):
    mol = Chem.MolFromSmiles(smiles)
    calc = MolecularDescriptorCalculator(descriptor_names)
    vals = list(calc.CalcDescriptors(mol))
    return descriptor_names, vals

def load_smiles(train_ready_path):
    df = pd.read_csv(train_ready_path, sep="\t")
    smiles = df["smiles"].astype(str).tolist()
    return smiles

def build_descriptor_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        names, values = rdkit_descriptors(smi, descriptor_names=molecular_descriptors)
        rows.append(values)
    X = np.asarray(rows, dtype=np.float32)
    return names, X

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ready", default="train-ready.dat")
    parser.add_argument("--out", default="rdkit-descriptors.dat")
    return parser

def main():
    args = parser().parse_args()
    train_ready = Path(args.train_ready)
    out_path = Path(args.out)

    smiles = load_smiles(train_ready)
    names, X = build_descriptor_matrix(smiles)

    np.savetxt(out_path, X, fmt="%.6e")
    names_out = Path(f"{out_path}.names.txt")
    with names_out.open("w") as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"Descriptor size: {X.shape}")
    print(f"Descriptors: {out_path}")

if __name__ == "__main__":
    main()
