# NN_Astrochem — Neural-network model to find correlations in astrochemical space

Daniel Giovanni Vargas Wilches (UNAL, Colombia) and Dr. Germán Molpeceres (IFF-CSIC, Spain)

*Paper accepted in Astronomy and Astrophysics (https://doi.org/10.48550/arXiv.2610.30842)*

Based on the original idea presented in Kin Long Kelvin Lee et al. 2021, ApJL 917, L6 ([DOI: 10.3847/2041-8213/ac194b](https://doi.org/10.3847/2041-8213/ac194b)).

---

## What is this?

NN_Astrochem trains small neural networks that learn the relation between the **molecular structure** of a species (its SMILES string) and its **observed column density** in an astronomical source (by default, the cold dark cloud TMC-1). Once trained, the model's latent space is used to **propose new candidate molecules** that are likely to be detectable in that same source.

| | Description |
|---|---|
| **Input** | SMILES of molecules already detected in the source |
| **Target** | Observed column density, *N* (cm⁻²), trained as log₁₀ *N* |
| **Output** | A list of candidate molecules (from a ~50k-molecule inventory) that fall in the "high-abundance" region of the model's latent space |

Everything runs from the **command line** through a `makefile`. 

---

## Pipeline at a glance

```
 TRAINING                                         
 ────────                                         
 tmc-1.smiles ─┐                                  
 tmc-1.column- ├─► build-train-ready ─► train-ready.dat  ─┐    
   density ────┘    │                                     |
                    ├─► featurize  (ChemBERTa, 768-d)     |─►train ──► n models
                    │                                     |
                    └─► rdkit-descriptors (50-d)      ────┘                   

```

---

## 1. Requirements

- Linux (we did not test Mac or Windows)
- [Miniconda / Anaconda](https://docs.conda.io/)
- `make`

Main Python packages (all installed by `environment.yml`): Python 3.11, Keras 3 (PyTorch backend, although other backends are possible, i.e. Tensorflow was used in the article), PyTorch, Hugging Face `transformers`, RDKit, scikit-learn, pandas, matplotlib.

## 2. Installation

```bash
# 1. Clone the repository
git clone <REPO_URL> NN_Astrochem
cd NN_Astrochem

# 2. Create the conda environment
conda env create -f environment.yml
conda activate keras3-env

```

> **Note:** `environment.yml` contains a `prefix:` line pointing to the authors' machine. If conda complains, delete that line and run the command again.

## 3. Repository layout

```
NN_Astrochem/
├── makefile                  # entry point: every step is a make target
├── environment.yml           # conda environment
├── datasets/
│   ├── tmc-1.smiles          # SMILES of molecules detected in TMC-1
│   └── tmc-1.columndensity   # their observed column densities
├── astrochemberta/           # Should be downloaded from Zenodo or use a HG one
│   └── chemberta_dapt_out/   # ChemBERTa adapted to astrochemical molecules
├── model/                    # scripts for the training pipeline
│   ├── build_train_ready.py
│   ├── featurizer.py
│   ├── rdkit_descriptors.py
│   └── model_test.py
```

All results of a run are written to a single folder, set by `MODEL_OUT` in the makefile (default: `output`).

---

## 4. Quick start

```bash
conda activate keras3-env
mkdir -p output        # output folder 

# Training
make all               # vectorize + train
```


The rest of this document explains what each step does and how to change it.

---

## 5. Step-by-step tutorial

### Part A — Training

#### A.1 Build the training table — `make build-train-ready`

Copies the input files into the output folder and merges SMILES and column densities into one table.

- **Reads:** `datasets/tmc-1.smiles`, `datasets/tmc-1.columndensity`
- **Writes:** `$(MODEL_OUT)/train-ready.dat` with columns:

| column | meaning |
|---|---|
| `smiles` | molecular structure |
| `formula` | molecular formula |
| column density | *N* in cm⁻² |
| `logn` | log₁₀ *N* (the value the model learns) |

The default dataset is the TMC-1 inventory (detections up to 2021, pre-QUIJOTE and GOTHAM, and current (January 2026)).

#### A.2 ChemBERTa embeddings — `make featurize`

Converts every SMILES into a **768-component vector** using ChemBERTa, a language model for molecules. The default model (`astrochemberta/chemberta_dapt_out`) was further adapted to astrochemistry using molecules generated with **CHAOS**, our algorithm for generating molecular permutations. For using the original ChemBERTa model, substitute in the makefile for the HuggingFace corresponding URL (commented in the Makefile). Other MLM transformer models are theoretically usable in this context, although we did not train any outside of the ChemBERTa ones.

- **Reads:** `train-ready.dat`
- **Writes:** `$(MODEL_OUT)/features-chemberta.dat`


#### A.3 RDKit descriptors — `make rdkit-descriptors`

Computes a **50-component vector** of classical chemical descriptors per molecule (number of heavy atoms, number of aromatic rings, etc.) using RDKit. Several other descriptors can be included in this generation, we did some test with some combinations of those, extracted from the blog post (https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit). We only discourage the use of descriptors related with partial charges as they produce NaNs for diatomic molecules

- **Reads:** `train-ready.dat`
- **Writes:** `$(MODEL_OUT)/rdkit-descriptors.dat` and `rdkit-descriptors.dat.names.txt` (the name of each descriptor)

> Steps A.1–A.3 can be run together with `make generate`.

#### A.4 Train the models — `make train`

Trains a neural network that combines both representations (ChemBERTa + RDKit) to predict log₁₀ *N*.

- **Reads:** `train-ready.dat`, `features-chemberta.dat`, `rdkit-descriptors.dat`
- **Writes:** `$(MODEL_OUT)/training_outputs/` — **n trained models** with the same architecture but different weights and train/validation splits, plus their training history.

Training three models gives an idea of how stable the predictions are against the choice of training data.

Default settings: 80/20 train/test split, batch size 32, 5000 epochs (max), seed 1.

---

## 6. Changing the configuration

Every setting in the makefile can be overridden from the command line without editing the file:

```bash
# Train for 50 epochs, 5 models, in a new folder
mkdir -p run_50ep
make all MODEL_OUT=run_50ep TRAIN_EPOCHS=50 TRAIN_N_MODELS=5
```

| Variable | Default | What it controls |
|---|---|---|
| `MODEL_OUT` | `output` | output folder for the run |
| `SMILES_SRC` | `./datasets/tmc-1.smiles` | input SMILES |
| `DENSITIES_SRC` | `./datasets/tmc-1.columndensity` | input column densities |
| `CHEMBERTA_MODEL_OR_PATH` | `astrochemberta/chemberta_dapt_out` | ChemBERTa model (local path or Hugging Face name, e.g. `DeepChem/ChemBERTa-100M-MLM`) |
| `FEATURIZER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `FEATURIZER_BATCH_SIZE` | `16` | molecules per batch during featurization |
| `FEATURIZER_MAX_LENGTH` | `128` | Change only if you own transformer mode with higer lengthl |
| `TRAIN_TEST_SIZE` | `0.2` | fraction of data held out for testing |
| `TRAIN_BATCH_SIZE` | `32` | training batch size |
| `TRAIN_EPOCHS` | `5000` | training epochs |
| `TRAIN_N_MODELS` | `3` | number of models trained |
| `SEED` | `1` | random seed (reproducibility) |

### Using another astronomical source

To train on a different source, prepare two files with the same format as the TMC-1 ones and point the makefile to them:

```bash
mkdir -p my_source_run
make all MODEL_OUT=my_source_run \
         SMILES_SRC=./datasets/my_source.smiles \
         DENSITIES_SRC=./datasets/my_source.columndensity
```

Keep in mind that the model learns from the detections you give it: a source with few detections will give a less reliable model. (See section 4.2 of the article)

## 7. Cleaning up

```bash
make clean       # removes training outputs and caches (keeps vectors)
make clean-all   # also removes train-ready.dat, features and descriptors
```

---

## Additional data

Several data discussed in the article can be retrieved at https://zenodo.org/records/23531603

## Citation

If you use this code, please cite the paper (definitive URL coming soon and https://doi.org/10.48550/arXiv.2610.30842) and Lee et al. 2021, ApJL 917, L6.

## Contact

Germán Molpeceres: german.molpeceres[at]iff.csic.es
Daniel Giovanni Vargas: dvargasw[at]unal.edu.co


## AI Use Disclosure

The current code originates from a series of legacy Jupyter notebooks developed by the authors in Bogotá during a research visit by GM to the Observatorio Nacional de Colombia in January 2026. From those OpenAI Codex has been used to assist with portions of the code refactoring. All scientific decisions (planning of the project, design of the research, pursue of results) and data-related work, including SMILES collection, model training, validation, and interpretation of results, were performed by the authors.