PYTHON = python #/home/german/miniconda3/envs/rdkit/bin/python

MODEL_DIR = model
MODEL_OUT = model
INFERENCE_DIR = inference

SMILES_SRC = ./datasets/tmc-1.smiles #tmc-1_withUL.smiles
DENSITIES_SRC = ./datasets/tmc-1.columndensity #tmc-1_withUL.columndensity
SMILES = $(MODEL_OUT)/tmc-1.smiles
DENSITIES = $(MODEL_OUT)/tmc-1.columndensity
TRAIN_READY = $(MODEL_OUT)/train-ready.dat
FEATURES = $(MODEL_OUT)/features-chemberta.dat
DESCRIPTORS = $(MODEL_OUT)/rdkit-descriptors.dat
TRAINING_OUTDIR = $(MODEL_OUT)/training_outputs
CHEMBERTA_MODEL_OR_PATH = /home/german/Projects_local/ISOCOSMOS/SP1/PrepareModelSubmission/astrochemberta/chemberta_dapt_out #DeepChem/ChemBERTa-100M-MLM #/home/german/Projects_local/ISOCOSMOS/SP1/PrepareModelSubmission/astrochemberta/chemberta_dapt_out
FEATURIZER_BATCH_SIZE = 16
FEATURIZER_MAX_LENGTH = 128
FEATURIZER_DEVICE = cuda
FEATURIZER_LOCAL_FILES_ONLY = 1
TRAIN_TEST_SIZE = 0.2
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 5000
TRAIN_N_MODELS = 3 #3
SEED = 1
MODEL_DIRECTORY = $(TRAINING_OUTDIR) # model.keras #$(TRAINING_OUTDIR)
FEATURIZER_OUTFILE_FLAG = $(if $(FEATURIZER_OUTFILE),--outfile $(FEATURIZER_OUTFILE),)
SEED_FLAG = $(if $(SEED),--seed $(SEED),)

all: build-train-ready featurize rdkit-descriptors train
generate: build-train-ready featurize rdkit-descriptors

prepare-inputs:
	cp $(SMILES_SRC) $(SMILES)
	cp $(DENSITIES_SRC) $(DENSITIES)

build-train-ready: prepare-inputs
	$(PYTHON) $(MODEL_DIR)/build_train_ready.py \
		--smiles $(SMILES) \
		--densities $(DENSITIES) \
		--outfile $(TRAIN_READY)

featurize:
	LD_LIBRARY_PATH='' $(PYTHON) $(MODEL_DIR)/featurizer.py \
		--train-ready $(TRAIN_READY) \
		--out $(FEATURES) \
		--model-name-or-path $(CHEMBERTA_MODEL_OR_PATH) \
		--batch-size $(FEATURIZER_BATCH_SIZE) \
		--max-length $(FEATURIZER_MAX_LENGTH) \
		--device $(FEATURIZER_DEVICE) $(FEATURIZER_LOCAL_FILES_FLAG)

rdkit-descriptors:
	$(PYTHON) $(MODEL_DIR)/rdkit_descriptors.py \
		--train-ready $(TRAIN_READY) \
		--out $(DESCRIPTORS)

train:
	LD_LIBRARY_PATH='' $(PYTHON) $(MODEL_DIR)/model_test.py \
		--train-ready $(TRAIN_READY) \
		--features $(FEATURES) \
		--descriptors $(DESCRIPTORS) \
		--outdir $(TRAINING_OUTDIR) \
		--test-size $(TRAIN_TEST_SIZE) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--epochs $(TRAIN_EPOCHS) \
		--n-models $(TRAIN_N_MODELS) $(SEED_FLAG) \
		--save-history 

clean:
	rm -rf $(TRAINING_OUTDIR)
	rm -rf $(MODEL_DIR)/__pycache__

clean-all: clean
	rm -f $(TRAIN_READY)
	rm -f $(FEATURES)
	rm -f $(DESCRIPTORS)
	rm -f $(DESCRIPTORS).names.txt
	rm -f $(SMILES)
	rm -f $(DENSITIES)

