import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

def load_smiles(train_ready_path):
    df = pd.read_csv(train_ready_path, sep="\t")
    smiles = df["smiles"].astype(str).tolist()
    return smiles

def masked_mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def featurize_smiles(smiles_list, model_name_or_path, batch_size, max_length, device, local_files_only,
    layer_indices): 

    print(f"Using (finetuned?) ChemBERTa model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only) # grab tokenizer from chemberta, the same
    model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only) # grab model weights, normally the finetuned model
    model.eval().to(device) # apparently device is not working

    blocks = []
    with torch.no_grad():
        for ii in range(0, len(smiles_list), batch_size):
            batch = smiles_list[ii:ii + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length) # work the rest of the function in pytorch tensors and then convert
            for key in enc:
                enc[key] = enc[key].to(device)

            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states
            n_layers = 1 #len(hidden_states) we calculate n_layers in case we want to extend the model, but for the time being only one layer. Variable not used
            pooled_layers = []
            for ii in layer_indices:
                pooled = masked_mean_pool(hidden_states[ii], enc["attention_mask"])
                pooled_layers.append(pooled)
            pooled = torch.cat(pooled_layers, dim=1)
            blocks.append(np.array(pooled.cpu().numpy().astype(np.float32)))

    return np.vstack(blocks)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ready", default="train-ready.dat")
    parser.add_argument("--out", default="features-chemberta.dat")
    parser.add_argument("--batch-size", type=int,  default=16)
    parser.add_argument("--max-length", type=int, default=128) # TO DO: introduce more hidden layers of the transformer model hardcoded to one
    parser.add_argument("--model-name-or-path") #we default the base chemberta model, but the idea is to read the finetuned model
    parser.add_argument("--device", default="cpu") # cude does not seem to work in my setup
    parser.add_argument("--local-files-only") #saves huge amount of time and space
    return parser

def main(): #test as standalone featurizer might be helpful to prepare a batch of inference molecules

    args = parser().parse_args()
    train_ready = Path(args.train_ready)
    out_path = Path(args.out)

    layers = [-1] # hardcoded that 
    smiles = load_smiles(train_ready)

    X = featurize_smiles(smiles_list=smiles, model_name_or_path=args.model_name_or_path, batch_size=args.batch_size, max_length=args.max_length, device=args.device,
        local_files_only=args.local_files_only, layer_indices=layers)

    np.savetxt(out_path, X, fmt="%.6e")
    print(f"Embedding shape: {X.shape}")
    print(f"Saved features to: {out_path}")

if __name__ == "__main__":
    main()
