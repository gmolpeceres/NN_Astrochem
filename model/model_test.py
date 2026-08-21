import argparse
import json
import random
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, Concatenate, Dense, Dropout, GaussianNoise, Input, LayerNormalization
from keras.optimizers import Adam, RMSprop, Nadam #choose from optimizer
from keras.losses import Huber
from keras.regularizers import l2
from keras.utils import set_random_seed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def load_matrix(path): #auxiliary function to read any input feature correctly
    X = np.loadtxt(path)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1: #required for input in nn
        X = X.reshape(1, -1)
    return X

def load_targets(train_ready_path): # read the column densities from the train-ready file
    df = pd.read_csv(train_ready_path, sep="\t")
    return df["logn"].to_numpy(dtype=np.float32)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

def build_model(emb_dim, phys_dim):
    emb_in = Input(shape=(emb_dim,), name="emb_in")
    e = GaussianNoise(0.09, name="emb_noise")(emb_in)
    e = Dense(512, kernel_regularizer=l2(0.008476864720199208), name="emb_mlp256")(e)
#    e = LayerNormalization(name="emb_ln1")(e)
    e = Activation("gelu", name="emb_act1")(e)
    e = Dropout(0.30, name="emb_drop1")(e) #choice from optimizer, note that phys_n should be changed by p below
    e = Dense(256, kernel_regularizer=l2(4.6157971293047096e-05), name="emb_mlp64")(e)
#    e = LayerNormalization(name="emb_ln2")(e)
    e = Activation("gelu", name="emb_act2")(e)
    e = Dropout(0.30, name="emb_drop2")(e)
    phys_in = Input(shape=(phys_dim,), name="phys_in")
    p = GaussianNoise(0.07, name="phys_noise")(phys_in) #choice from optimizer, note that phys_n should be changed by p below
    p = Dense(64, kernel_regularizer=l2(0.00046303624422214923), name="phys_mlp16")(p)
    p = LayerNormalization(name="phys_ln")(p)
    p = Activation("gelu", name="phys_act")(p)
    p = Dropout(0.25, name="phys_drop")(p)
    x = Concatenate(name="concat")([e, p])
    x = Dense(198, kernel_regularizer=l2(0.004070611711771424), name="fusion_mlp64")(x)
#    x = LayerNormalization(name="fusion_ln")(x)
    x = Activation("gelu", name="fusion_act")(x)
    x = Dropout(0.1, name="fusion_drop")(x)
    out = Dense(1, name="LogN")(x)
    model = Model(inputs=[emb_in, phys_in], outputs=out)
    model.compile(optimizer=Nadam(learning_rate=0.0007406716024474344), loss=Huber(delta=2.0), metrics=["mse", "mae"])
    model.summary()
    return model

def preprocessing( X_embed, X_phys, y, test_size: float = 0.20, random_state: int = 42):

    indices = np.arange(y.shape[0])
    idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)

    Xe_train, Xe_val = X_embed[idx_train], X_embed[idx_val]
    Xp_train, Xp_val = X_phys[idx_train], X_phys[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    emb_scaler = StandardScaler()
    phys_scaler = StandardScaler()
    Xe_train = emb_scaler.fit_transform(Xe_train)
    Xe_val = emb_scaler.transform(Xe_val)
    Xp_train = phys_scaler.fit_transform(Xp_train)
    Xp_val = phys_scaler.transform(Xp_val)

    return Xe_train, Xe_val, Xp_train, Xp_val, y_train, y_val, idx_train, idx_val, emb_scaler, phys_scaler


def train_and_evaluate( Xe_train, Xe_val, Xp_train, Xp_val, y_train, y_val,
    epochs: int = 2000, batch_size: int = 32):

    model = build_model(emb_dim=Xe_train.shape[1], phys_dim=Xp_train.shape[1])
    
    early = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=800,          
    min_delta=1e-4,        
    restore_best_weights=True,
    verbose=1,
)

    reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=400,          
    min_delta=1e-4,        
    cooldown=100,          
    min_lr=1e-6,
    verbose=1,
)

    history = model.fit(
        [Xe_train, Xp_train],
        y_train,
        validation_data=([Xe_val, Xp_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early, reduce_lr],
    )
    
    y_pred = model.predict([Xe_val, Xp_val], verbose=0).reshape(-1)
    metrics = get_metrics(y_val, y_pred)

    return model, metrics, y_pred, history.history


def summarize_metric_list(metric_list):
    keys = ("mse", "mae", "r2")
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([m[key] for m in metric_list], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return summary

def save_diagnostic_plots(y_val, y_pred_val, y_full, y_pred_full, outdir, val_names, full_names): # correlation/outlier plots for validation and full dataset

    plot_jobs = [
        (y_val, y_pred_val, val_names, "nn_correlation.pdf", "nn_outliers.pdf"),
        (y_full, y_pred_full, full_names, "nn_full_correlation.pdf", "nn_full_outliers.pdf"),
    ]

    for y_true, y_pred, sample_names, corr_name, out_name in plot_jobs:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_pred, y_true, alpha=0.8)
        plt.xlabel("Predicted LogN")
        plt.ylabel("Experimental LogN")
        plt.tight_layout()
        plt.savefig(outdir / corr_name, format="pdf", bbox_inches="tight")
        plt.close()

        n_outliers = min(10, len(abs_residuals)) #hardcoded to 10 residuals
        outlier_idx = np.argsort(abs_residuals)[-n_outliers:][::-1]

        plt.figure(figsize=(8, 5))
        plt.scatter(np.arange(len(abs_residuals)), abs_residuals, alpha=0.35, label="Validation samples")
        plt.scatter(outlier_idx, abs_residuals[outlier_idx], color="red", label=f"Top {n_outliers} outliers")

        for idx in outlier_idx:
            label = f"{sample_names[idx]}"
            plt.annotate(label, (idx, abs_residuals[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)

        plt.xlabel("Validation sample index")
        plt.ylabel("Absolute error |y_true - y_pred|")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / out_name, format="pdf", bbox_inches="tight")
        plt.close()


def parser():
    parser = argparse.ArgumentParser(description="Train a logN regressor from embeddings + RDKit descriptors.")
    parser.add_argument("--train-ready", default="train-ready.dat")
    parser.add_argument("--features", default="features-chemberta.dat")
    parser.add_argument("--descriptors", default="rdkit-descriptors.dat")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-models", type=int, default=1)
    parser.add_argument("--outdir", default="training_outputs")
    parser.add_argument("--save-history", action="store_true", default=False)
    return parser

def main():
    args = parser().parse_args()

    train_ready_path = Path(args.train_ready)
    features_path = Path(args.features)
    descriptors_path = Path(args.descriptors)
    outdir = Path(args.outdir)

    train_df = pd.read_csv(train_ready_path, sep="\t")

    X_embed = np.loadtxt(features_path)
    X_embed = np.asarray(X_embed, dtype=np.float32)

    if X_embed.ndim == 0:
        X_embed = X_embed.reshape(1, 1)
    elif X_embed.ndim == 1:
        X_embed = X_embed.reshape(1, -1)

    X_phys = np.loadtxt(descriptors_path)
    X_phys = np.asarray(X_phys, dtype=np.float32)
    
    if X_phys.ndim == 0:
        X_phys = X_phys.reshape(1, 1)    
    
    y = train_df["logn"].to_numpy(dtype=np.float32)
    names = train_df["name"].astype(str).to_numpy()

    outdir.mkdir(parents=True, exist_ok=True)

    model_seeds = []
    split_seeds = []
    models = []
    val_metrics_per_model = []
    val_predictions_by_model = []
    emb_scalers = []
    phys_scalers = []

    for model_idx in range(args.n_models):
        seed = args.seed + model_idx

        model_seeds.append(seed)
        split_seeds.append(seed)

        set_global_seed(seed)

        Xe_train, Xe_val, Xp_train, Xp_val, y_train, y_val, idx_train, idx_val, emb_scaler, phys_scaler = preprocessing(
            X_embed=X_embed,
            X_phys=X_phys,
            y=y,
            test_size=args.test_size,
            random_state=seed)

        model, nn_metrics, y_pred_nn, history_dict = train_and_evaluate(
            Xe_train=Xe_train,
            Xe_val=Xe_val,
            Xp_train=Xp_train,
            Xp_val=Xp_val,
            y_train=y_train,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if args.save_history:
            hist_df = pd.DataFrame(history_dict)
            hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
            hist_df.to_csv(outdir / f"history_model_{model_idx}.csv", index=False)
            
        models.append(model)
        val_metrics_per_model.append(nn_metrics)
        val_predictions_by_model.append({
                "idx_train": idx_train,
                "idx_val": idx_val,
                "y_val": y_val,
                "y_pred_val": y_pred_nn})
        
        emb_scalers.append(emb_scaler)
        phys_scalers.append(phys_scaler)

        if args.n_models == 1:
            model_name = "model.keras"
        else:
            model_name = f"model_{model_idx + 1}.keras"
        model.save(outdir / model_name)

        if args.n_models == 1:
            joblib.dump(emb_scaler, outdir / "embedding_scaler.joblib")
            joblib.dump(phys_scaler, outdir / "descriptor_scaler.joblib")
        else:
            joblib.dump(emb_scaler, outdir / f"embedding_scaler_{model_idx + 1}.joblib")
            joblib.dump(phys_scaler, outdir / f"descriptor_scaler_{model_idx + 1}.joblib")

    y_pred_full_per_model: list[np.ndarray] = []
    full_metrics_per_model: list[dict[str, float]] = []
    for model, emb_scaler, phys_scaler in zip(models, emb_scalers, phys_scalers):
        Xe_full = emb_scaler.transform(X_embed)
        Xp_full = phys_scaler.transform(X_phys)
        y_pred_full = model.predict([Xe_full, Xp_full], verbose=0).reshape(-1)
        y_pred_full_per_model.append(y_pred_full)
        full_metrics_per_model.append(get_metrics(y, y_pred_full))

    val_pred_sum = np.zeros_like(y)
    val_pred_count = np.zeros_like(y)
    for val_record in val_predictions_by_model:
        idx_val = np.asarray(val_record["idx_val"])
        y_pred_val = np.asarray(val_record["y_pred_val"])
        val_pred_sum[idx_val] += y_pred_val
        val_pred_count[idx_val] += 1

    covered_val_mask = val_pred_count > 0
    y_val_ensemble = y[covered_val_mask]
    y_pred_nn = (val_pred_sum[covered_val_mask] / val_pred_count[covered_val_mask]).astype(np.float32) # with the mask we are able to determine the validation metrics only computed with the amount of times a sample was in the validation set
    y_pred_nn_full = np.mean(np.stack(y_pred_full_per_model, axis=0), axis=0)
    ensemble_val_metrics = get_metrics(y_val_ensemble, y_pred_nn)
    ensemble_full_metrics = get_metrics(y, y_pred_nn_full)

    metrics = {
        "validation": {
            "nn": summarize_metric_list(val_metrics_per_model),
            "ensemble_on_validation": ensemble_val_metrics,
        },
        "full_dataset": {
            "nn": summarize_metric_list(full_metrics_per_model),
            "ensemble_training": ensemble_full_metrics,
        },
        "ensemble": {
            "n_models": args.n_models,
            "base_seed": args.seed,
            "model_seeds": model_seeds,
            "split_seeds": split_seeds,
        },
    }
    with (outdir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {
            "sample_index": np.flatnonzero(covered_val_mask),
            "sample_name": names[covered_val_mask],
            "y_true": y_val_ensemble,
            "y_pred_nn": y_pred_nn,
            "n_predictions": val_pred_count[covered_val_mask],
        }
    ).to_csv(outdir / "validation_predictions.csv", sep=",", index=False)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(y)),
            "sample_name": names,
            "y_true": y,
            "y_pred_nn": y_pred_nn_full,
        }
    ).to_csv(outdir / "full_dataset_predictions.csv", sep=",", index=False)

    save_diagnostic_plots(
        y_val=y_val_ensemble,
        y_pred_val=y_pred_nn,
        y_full=y,
        y_pred_full=y_pred_nn_full,
        outdir=outdir,
        val_names=names[covered_val_mask],
        full_names=names,
    )

    print("Validation metrics (NN):")
    print(json.dumps(metrics["validation"], indent=2))
    print("Full-dataset metrics train+validation (NN):")
    print(json.dumps(metrics["full_dataset"], indent=2))
    print(f"Saved outputs in: {outdir}")


if __name__ == "__main__":
    main()
