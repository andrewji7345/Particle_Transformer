"""
evaluate.py

Particle Transformer evaluation for Suu -> chichi reconstruction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

from train import load_particle_datasets, PretrainingLoss, TwoBodyLoss, build_model

###########################################################################
# Model hyperparameters
###########################################################################

EMBED_DIM = 128

NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS

NUM_LAYERS = 8

MLP_RATIO = 4
DROPOUT = 0.1

INTERACTION_DIM = 16
NUM_CLASSES = 3

LAMBDA_MASS      = 1.0
LAMBDA_ENTROPY   = 0.2
LAMBDA_OCCUPANCY = 0.5
LAMBDA_SPLIT     = 1.0
LAMBDA_BKG       = 0.5

###########################################################################
# Write model summary
###########################################################################

def write_model_summary(
    model,
    output_path="test_model"
):
    
    device = next(model.parameters()).device

    # Input dimensions
    B = 2
    N = 128
    F = 13

    model_summary = summary(
        model._orig_mod,
        input_data=(
            torch.randn(B, N, F, device=device),                    # particles
            torch.rand(B, N, device=device),                        # raw_pt
            torch.randn(B, N, device=device),                       # raw_eta
            torch.randn(B, N, device=device),                       # raw_phi
            torch.rand(B, N, device=device),                        # raw_E
            torch.ones(B, N, dtype=torch.bool, device=device),      # mask
        ),
        depth=10,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "trainable",
        ),
        row_settings=("depth", "var_names"),
        verbose=0,
    )

    with open(f"evaluation/{output_path}_model_summary.txt", "w") as f:
        f.write(str(model_summary))        

###########################################################################
# Save test metrics
###########################################################################

def save_test_metrics(
    test_metrics,
    output_path="test_model"
):
    
    torch.save(
        test_metrics,
        "evaluation/" + output_path + "_test_metrics.pt",
    )

    with open(f"evaluation/{output_path}_summary.txt", "w") as f:

        f.write("Test metrics\n")
        f.write("====================\n\n")

        for key, value in test_metrics["test_losses"].items():
            f.write(f"{key:20s}: {value:.6f}\n")

###########################################################################
# Plot training losses
###########################################################################

def plot_losses(
    output_path="test_model"
):
    
    colors = {
        "loss":           "black",
        "mass_loss":      "tab:blue",
        "entropy_loss":   "tab:red",
        "occupancy_loss": "tab:green",
        "split_loss":     "tab:orange",
        "bkg_loss":       "tab:gray",
    }
    
    loss_history = torch.load(
        f"checkpoints/{output_path}_losses.pt",
        map_location="cpu",
    )

    epochs     = loss_history["epoch"].numpy()
    loss_names = loss_history["train_loss"][0].keys()
    
    for loss_name in loss_names:

        if loss_name == "event_loss": continue;

        train_curve = [x[loss_name] for x in loss_history["train_loss"]]
        val_curve   = [x[loss_name] for x in loss_history["val_loss"]]

        plt.figure(figsize=(6,4))

        plt.plot(
            epochs,
            train_curve,
            label="Train",
            color = colors[loss_name],
            ls = '-',
        )

        plt.plot(
            epochs,
            val_curve,
            label="Validation",
            color = colors[loss_name],
            ls = '--',
        )

        plt.xlabel("Epoch")
        plt.ylabel(loss_name)
        plt.title(loss_name)
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"plots/{output_path}_{loss_name}.png",
            dpi=200,
        )

        plt.close()

    plt.figure(figsize=(8,6))

    for loss_name in loss_names:

        if loss_name == "event_loss": continue;

        train_curve = [x[loss_name] for x in loss_history["train_loss"]]
        val_curve   = [x[loss_name] for x in loss_history["val_loss"]]

        plt.plot(
            epochs,
            train_curve,
            label=f"{loss_name} (train)",
            color = colors[loss_name],
            ls = '-',
        )

        plt.plot(
            epochs,
            val_curve,
            label=f"{loss_name} (validation)",
            color = colors[loss_name],
            ls = '--',
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("All losses")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"plots/{output_path}_all_losses.png",
        dpi=200,
    )

    plt.close()

###########################################################################
# Plot confusion matrices
###########################################################################

def plot_confusion_matrices(
    test_metrics,
    output_path="test_model",
):
    
    mask            = test_metrics["data"]["mask"].numpy()

    algorithmlabels = test_metrics["data"]["algorithmlabels"][mask].numpy()
    truthlabels     = test_metrics["data"]["truthlabels"][mask].numpy()
    preds           = test_metrics["data"]["predictions"][mask].numpy()
    
    algoconfusion = torch.zeros(3,3,dtype=torch.int64)
    tranconfusion = torch.zeros(3,3,dtype=torch.int64)
    compconfusion = torch.zeros(3,3,dtype=torch.int64)

    for truthlabel, pred in zip(truthlabels, preds):
        tranconfusion[truthlabel, pred] += 1
    
    for truthlabel, algorithmlabel in zip(truthlabels, algorithmlabels):
        algoconfusion[truthlabel, algorithmlabel] += 1

    for algorithmlabel, pred in zip(algorithmlabels, preds):
        compconfusion[algorithmlabel, pred] += 1

    tranconfusion = tranconfusion.float()
    algoconfusion = algoconfusion.float()
    compconfusion = compconfusion.float()

    tranconfusion /= (tranconfusion.sum(dim=1,keepdim=True) + 1e-8)
    algoconfusion /= (algoconfusion.sum(dim=1,keepdim=True) + 1e-8)
    compconfusion /= (compconfusion.sum(dim=1,keepdim=True) + 1e-8)

    for confusion, name in zip([tranconfusion, algoconfusion, compconfusion], ["tran", "algo", "comp"]):

        plt.figure(figsize=(5,5))

        plt.imshow(confusion)

        plt.colorbar()

        classes = ["bkg", "χ0", "χ1"]

        plt.xticks(range(3),classes)
        plt.yticks(range(3),classes)

        for i in range(3):
            for j in range(3):
                plt.text(
                    j,
                    i,
                    f"{confusion[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if confusion[i,j] > 0.5 else "black",
                )

        plt.tight_layout()

        if (name == "algo"):
            plt.xlabel("Algorithm")
            plt.ylabel("Truth")
            plt.title("Truth vs algorithm confusion")
        elif (name == "tran"):
            plt.xlabel("Transformer")
            plt.ylabel("Truth")
            plt.title("Truth vs transformer confusion")
        else:
            plt.xlabel("Transformer")
            plt.ylabel("Algorithm")
            plt.title("Transformer vs algorithm confusion")

        plt.savefig(
            f"plots/{output_path}_{name}_confusion_matrix.png",
            dpi=200,
        )

        plt.close()

###########################################################################
# Plot event displays
###########################################################################

def plot_event_display(
    test_metrics,
    output_path="test_model",
):
    
    truth_colors = [
        "tab:gray",
        "tab:blue",
        "tab:red",
    ]

    markers = {
        0: "P",
        1: "s",
        2: "o",
    }

    legend_label = ["bkg", "chi0", "chi1"]
    
    worst_event_indices  = test_metrics["idx"]["worst_event_idx"]
    worst_event_losses   = test_metrics["idx"]["worst_event_losses"]
    best_event_indices   = test_metrics["idx"]["best_event_idx"]
    best_event_losses    = test_metrics["idx"]["best_event_losses"]

    mask                 = test_metrics["data"]["mask"].numpy()

    algorithmlabels      = test_metrics["data"]["algorithmlabels"].numpy()
    truthlabels          = test_metrics["data"]["truthlabels"].numpy()
    preds                = test_metrics["data"]["predictions"].numpy()

    pt                   = test_metrics["data"]["raw_pt"].numpy()
    eta                  = test_metrics["data"]["raw_eta"].numpy()
    phi                  = test_metrics["data"]["raw_phi"].numpy()

    for event_indices, event_losses, category in zip(
        [worst_event_indices, best_event_indices],
        [worst_event_losses, best_event_losses],
        ["worst", "best"],
        ):

        for idx, loss in zip(event_indices, event_losses):

            algo = algorithmlabels[idx][mask[idx]]
            truth = truthlabels[idx][mask[idx]]
            pred = preds[idx][mask[idx]]

            pt_evt = pt[idx][mask[idx]]
            eta_evt = eta[idx][mask[idx]]
            phi_evt = phi[idx][mask[idx]]

            sizes = 20 + 15 * np.sqrt(pt_evt)

            for pred, name in zip([pred, algo], ["tran", "algo"]):

                plt.figure(figsize=(8,6))

                for truth_cls in range(3):

                    for pred_cls in range(3):

                        selection = ((truth == truth_cls) & (pred == pred_cls))

                        if selection.sum() == 0:
                            continue

                        plt.scatter(
                            phi_evt[selection],
                            eta_evt[selection],
                            s=sizes[selection],
                            c=truth_colors[truth_cls],
                            marker=markers[pred_cls],
                            edgecolors="black",
                            linewidths=0.3,
                            alpha=0.4,
                            label=f"Truth {legend_label[truth_cls]}, {name} {legend_label[pred_cls]}",
                        )

                plt.xlabel(r"$\phi$")
                plt.ylabel(r"$\eta$")
                plt.xlim(-math.pi,math.pi)
                plt.title(f"{name} {category} loss: {loss:.3f}")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                plt.savefig(
                    f"plots/{output_path}_{name}_{category}_loss_{loss:.3f}_evt_{idx}.png",
                    dpi=200,
                )

                plt.close()

###########################################################################
# Helper for reconstructing mass peaks
###########################################################################

def reconstruct(
    labels,
    mask,
    raw_pt,
    raw_eta,
    raw_phi,
    raw_E,
):

    B = labels.shape[0]

    chi_masses = []
    suu_masses = []

    for b in range(B):

        label = labels[b][mask[b]]

        pt = raw_pt[b][mask[b]]
        eta = raw_eta[b][mask[b]]
        phi = raw_phi[b][mask[b]]
        E = raw_E[b][mask[b]]

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        chi_fourvecs = []

        for cls in [1, 2]:

            sel = (label == cls)

            if sel.sum() == 0:
                chi_fourvecs.append(None)
                continue

            px_sum = px[sel].sum()
            py_sum = py[sel].sum()
            pz_sum = pz[sel].sum()
            E_sum = E[sel].sum()

            chi_fourvecs.append(
                (
                    px_sum,
                    py_sum,
                    pz_sum,
                    E_sum,
                )
            )

        if chi_fourvecs[0] is None or chi_fourvecs[1] is None:
            continue

        masses = []

        for vec in chi_fourvecs:

            px_sum, py_sum, pz_sum, E_sum = vec

            m2 = (
                E_sum**2
                - px_sum**2
                - py_sum**2
                - pz_sum**2
            )

            masses.append(
                torch.sqrt(
                    torch.clamp(m2, min=0)
                ).item()
            )

        chi_masses.append(
            0.5 * (masses[0] + masses[1])
        )

        px_sum = chi_fourvecs[0][0] + chi_fourvecs[1][0]
        py_sum = chi_fourvecs[0][1] + chi_fourvecs[1][1]
        pz_sum = chi_fourvecs[0][2] + chi_fourvecs[1][2]
        E_sum = chi_fourvecs[0][3] + chi_fourvecs[1][3]

        m2 = (
            E_sum**2
            - px_sum**2
            - py_sum**2
            - pz_sum**2
        )

        suu_masses.append(
            torch.sqrt(
                torch.clamp(m2, min=0)
            ).item()
        )

    return chi_masses, suu_masses

###########################################################################
# Plot mass peaks
###########################################################################

def plot_mass_peaks(
    test_metrics,
    output_path="test_model",
    trainMode="pretrained",
):

    mask = test_metrics["data"]["mask"]

    raw_pt = test_metrics["data"]["raw_pt"]
    raw_eta = test_metrics["data"]["raw_eta"]
    raw_phi = test_metrics["data"]["raw_phi"]
    raw_E = test_metrics["data"]["raw_E"]

    truthlabels = test_metrics["data"]["truthlabels"]
    algorithmlabels = test_metrics["data"]["algorithmlabels"]
    predictions = test_metrics["data"]["predictions"]

    truth_chi, truth_suu = reconstruct(
        truthlabels, 
        mask, 
        raw_pt, 
        raw_eta, 
        raw_phi, 
        raw_E)
    
    algo_chi, algo_suu = reconstruct(
        algorithmlabels, 
        mask, 
        raw_pt, 
        raw_eta, 
        raw_phi, 
        raw_E)
    
    pred_chi, pred_suu = reconstruct(
        predictions, 
        mask, 
        raw_pt, 
        raw_eta, 
        raw_phi, 
        raw_E)

    # Average reco chi mass distribution
    plt.figure(figsize=(6,4))

    plt.hist(
        truth_chi,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Truth",
    )

    plt.hist(
        algo_chi,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Algorithm",
    )

    plt.hist(
        pred_chi,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Transformer",
    )

    plt.xlabel("Average reconstructed chi mass [GeV]")
    plt.ylabel("Events")
    plt.title(f"Average chi mass ({trainMode} transformer)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        f"plots/{output_path}_chi_mass.png",
        dpi=200,
    )

    plt.close()

    # Reco suu mass distribution
    plt.figure(figsize=(6,4))

    plt.hist(
        truth_suu,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Truth",
    )

    plt.hist(
        algo_suu,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Algorithm",
    )

    plt.hist(
        pred_suu,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Transformer",
    )

    plt.xlabel("Reconstructed Suu mass [GeV]")
    plt.ylabel("Events")
    plt.title(f"Suu Mass ({trainMode} transformer)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        f"plots/{output_path}_suu_mass.png",
        dpi=200,
    )

    plt.close()

###########################################################################
# Testing loop
###########################################################################

def test(
    model,
    loader,
    criterion,
    device,
    trainMode="pretrain",
    num_worst=10,
    num_best=10,
):
    """
    Evaluate model on validation set.
    """

    model.eval()

    total_losses = {}
    total_events = 0

    all_event_losses = []

    all_masks = []

    all_probabilities = []
    all_predictions = []
    all_algorithmCA8labels = []
    all_algorithmlabels = []
    all_truthlabels = []

    all_raw_pt = []
    all_raw_eta = []
    all_raw_phi = []
    all_raw_E = []

    with torch.no_grad():

        desc = "Testing"

        for batch in loader:

            particles = batch["particles"].to(device)
            mask = batch["mask"].to(device)

            raw_pt = batch["raw_pt"].to(device)
            raw_eta = batch["raw_eta"].to(device)
            raw_phi = batch["raw_phi"].to(device)
            raw_E = batch["raw_E"].to(device)

            algorithm_labels = batch["algorithmLabel"].to(device)
            algorithm_CA8labels = batch["algorithmCA8Label"].to(device)
            truthlabels = batch["truthLabel"].to(device)

            outputs = model(
                particles,
                raw_pt,
                raw_eta,
                raw_phi,
                raw_E,
                mask,
            )

            logits = outputs["logits"]
            probabilities = outputs["probabilities"]
            predictions = logits.argmax(dim=-1)

            if trainMode == "pretrain":
                losses = criterion(
                    logits,
                    algorithm_labels,
                )
            else:
                losses = criterion(
                    probabilities,
                    raw_pt,
                    raw_eta,
                    raw_phi,
                    raw_E,
                    algorithm_CA8labels,
                    mask,
                )

            batch_size = particles.shape[0]

            for key in losses:
                if key not in total_losses:
                    total_losses[key] = losses[key].mean().item()
                else:
                    total_losses[key] += losses[key].mean().item()

            total_events += batch_size

            all_event_losses.append(losses["event_loss"].cpu())
                    
            all_masks.append(mask.cpu())

            all_probabilities.append(probabilities.cpu())
            all_predictions.append(predictions.cpu())
            all_algorithmCA8labels.append(algorithm_CA8labels.cpu())
            all_algorithmlabels.append(algorithm_labels.cpu())
            all_truthlabels.append(truthlabels.cpu())

            all_raw_pt.append(raw_pt.cpu())
            all_raw_eta.append(raw_eta.cpu())
            all_raw_phi.append(raw_phi.cpu())
            all_raw_E.append(raw_E.cpu())

    event_losses = torch.cat(all_event_losses)

    mask = torch.cat(all_masks)

    probabilities = torch.cat(all_probabilities)
    predictions = torch.cat(all_predictions)
    algorithmCA8labels = torch.cat(all_algorithmCA8labels)
    algorithmlabels = torch.cat(all_algorithmlabels)
    truthlabels = torch.cat(all_truthlabels)

    raw_pt = torch.cat(all_raw_pt)
    raw_eta = torch.cat(all_raw_eta)
    raw_phi = torch.cat(all_raw_phi)
    raw_E = torch.cat(all_raw_E)
    
    for key in total_losses:
        total_losses[key] /= total_events

    idx_worst = np.argpartition(event_losses, -num_worst)[-num_worst:]
    idx_best  = np.argpartition(event_losses, num_best - 1)[:num_best]

    idx_worst = idx_worst[torch.argsort(event_losses[idx_worst], descending=True)]
    idx_best  = idx_best[torch.argsort(event_losses[idx_best], descending=False)]

    test_metrics = {
        "test_losses": total_losses,
        "data": {
            "mask": mask,

            "probabilities": probabilities,
            "predictions": predictions,
            "algorithmCA8labels": algorithmCA8labels,
            "algorithmlabels": algorithmlabels,
            "truthlabels": truthlabels,

            "raw_pt": raw_pt,
            "raw_eta": raw_eta,
            "raw_phi": raw_phi,
            "raw_E": raw_E,
        },
        "idx": {
            "worst_event_idx": idx_worst,
            "worst_event_losses": event_losses[idx_worst],
            "best_event_idx": idx_best,
            "best_event_losses": event_losses[idx_best],
        }
    }

    return test_metrics

###########################################################################
# Main evaluation function
###########################################################################

def evaluate(
    dataset_path,
    batch_size=64,
    output_path="test_model",
    trainMode="pretrain",
    num_worst=10,
    num_best=10,
):

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_particle_datasets(dataset_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print("Finished loading data")

    # Determine input feature dimension
    sample = test_dataset[0]

    input_dim = sample["particles"].shape[-1]

    # Model
    tmp_path = f"checkpoints/{output_path}.pt"

    if trainMode == "use_pretrained":
        checkpoint_path = tmp_path.replace(
            "_use_pretrained",
            "_pretrain",
        )
    else:
        checkpoint_path = tmp_path

    model = build_model(
        input_dim=input_dim,
        trainMode=trainMode,
        checkpoint_path=checkpoint_path,
    )

    model = model.to(device)

    model = torch.compile(model)

    print("Finished loading model")

    # Loss
    if trainMode == "pretrain":
        criterion = PretrainingLoss()
    else:
        criterion = TwoBodyLoss()

    # Test
    test_metrics = test(
        model,
        test_loader,
        criterion,
        device,
        trainMode,
        num_worst,
        num_best,
    )

    print("Finished testing")

    # Write model summary
    write_model_summary(model, output_path)
    print("Finished writing model summary")

    # Save losses, write summary
    save_test_metrics(test_metrics, output_path)
    print("Finished saving test metrics")

    # Plot losses
    plot_losses(output_path)
    print("Finished plotting loss curves")

    # Plot confusion matrices
    plot_confusion_matrices(test_metrics, output_path)
    print("Finished plotting confusion matrices")

    # Plot event displays
    plot_event_display(test_metrics, output_path)
    print("Finished plotting event displays")

    # Plot mass peaks
    plot_mass_peaks(test_metrics, output_path, trainMode)
    print("Finished plotting mass peaks")

###########################################################################
# Main
###########################################################################

def main(args):

    Path("evaluation").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    # Assign data modes, pretraining modes
    dataModes = (
        ["all_pf", "ak8_constituents", "ak4_constituents", "all_constituents"]
        if args.dataMode == "all"
        else [args.dataMode]
    )

    trainModes = (
        ["pretrain", "use_pretrained", "no_use_pretrained"]
        if args.trainMode == "all"
        else [args.trainMode]
    )

    jobs = []

    for dataMode in dataModes:
        for trainMode in trainModes:
            jobs.append({
                "dataset_path": f"{args.input}_{dataMode}",
                "output_path": f"{args.output}_{dataMode}_{trainMode}",
                "trainMode": trainMode,
            })

    for job in jobs:
        
        print("------------------Evaluating------------------")
        print("Dataset path: " + job["dataset_path"])
        print("Output path: " + job["output_path"])
        print("Train mode: " + job["trainMode"])

        evaluate(
            dataset_path=job["dataset_path"],
            batch_size=args.batch_size,
            output_path=job["output_path"],
            trainMode=job["trainMode"],
            num_worst=args.num_worst,
            num_best=args.num_best,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate a Particle Transformer."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="WbWb_4000_1000",
        help="Input preprocessed filename stub",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="test_model",
        help="Output checkpoint file",
    )

    parser.add_argument(
        "--dataMode",
        choices=[
            "all_pf",
            "ak8_constituents",
            "ak4_constituents",
            "all_constituents",
            "all",
        ],
        default="ak8_constituents",
        help="Train with all_pf, ak8_constituents, ak4_constituents, all_constituents, or all",
    )

    parser.add_argument(
        "--trainMode",
        choices=[
            "no_use_pretrained",
            "use_pretrained",
            "pretrain",
            "all"
        ],
        default="no_use_pretrained",
        help="Either pretrain, use_pretrained, no_use_pretrained, or all",
    )

    parser.add_argument(
        "--num-worst",
        type=int,
        default=2,
        help="How many of the worst events to plot in the event display",
    )

    parser.add_argument(
        "--num-best",
        type=int,
        default=2,
        help="How many of the best events to plot in the event display",
    )

    args = parser.parse_args()

    main(args)