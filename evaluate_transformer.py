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
import csv
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

from train import load_particle_datasets, PretrainingLoss, TwoBodyLoss, CombinedLoss, build_model

OUTPUT_DIR = Path("transformer_evaluation")


def output_file(output_path, suffix):
    """Return an evaluation artifact path below the shared output directory."""
    return OUTPUT_DIR / f"{output_path}_{suffix}"

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

###########################################################################
# Write model summary
###########################################################################

def write_model_summary(
    model,
    input_dim,
    num_particles,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
):
    
    device = next(model.parameters()).device

    # Input dimensions
    B = 2
    N = num_particles
    F = input_dim

    model_summary = summary(
        model._orig_mod if hasattr(model, "_orig_mod") else model,
        input_data=(
            torch.randn(B, N, F, device=device),                    # particles
            torch.rand(B, N, device=device),                        # puppi_pt
            torch.randn(B, N, device=device),                       # puppi_eta
            torch.randn(B, N, device=device),                       # puppi_phi
            torch.randn(B, N, device=device),                       # puppi_px
            torch.randn(B, N, device=device),                       # puppi_py
            torch.randn(B, N, device=device),                       # puppi_pz
            torch.rand(B, N, device=device),                        # puppi_E
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

    with output_file(output_path, "model_summary.txt").open("w") as f:
        f.write(str(model_summary))        

###########################################################################
# Save test metrics
###########################################################################

def save_test_metrics(
    test_metrics,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
    dataset_name="WbWb_4000_1000_slimmed_all_ak_constituents",
):
    
    torch.save(
        test_metrics,
        output_file(output_path, "test_metrics.pt"),
    )

    with output_file(output_path, "summary.txt").open("w") as f:

        f.write(f"Test metrics: {dataset_name}\n")
        f.write("====================\n\n")

        for key, value in test_metrics["test_losses"].items():
            f.write(f"{key:20s}: {value:.6f}\n")

###########################################################################
# Plot training losses
###########################################################################

def plot_losses(
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents"
):
    
    colors = {
        "loss":           "black",
        "ce_loss":        "tab:brown",
        "twobody_loss":   "tab:olive",
        "mass_loss":      "tab:blue",
        "entropy_loss":   "tab:red",
        "nonempty_loss":  "tab:green",
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

        if loss_name == "event_loss" or "loss" not in loss_name: continue;

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
            output_file(output_path, f"loss_{loss_name}.png"),
            dpi=200,
        )

        plt.close()

    plt.figure(figsize=(8,6))

    for loss_name in loss_names:

        if loss_name == "event_loss" or "loss" not in loss_name: continue;

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
        output_file(output_path, "loss_all_losses.png"),
        dpi=200,
    )

    plt.close()

    # Plot all scheduled loss weights together.  Student training has no
    # scheduled weights, so no weight plot is produced for that mode.
    weight_names = [name for name in loss_names if name.endswith("_weight")]

    if weight_names:
        plt.figure(figsize=(7, 5))

        for weight_name in weight_names:
            weight_curve = [x[weight_name] for x in loss_history["train_loss"]]
            plt.plot(epochs, weight_curve, label=weight_name, linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        plt.ylim(-0.05, 1.05)
        plt.title("Loss weights")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file(output_path, "loss_weights.png"), dpi=200)
        plt.close()

###########################################################################
# Confusion matrix helper
###########################################################################

def best_permutation(reference, labels):

    # Original
    score_original = np.sum(reference == labels)

    # Swap 1 <-> 2
    swapped = labels.copy()
    swapped[labels == 1] = 2
    swapped[labels == 2] = 1

    score_swapped = np.sum(reference == swapped)

    if score_swapped > score_original:
        return swapped
    else:
        return labels

###########################################################################
# Plot confusion matrices
###########################################################################

def plot_confusion_matrices(
    test_metrics,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
):
    
    mask             = test_metrics["data"]["mask"].numpy()

    algorlabels      = test_metrics["data"]["algorithmlabels"][mask].numpy()
    truthlabels      = test_metrics["data"]["truthlabels"][mask].numpy()
    translabels      = test_metrics["data"]["predictions"][mask].numpy()
    
    # Get best permutation of labels by aligning with truth convention
    algorlabels_best_against_truthlabels = best_permutation(truthlabels, algorlabels)
    truthlabels_best_against_translabels = best_permutation(translabels, truthlabels)
    translabels_best_against_algorlabels = best_permutation(algorlabels, translabels)

    algorvstruthconfusion = torch.zeros(3,3,dtype=torch.int64)
    truthvstransconfusion = torch.zeros(3,3,dtype=torch.int64)
    transvsalgorconfusion = torch.zeros(3,3,dtype=torch.int64)

    for algorlabel, truthlabel in zip(algorlabels_best_against_truthlabels, truthlabels):
        algorvstruthconfusion[algorlabel, truthlabel] += 1

    for truthlabel, translabel in zip(truthlabels_best_against_translabels, translabels):
        truthvstransconfusion[truthlabel, translabel] += 1

    for translabel, algorlabel in zip(translabels_best_against_algorlabels, algorlabels):
        transvsalgorconfusion[translabel, algorlabel] += 1

    algorvstruthconfusion = algorvstruthconfusion.float()
    truthvstransconfusion = truthvstransconfusion.float()
    transvsalgorconfusion = transvsalgorconfusion.float()

    algorvstruthconfusion /= (algorvstruthconfusion.sum(dim=1,keepdim=True) + 1e-8)
    truthvstransconfusion /= (truthvstransconfusion.sum(dim=1,keepdim=True) + 1e-8)
    transvsalgorconfusion /= (transvsalgorconfusion.sum(dim=1,keepdim=True) + 1e-8)

    for confusion, name in zip([algorvstruthconfusion, truthvstransconfusion, transvsalgorconfusion], 
                               ["Algo vs Truth", "Truth vs Transf", "Transf vs Algo"]):

        plt.figure(figsize=(6,6))

        plt.imshow(confusion, vmin=0., vmax=1.)

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

        plt.xlabel(name.split()[2])
        plt.ylabel(name.split()[0])
        plt.title(name)

        plt.savefig(
            output_file(output_path, f"confusion_matrix_{name}.png"),
            dpi=200,
        )

        plt.close()

###########################################################################
# Plot event displays
###########################################################################

def plot_event_display(
    test_metrics,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
):
    
    colors = [
        "tab:gray",
        "tab:blue",
        "tab:red",
    ]

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

            for pred, name in zip([pred, algo, truth], ["Transformer", "Algorithm", "Truth"]):

                plt.figure(figsize=(8,6))

                for pred_cls in range(3):

                    selection = (pred == pred_cls)

                    if selection.sum() == 0:
                        continue

                    plt.scatter(
                        phi_evt[selection],
                        eta_evt[selection],
                        s=sizes[selection],
                        c=colors[pred_cls],
                        marker="o",
                        edgecolors="black",
                        linewidths=0.3,
                        alpha=0.4,
                        label=f"{name} {legend_label[pred_cls]}",
                    )

                plt.xlabel(r"$\phi$")
                plt.ylabel(r"$\eta$")
                plt.xlim(-math.pi,math.pi)
                plt.title(f"Transformer {category} loss: {loss:.3f} ({name})")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                plt.savefig(
                    output_file(output_path, f"{category}_loss_{loss:.3f}_evt_{idx}_{name}.png"),
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

    avg_chi_masses = []
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

            masses.append(torch.sqrt(torch.clamp(m2, min=0)).item())

        avg_chi_masses.append(0.5 * (masses[0] + masses[1]))

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

    return avg_chi_masses, suu_masses


def assignment_metrics(labels, mask, pt, eta, phi, energy):
    """Return per-event chi masses and assignment fractions for hard labels."""
    labels, mask = labels.numpy(), mask.numpy()
    pt, eta, phi, energy = pt.numpy(), eta.numpy(), phi.numpy(), energy.numpy()
    chi0, chi1, suu, fractions, sjs = [], [], [], [], []
    for lab, keep, p, e, f, en in zip(labels, mask, pt, eta, phi, energy):
        lab, p, e, f, en = lab[keep], p[keep], e[keep], f[keep], en[keep]
        fractions.append([np.mean(lab == cls) if len(lab) else 0. for cls in range(3)])
        px, py, pz = p * np.cos(f), p * np.sin(f), p * np.sinh(e)
        vectors, event_sjs = [], []
        for cls in (1, 2):
            selected = lab == cls
            vector = np.array([px[selected].sum(), py[selected].sum(), pz[selected].sum(), en[selected].sum()])
            vectors.append(vector)
            mass = np.sqrt(max(vector[3] ** 2 - np.dot(vector[:3], vector[:3]), 0.))
            event_sjs.append((vector[3], np.hypot(vector[0], vector[1]), int(selected.sum()), mass))
        masses = [item[3] for item in event_sjs]
        chi0.append(masses[0]); chi1.append(masses[1])
        total = vectors[0] + vectors[1]
        suu.append(np.sqrt(max(total[3] ** 2 - np.dot(total[:3], total[:3]), 0.)))
        sjs.extend(event_sjs)
    return np.asarray(chi0), np.asarray(chi1), np.asarray(suu), np.asarray(fractions), sjs


def plot_assignment_suite(test_metrics, output_path):
    data = test_metrics["data"]
    inputs = (data["mask"], data["raw_pt"], data["raw_eta"], data["raw_phi"], data["raw_E"])
    truth = assignment_metrics(data["truthlabels"], *inputs)
    algorithm = assignment_metrics(data["algorithmlabels"], *inputs)
    transformer = assignment_metrics(data["predictions"], *inputs)
    methods = {"Truth": (truth, "tab:blue"), "Algorithm": (algorithm, "tab:orange"),
               "Transformer": (transformer, "tab:green")}

    for name, (metrics, color) in {"algorithm": (algorithm, "tab:orange"),
                                   "transformer": (transformer, "tab:green")}.items():
        plt.figure(figsize=(6, 6)); plt.scatter(metrics[0], metrics[1], s=5, alpha=.35, color=color)
        plt.xlabel(rf"{name.title()} $m(\chi_0)$ [GeV]"); plt.ylabel(rf"{name.title()} $m(\chi_1)$ [GeV]")
        plt.title(rf"{name.title()} $m(\chi_1)$ vs $m(\chi_0)$"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(output_file(output_path, f"{name}_chi_scatter.png"), dpi=200); plt.close()
        for index, xlabel, stem in [(0, "SJ energy [GeV]", "energy"), (1, r"SJ $p_T$ [GeV]", "pt"),
                                    (2, "SJ constituents", "nconstituents")]:
            plt.figure(figsize=(6, 4)); plt.hist([sj[index] for sj in metrics[4]], bins=60, histtype="step",
                                                 linewidth=2, color=color, label=name.title())
            plt.xlabel(xlabel); plt.ylabel("SJs"); plt.title(f"{name.title()} SJ {stem}"); plt.legend(); plt.grid(alpha=.3)
            plt.tight_layout(); plt.savefig(output_file(output_path, f"{name}_sj_{stem}.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 6))
    for name, (metrics, color) in methods.items():
        fractions = metrics[3]; x = fractions[:, 1] + .5 * fractions[:, 2]; y = np.sqrt(3)/2 * fractions[:, 2]
        plt.scatter(x, y, s=5, alpha=.25, color=color, label=name)
    plt.text(0, -.05, "background"); plt.text(1, -.05, r"$\chi_0$"); plt.text(.5, np.sqrt(3)/2+.03, r"$\chi_1$")
    plt.xlim(-.05, 1.05); plt.ylim(-.08, 1); plt.gca().set_aspect("equal"); plt.legend(); plt.title("Particle assignment fractions")
    plt.tight_layout(); plt.savefig(output_file(output_path, "assignment_triangle.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 5))
    for name, metrics, color in (("Algorithm", algorithm, "tab:orange"), ("Transformer", transformer, "tab:green")):
        plt.hist(metrics[2] - truth[2], bins=100, histtype="step", linewidth=2, color=color, label=name)
    plt.axvline(0, color="black", linewidth=1); plt.xlabel(r"$\Delta m_{Suu}$ relative to truth [GeV]"); plt.ylabel("Events")
    plt.title(r"$Suu$ mass residual"); plt.legend(); plt.tight_layout(); plt.savefig(output_file(output_path, "suu_mass_residual.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 5))
    for name, metrics, color in (("Algorithm", algorithm, "tab:orange"), ("Transformer", transformer, "tab:green")):
        plt.scatter(truth[2], (metrics[2] - truth[2]) / np.maximum(truth[2], 1.), s=5, alpha=.25, color=color, label=name)
    plt.axhline(0, color="black", linewidth=1); plt.xlabel(r"Truth $m_{Suu}$ [GeV]"); plt.ylabel("Relative mass residual")
    plt.title(r"$Suu$ mass resolution"); plt.legend(); plt.tight_layout(); plt.savefig(output_file(output_path, "suu_mass_resolution.png"), dpi=200); plt.close()

    rows = [("Truth", truth), ("Algorithm", algorithm), ("Transformer", transformer)]
    with output_file(output_path, "assignment_summary.txt").open("w") as handle:
        handle.write("method mean_mchi mean_mSuu bias_mSuu resolution_mSuu\n")
        for name, metrics in rows:
            residual = metrics[2] - truth[2]
            handle.write(f"{name} {np.mean(.5*(metrics[0]+metrics[1])):.3f} {np.mean(metrics[2]):.3f} "
                         f"{np.mean(residual):.3f} {np.std(residual):.3f}\n")


def robust_relative_resolution(masses):
    """Return (q84 - q16) / (2 * median)."""

    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[np.isfinite(masses)]
    if not masses.size:
        return float("nan")
    q16, median, q84 = np.percentile(masses, [16, 50, 84])
    if median <= 0.0:
        return float("nan")
    return float((q84 - q16) / (2.0 * median))


def summarize_reconstruction_epoch(model, loader, device):
    """Compute common reconstruction metrics for one epoch checkpoint."""

    model.eval()
    chi0_masses, chi1_masses = [], []
    mass_asymmetries, both_in_target, occupancies = [], [], []
    truth_chi0_masses, truth_chi1_masses = [], []

    with torch.no_grad():
        for batch in loader:
            particles = batch["particles"].to(device)
            mask = batch["mask"].to(device)
            pt = batch["puppi_pt"].to(device)
            eta = batch["puppi_eta"].to(device)
            phi = batch["puppi_phi"].to(device)
            px = batch["puppi_px"].to(device)
            py = batch["puppi_py"].to(device)
            pz = batch["puppi_pz"].to(device)
            energy = batch["puppi_E"].to(device)

            outputs = model(particles, pt, eta, phi, px, py, pz, energy, mask)
            probabilities = outputs["probabilities"]
            predictions = outputs["logits"].argmax(dim=-1)

            event_masses = []
            for cls in (1, 2):
                selected = (predictions == cls) & mask
                reco_px = (px * selected).sum(dim=1)
                reco_py = (py * selected).sum(dim=1)
                reco_pz = (pz * selected).sum(dim=1)
                reco_energy = (energy * selected).sum(dim=1)
                mass2 = reco_energy.square() - reco_px.square() - reco_py.square() - reco_pz.square()
                event_masses.append(torch.sqrt(torch.clamp(mass2, min=0.0)))

            mass0, mass1 = event_masses
            truth0 = batch["chi0"][:, 7].to(device)
            truth1 = batch["chi1"][:, 7].to(device)
            direct_match = (
                (mass0 >= 0.7 * truth0) & (mass0 <= 1.3 * truth0)
                & (mass1 >= 0.7 * truth1) & (mass1 <= 1.3 * truth1)
            )
            swapped_match = (
                (mass0 >= 0.7 * truth1) & (mass0 <= 1.3 * truth1)
                & (mass1 >= 0.7 * truth0) & (mass1 <= 1.3 * truth0)
            )

            valid_count = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            event_occupancy = (probabilities * mask.unsqueeze(-1)).sum(dim=1) / valid_count

            chi0_masses.append(mass0.cpu())
            chi1_masses.append(mass1.cpu())
            mass_asymmetries.append(
                (torch.abs(mass0 - mass1) / (mass0 + mass1 + 1.0e-8)).cpu()
            )
            both_in_target.append((direct_match | swapped_match).cpu())
            occupancies.append(event_occupancy.cpu())
            truth_chi0_masses.append(truth0.cpu())
            truth_chi1_masses.append(truth1.cpu())

    mass0 = torch.cat(chi0_masses).numpy()
    mass1 = torch.cat(chi1_masses).numpy()
    asymmetry = torch.cat(mass_asymmetries).numpy()
    target = torch.cat(both_in_target).float().numpy()
    occupancy = torch.cat(occupancies).numpy()
    truth0 = torch.cat(truth_chi0_masses).numpy()
    truth1 = torch.cat(truth_chi1_masses).numpy()

    return {
        "chi0_median_mass": float(np.median(mass0)),
        "chi1_median_mass": float(np.median(mass1)),
        "chi0_resolution": robust_relative_resolution(mass0),
        "chi1_resolution": robust_relative_resolution(mass1),
        "median_mass_asymmetry": float(np.median(asymmetry)),
        "fraction_both_in_target": float(np.mean(target)),
        "bkg_occupancy": float(np.mean(occupancy[:, 0])),
        "chi0_occupancy": float(np.mean(occupancy[:, 1])),
        "chi1_occupancy": float(np.mean(occupancy[:, 2])),
        "truth_chi0_median_mass": float(np.median(truth0)),
        "truth_chi1_median_mass": float(np.median(truth1)),
    }


def discover_epoch_checkpoints(output_path, checkpoint_dir="checkpoints"):
    """Return ordered (one-based epoch, checkpoint path) pairs."""

    pattern = re.compile(rf"^{re.escape(output_path)}_epoch(?P<epoch>\d+)\.pt$")
    checkpoints = []
    for path in Path(checkpoint_dir).glob(f"{output_path}_epoch*.pt"):
        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match["epoch"]) + 1, path))
    return sorted(checkpoints)


def plot_epoch_reconstruction_history(model, loader, device, output_path):
    """Evaluate and plot reconstruction behavior for every saved epoch."""

    checkpoints = discover_epoch_checkpoints(output_path)
    if not checkpoints:
        print(f"[WARNING] No numbered checkpoints found for {output_path}; skipping epoch plots")
        return

    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    history = []
    for epoch, checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        row = {"epoch": epoch}
        row.update(summarize_reconstruction_epoch(model, loader, device))
        history.append(row)
        print(f"[INFO] Evaluated reconstruction history epoch {epoch}")

    csv_path = output_file(output_path, "epoch_reconstruction.csv")
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = np.asarray([row["epoch"] for row in history])

    plt.figure(figsize=(7, 5))
    for key, label, color in (
        ("chi0_median_mass", r"Reco $\chi_0$", "tab:blue"),
        ("chi1_median_mass", r"Reco $\chi_1$", "tab:orange"),
    ):
        plt.plot(epochs, [row[key] for row in history], marker="o", label=label, color=color)
    plt.axhline(history[0]["truth_chi0_median_mass"], color="tab:blue", linestyle="--", alpha=.6,
                label=r"Truth $\chi_0$")
    plt.axhline(history[0]["truth_chi1_median_mass"], color="tab:orange", linestyle="--", alpha=.6,
                label=r"Truth $\chi_1$")
    plt.xlabel("Epoch"); plt.ylabel("Median mass [GeV]"); plt.title("Reconstructed chi mass vs epoch")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_file(output_path, "epoch_chi_median_mass.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["chi0_resolution"] for row in history], marker="o", label=r"$\chi_0$")
    plt.plot(epochs, [row["chi1_resolution"] for row in history], marker="o", label=r"$\chi_1$")
    plt.xlabel("Epoch"); plt.ylabel(r"$(q_{84}-q_{16})/(2\,\mathrm{median})$")
    plt.title("Robust relative chi mass resolution vs epoch")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_file(output_path, "epoch_chi_resolution.png"), dpi=200); plt.close()

    figure, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(epochs, [row["median_mass_asymmetry"] for row in history], marker="o")
    axes[0].set_ylabel(r"Median $|m_0-m_1|/(m_0+m_1)$")
    axes[0].set_title("Chi mass balance vs epoch")
    axes[1].plot(epochs, [row["fraction_both_in_target"] for row in history], marker="o", color="tab:green")
    axes[1].set(xlabel="Epoch", ylabel="Fraction of events", ylim=(0, 1.02))
    axes[1].set_title(r"Both chi masses within $0.7$--$1.3\times$ truth")
    for axis in axes:
        axis.grid(alpha=.3)
    figure.tight_layout(); figure.savefig(output_file(output_path, "epoch_mass_quality.png"), dpi=200); plt.close(figure)

    plt.figure(figsize=(7, 5))
    for key, label, color in (
        ("bkg_occupancy", "Background", "tab:gray"),
        ("chi0_occupancy", r"$\chi_0$", "tab:blue"),
        ("chi1_occupancy", r"$\chi_1$", "tab:orange"),
    ):
        plt.plot(epochs, [row[key] for row in history], marker="o", label=label, color=color)
    plt.xlabel("Epoch"); plt.ylabel("Mean probability-weighted particle fraction")
    plt.ylim(0, 1); plt.title("Predicted occupancy vs epoch")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_file(output_path, "epoch_occupancy.png"), dpi=200); plt.close()

###########################################################################
# Plot mass peaks
###########################################################################

def plot_mass_peaks(
    test_metrics,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
    trainMode="student",
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
        label="Truth", color="tab:blue",
    )

    plt.hist(
        algo_chi,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Algorithm", color="tab:orange",
    )

    plt.hist(
        pred_chi,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Transformer", color="tab:green",
    )

    plt.xlabel("Average reconstructed chi mass [GeV]")
    plt.ylabel("Events")
    plt.title(f"Average chi mass ({trainMode} transformer)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        output_file(output_path, "mass_chi.png"),
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
        label="Truth", color="tab:blue",
    )

    plt.hist(
        algo_suu,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Algorithm", color="tab:orange",
    )

    plt.hist(
        pred_suu,
        bins=100,
        histtype="step",
        linewidth=2,
        label="Transformer", color="tab:green",
    )

    plt.xlabel("Reconstructed Suu mass [GeV]")
    plt.ylabel("Events")
    plt.title(f"Suu Mass ({trainMode} transformer)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        output_file(output_path, "mass_suu.png"),
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
    trainMode="student",
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

            raw_pt = batch["puppi_pt"].to(device)
            raw_eta = batch["puppi_eta"].to(device)
            raw_phi = batch["puppi_phi"].to(device)
            raw_px = batch["puppi_px"].to(device)
            raw_py = batch["puppi_py"].to(device)
            raw_pz = batch["puppi_pz"].to(device)
            raw_E = batch["puppi_E"].to(device)

            algorithm_labels = batch["algorithmLabel"].to(device)
            algorithm_CA8labels = batch["algorithmCAIndex"].to(device)
            truthlabels = batch["truthLabel"].to(device)

            outputs = model(
                particles,
                raw_pt,
                raw_eta,
                raw_phi,
                raw_px,
                raw_py,
                raw_pz,
                raw_E,
                mask,
            )

            logits = outputs["logits"]
            probabilities = outputs["probabilities"]
            predictions = logits.argmax(dim=-1)

            if trainMode == "student":
                losses = criterion(
                    logits,
                    algorithm_labels,
                )
            elif trainMode == "student_to_scratch":
                losses = criterion(
                    9,  # final epoch
                    10, # num_epochs, want to show final losses
                    logits,
                    probabilities,
                    raw_px,
                    raw_py,
                    raw_pz,
                    raw_E,
                    algorithm_labels,
                    algorithm_CA8labels,
                    mask,
                )
            else:
                losses = criterion(
                    probabilities,
                    raw_px,
                    raw_py,
                    raw_pz,
                    raw_E,
                    algorithm_CA8labels,
                    mask,
                    9,  # final epoch: use the fully ramped entropy weight
                    10,
                )

            batch_size = particles.shape[0]

            for key, value in losses.items():

                # Skip non-tensor metadata
                if "weight" in key:
                    total_losses[key] = value
                    continue

                # Per-event vector
                if value.ndim > 0:
                    continue

                # Scalar already averaged over batch
                else:
                    total_losses[key] = total_losses.get(key, 0.0) + value.item() * batch_size

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

        if "weight" in key or key == "event_loss":
            continue

        total_losses[key] /= total_events

    num_events = len(event_losses)
    num_worst = min(num_worst, num_events)
    num_best = min(num_best, num_events)
    idx_worst = torch.topk(event_losses, num_worst, largest=True).indices
    idx_best = torch.topk(event_losses, num_best, largest=False).indices

    idx_worst = idx_worst[torch.argsort(event_losses[idx_worst], descending=True)]
    idx_best = idx_best[torch.argsort(event_losses[idx_best], descending=False)]

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
    batch_size=32,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
    trainMode="student",
    num_worst=10,
    num_best=10,
    pt_dir="ptfiles",
):

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_particle_datasets(
        dataset_path,
        pt_dir=pt_dir,
    )

    if len(test_dataset) == 0:
        raise ValueError(f"Dataset {dataset_path!r} has an empty test split")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Finished loading data")

    # Determine input feature dimension
    sample = test_dataset[0]

    input_dim = sample["particles"].shape[-1]

    # Model
    tmp_path = f"checkpoints/{output_path}.pt"

    checkpoint_path = tmp_path

    feature_names = test_dataset.metadata["features"]["particle_names"]
    model = build_model(
        input_dim=input_dim,
        checkpoint_path=checkpoint_path,
        feature_names=feature_names,
        load_checkpoint=True,
    )

    model = model.to(device)

    model = torch.compile(model)

    print("Finished loading model")

    # Loss
    if trainMode == "student":
        criterion = PretrainingLoss()
    elif trainMode == "student_to_scratch":
        criterion = CombinedLoss()
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
    write_model_summary(
        model,
        input_dim=input_dim,
        num_particles=sample["particles"].shape[0],
        output_path=output_path,
    )
    print("Finished writing model summary")

    # Save losses, write summary
    save_test_metrics(test_metrics, output_path, dataset_path)
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

    plot_assignment_suite(test_metrics, output_path)
    print("Finished plotting assignment diagnostics")

    plot_epoch_reconstruction_history(model, test_loader, device, output_path)
    print("Finished plotting reconstruction metrics over epochs")

###########################################################################
# Main
###########################################################################

def main(args):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainModes = (
        ["student", "from_scratch", "student_to_scratch"]
        if args.trainMode == "all"
        else [args.trainMode]
    )

    jobs = []

    for trainMode in trainModes:
        jobs.append({
            "dataset_path": args.input,
            "output_path": f"{args.output}_{trainMode}",
            "trainMode": trainMode,
        })

    for job in jobs:
        
        print("------------------Evaluating------------------")
        print("Dataset path: " + job["dataset_path"])
        print("Output path: " + job["output_path"])
        print("Train mode: " + job["trainMode"])

        evaluate(
            dataset_path=job["dataset_path"],
            pt_dir=args.pt_dir,
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
        default="WbWb_4000_1000_slimmed_all_ak_constituents",
        help="Input preprocessed filename stub",
    )

    parser.add_argument(
        "--pt-dir",
        type=Path,
        default=Path("ptfiles"),
        help="Directory containing the preprocessed .pt shards",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="WbWb_4000_1000_slimmed_all_ak_constituents",
        help="Output checkpoint file",
    )

    parser.add_argument(
        "--trainMode",
        choices=[
            "from_scratch",
            "student",
            "student_to_scratch",
            "all"
        ],
        default="from_scratch",
        help="Choose student, from_scratch, student_to_scratch, or all",
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
