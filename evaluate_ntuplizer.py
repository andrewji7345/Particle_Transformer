"""
evaluate_ntuplizer.py

Validation script for the ParticleTransformer ntuplizer.

Performs sanity checks on the ntuplizer before preprocessing/training.

Checks:
- Event statistics
- Truth-label consistency
- Algorithm-label consistency
- Four-vector reconstruction
- Energy accounting
- Mass reconstruction
- Diagnostic plots
"""

import os
import argparse

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

################################################################################
# Argument parser
################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    default="/eos/uscms/store/user/aji/rootfile_ParticleTransformer/WbWb_4000_1000.root",
    help="Input ROOT ntuple",
)

parser.add_argument(
    "--output",
    default="evaluate_ntuplizer/",
    help="Directory for plots/printed results",
)

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

################################################################################
# Load tree
################################################################################

print("Opening file...")

file = uproot.open(args.input)
tree = file["Events"]

print(f"Loaded {tree.num_entries:,} events")

################################################################################
# Read branches
################################################################################

branches = tree.arrays(library="ak")

print(f"Loaded {len(branches.fields)} branches")

################################################################################
# Helper functions
################################################################################

def make_p4(px, py, pz, E):
    """
    Construct four-vectors stored as simple dictionaries.
    """

    return {
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    }

def invariant_mass(px, py, pz, E):
    """
    Calculate invariant mass.

    Works on scalars, numpy arrays, or awkward arrays.
    """

    m2 = E**2 - px**2 - py**2 - pz**2

    return np.sqrt(np.maximum(m2, 0.0))

################################################################################
# Quick summary
################################################################################

print("\n")
print("=" * 80)
print("NTUPLIZER SUMMARY")

print(f"Number of events : {tree.num_entries:,}")

print(f"Average PF candidates/event : "
      f"{ak.mean(branches['nParticles']):.1f}")

print(f"Maximum PF candidates/event : "
      f"{ak.max(branches['nParticles'])}")

print(f"Minimum PF candidates/event : "
      f"{ak.min(branches['nParticles'])}")

print("\nBranch names:")

for name in sorted(branches.fields):
    print("   ", name)

print("=" * 80)

################################################################################
# Four-vector reconstruction helpers
################################################################################

def sum_p4(mask, px, py, pz, E):
    """
    Sum the four-vectors of all particles passing a boolean mask.

    Returns
    -------
    px, py, pz, E
    """

    return (
        ak.sum(px[mask]),
        ak.sum(py[mask]),
        ak.sum(pz[mask]),
        ak.sum(E[mask]),
    )

def event_mass(mask, px, py, pz, E):
    """
    Reconstruct invariant mass from selected particles.
    """

    px_sum, py_sum, pz_sum, E_sum = sum_p4(mask, px, py, pz, E)

    return invariant_mass(px_sum, py_sum, pz_sum, E_sum)

################################################################################
# Build simple truth labels
################################################################################

def simplify_truth_labels(truth_labels):
    """
    Convert the ntuplizer truth labels into

        0 = background / ISR
        1 = chi0
        2 = chi1

    The ntuplizer stores detailed decay labels

        1-10   -> chi0 decay chain
        11-20  -> chi1 decay chain

    A particle with labels from both chains is considered ambiguous and is
    assigned to background for now.
    """

    simple = np.zeros(len(truth_labels), dtype=np.int32)

    for i, labels in enumerate(truth_labels):

        labels = list(labels)

        has_chi0 = any(1 <= x <= 10 for x in labels)
        has_chi1 = any(11 <= x <= 20 for x in labels)

        if has_chi0 and not has_chi1:
            simple[i] = 1

        elif has_chi1 and not has_chi0:
            simple[i] = 2

        else:
            simple[i] = 0

    return simple

################################################################################
# Event reconstruction
################################################################################

def reconstruct_event(event):
    """
    Reconstruct masses for one event.

    Returns a dictionary containing

        total PF mass
        truth Suu mass
        truth chi masses
        algorithm Suu mass
        algorithm chi masses
    """

    px = event["particle_px"]
    py = event["particle_py"]
    pz = event["particle_pz"]
    E  = event["particle_energy"]

    # Labels
    truth = simplify_truth_labels(event["particle_truthLabel"])
    algorithm = np.asarray(event["particle_algorithmLabel"])

    # Total PF system
    all_mask = np.ones(len(px), dtype=bool)
    total_mass = event_mass(
        all_mask,
        px,
        py,
        pz,
        E,
    )

    # Truth reconstruction
    truth_chi0 = truth == 1
    truth_chi1 = truth == 2

    truth_chi0_mass = event_mass(
        truth_chi0,
        px,
        py,
        pz,
        E,
    )

    truth_chi1_mass = event_mass(
        truth_chi1,
        px,
        py,
        pz,
        E,
    )

    truth_suu_mass = event_mass(
        truth_chi0 | truth_chi1,
        px,
        py,
        pz,
        E,
    )

    # Algorithm reconstruction
    algo_chi0 = algorithm == 1
    algo_chi1 = algorithm == 2

    algo_chi0_mass = event_mass(
        algo_chi0,
        px,
        py,
        pz,
        E,
    )

    algo_chi1_mass = event_mass(
        algo_chi1,
        px,
        py,
        pz,
        E,
    )

    algo_suu_mass = event_mass(
        algo_chi0 | algo_chi1,
        px,
        py,
        pz,
        E,
    )

    return {

        "total_mass": total_mass,

        "truth_suu_mass": truth_suu_mass,
        "truth_chi0_mass": truth_chi0_mass,
        "truth_chi1_mass": truth_chi1_mass,

        "algo_suu_mass": algo_suu_mass,
        "algo_chi0_mass": algo_chi0_mass,
        "algo_chi1_mass": algo_chi1_mass,

        "n_pf": len(px),

        "n_truth_chi0": int(np.sum(truth_chi0)),
        "n_truth_chi1": int(np.sum(truth_chi1)),
        "n_truth_isr": int(np.sum(truth == 0)),

        "n_algo_chi0": int(np.sum(algo_chi0)),
        "n_algo_chi1": int(np.sum(algo_chi1)),
        "n_algo_isr": int(np.sum(algorithm == 0)),
    }

################################################################################
# Loop over all events
################################################################################

print("\nReconstructing events...\n")

results = []

for i in range(tree.num_entries):

    if (i + 1) % 1000 == 0:
        print(f"  {i+1:6d} / {tree.num_entries}")

    event = branches[i]

    results.append(reconstruct_event(event))

print("\nFinished reconstruction.\n")

################################################################################
# Convert results into numpy arrays
################################################################################

total_mass = np.array([r["total_mass"] for r in results])

truth_suu_mass = np.array([r["truth_suu_mass"] for r in results])
algo_suu_mass  = np.array([r["algo_suu_mass"]  for r in results])

truth_chi0_mass = np.array([r["truth_chi0_mass"] for r in results])
truth_chi1_mass = np.array([r["truth_chi1_mass"] for r in results])

algo_chi0_mass = np.array([r["algo_chi0_mass"] for r in results])
algo_chi1_mass = np.array([r["algo_chi1_mass"] for r in results])

n_pf = np.array([r["n_pf"] for r in results])

n_truth_chi0 = np.array([r["n_truth_chi0"] for r in results])
n_truth_chi1 = np.array([r["n_truth_chi1"] for r in results])
n_truth_isr  = np.array([r["n_truth_isr"]  for r in results])

n_algo_chi0 = np.array([r["n_algo_chi0"] for r in results])
n_algo_chi1 = np.array([r["n_algo_chi1"] for r in results])
n_algo_isr  = np.array([r["n_algo_isr"]  for r in results])

################################################################################
# Print overall statistics
################################################################################

print("=" * 80)
print("EVENT STATISTICS")
print("=" * 80)

print(f"Events:                  {len(results)}")
print()

print(f"Average PF candidates:   {np.mean(n_pf):8.2f}")
print(f"Minimum PF candidates:   {np.min(n_pf):8d}")
print(f"Maximum PF candidates:   {np.max(n_pf):8d}")

print()

print("Truth labels")
print("---------------------------")
print(f"Average chi0 particles :  {np.mean(n_truth_chi0):8.2f}")
print(f"Average chi1 particles :  {np.mean(n_truth_chi1):8.2f}")
print(f"Average ISR particles  :  {np.mean(n_truth_isr):8.2f}")

print()

print("Algorithm labels")
print("---------------------------")
print(f"Average chi0 particles :  {np.mean(n_algo_chi0):8.2f}")
print(f"Average chi1 particles :  {np.mean(n_algo_chi1):8.2f}")
print(f"Average ISR particles  :  {np.mean(n_algo_isr):8.2f}")

print()

print("Mass reconstruction")
print("---------------------------")

print(f"Average total PF mass      : {np.mean(total_mass):10.2f} GeV")

print(f"Average truth Suu mass     : {np.mean(truth_suu_mass):10.2f} GeV")
print(f"Average algorithm Suu mass : {np.mean(algo_suu_mass):10.2f} GeV")

print()

print(f"Average truth chi0 mass    : {np.mean(truth_chi0_mass):10.2f} GeV")
print(f"Average truth chi1 mass    : {np.mean(truth_chi1_mass):10.2f} GeV")

print(f"Average algo chi0 mass     : {np.mean(algo_chi0_mass):10.2f} GeV")
print(f"Average algo chi1 mass     : {np.mean(algo_chi1_mass):10.2f} GeV")

print("=" * 80)

################################################################################
# Label sanity checks
################################################################################

print("\n")
print("=" * 80)
print("LABEL SANITY CHECKS")
print("=" * 80)

missing_truth_chi0 = np.sum(n_truth_chi0 == 0)
missing_truth_chi1 = np.sum(n_truth_chi1 == 0)

missing_algo_chi0 = np.sum(n_algo_chi0 == 0)
missing_algo_chi1 = np.sum(n_algo_chi1 == 0)

print(f"Events missing truth chi0 : {missing_truth_chi0}")
print(f"Events missing truth chi1 : {missing_truth_chi1}")

print()

print(f"Events missing algo chi0  : {missing_algo_chi0}")
print(f"Events missing algo chi1  : {missing_algo_chi1}")

print()

truth_total = n_truth_chi0 + n_truth_chi1 + n_truth_isr
algo_total  = n_algo_chi0 + n_algo_chi1 + n_algo_isr

truth_bad = np.sum(truth_total != n_pf)
algo_bad  = np.sum(algo_total != n_pf)

print(f"Truth particle-count mismatches : {truth_bad}")
print(f"Algo particle-count mismatches  : {algo_bad}")

print("=" * 80)

################################################################################
# Energy accounting
################################################################################

print("\n")
print("=" * 80)
print("ENERGY ACCOUNTING")
print("=" * 80)

truth_chi0_frac = []
truth_chi1_frac = []

algo_chi0_frac = []
algo_chi1_frac = []

bad_events = []

for iev in range(tree.num_entries):

    event = branches[iev]

    px = event["particle_px"]
    py = event["particle_py"]
    pz = event["particle_pz"]
    E  = np.asarray(event["particle_energy"])

    truth = simplify_truth_labels(event["particle_truthLabel"])
    algo  = np.asarray(event["particle_algorithmLabel"])

    total_E = np.sum(E)

    if total_E <= 0:
        continue

    # Truth energy fractions
    E_truth_chi0 = np.sum(E[truth == 1])
    E_truth_chi1 = np.sum(E[truth == 2])

    truth_chi0_frac.append(E_truth_chi0 / total_E)
    truth_chi1_frac.append(E_truth_chi1 / total_E)

    # Algorithm energy fractions
    E_algo_chi0 = np.sum(E[algo == 1])
    E_algo_chi1 = np.sum(E[algo == 2])

    algo_chi0_frac.append(E_algo_chi0 / total_E)
    algo_chi1_frac.append(E_algo_chi1 / total_E)

    # Flag suspicious events
    suspicious = False
    reasons = []

    if results[iev]["truth_suu_mass"] < 2500:
        suspicious = True
        reasons.append("low truth Suu")

    if results[iev]["algo_suu_mass"] < 2500:
        suspicious = True
        reasons.append("low algo Suu")

    if results[iev]["truth_chi0_mass"] < 200:
        suspicious = True
        reasons.append("low truth chi0")

    if results[iev]["truth_chi1_mass"] < 200:
        suspicious = True
        reasons.append("low truth chi1")

    if results[iev]["algo_chi0_mass"] < 200:
        suspicious = True
        reasons.append("low algo chi0")

    if results[iev]["algo_chi1_mass"] < 200:
        suspicious = True
        reasons.append("low algo chi1")

    if suspicious:

        bad_events.append(
            {
                "event": iev,
                "reasons": reasons,
                "truth_suu": results[iev]["truth_suu_mass"],
                "algo_suu": results[iev]["algo_suu_mass"],
                "truth_chi0": results[iev]["truth_chi0_mass"],
                "truth_chi1": results[iev]["truth_chi1_mass"],
                "algo_chi0": results[iev]["algo_chi0_mass"],
                "algo_chi1": results[iev]["algo_chi1_mass"],
            }
        )

################################################################################
# Print energy fraction, suspicious event summary
################################################################################

truth_chi0_frac = np.asarray(truth_chi0_frac)
truth_chi1_frac = np.asarray(truth_chi1_frac)

algo_chi0_frac = np.asarray(algo_chi0_frac)
algo_chi1_frac = np.asarray(algo_chi1_frac)

print(f"Truth χ0 average energy fraction : {truth_chi0_frac.mean():.3f}")
print(f"Truth χ1 average energy fraction : {truth_chi1_frac.mean():.3f}")

print()

print(f"Algo  χ0 average energy fraction : {algo_chi0_frac.mean():.3f}")
print(f"Algo  χ1 average energy fraction : {algo_chi1_frac.mean():.3f}")

print()

print(f"Suspicious events identified : {len(bad_events)}")

print("=" * 80)

################################################################################
# Print first few suspicious events
################################################################################

print("\nFirst suspicious events:\n")

for event in bad_events[:5]:

    print("-" * 60)
    print(f"Event {event['event']}")
    print("Reasons:", ", ".join(event["reasons"]))

    print(f"Truth Suu : {event['truth_suu']:.1f}")
    print(f"Algo  Suu : {event['algo_suu']:.1f}")

    print(f"Truth χ0  : {event['truth_chi0']:.1f}")
    print(f"Truth χ1  : {event['truth_chi1']:.1f}")

    print(f"Algo  χ0  : {event['algo_chi0']:.1f}")
    print(f"Algo  χ1  : {event['algo_chi1']:.1f}")

print()

################################################################################
# Fraction of visible energy assigned to the chi system
################################################################################

truth_total_frac = truth_chi0_frac + truth_chi1_frac
algo_total_frac  = algo_chi0_frac  + algo_chi1_frac

print()
print(f"Truth χ system energy fraction :")
print(f"    mean = {truth_total_frac.mean():.3f}")
print(f"    std  = {truth_total_frac.std():.3f}")
print(f"    min  = {truth_total_frac.min():.3f}")
print(f"    max  = {truth_total_frac.max():.3f}")

print()

print(f"Algorithm χ system energy fraction :")
print(f"    mean = {algo_total_frac.mean():.3f}")
print(f"    std  = {algo_total_frac.std():.3f}")
print(f"    min  = {algo_total_frac.min():.3f}")
print(f"    max  = {algo_total_frac.max():.3f}")

################################################################################
# Plotting helper
################################################################################

def save_hist(
    data,
    bins,
    xlabel,
    title,
    filename,
    range=None,
    density=False,
):
    plt.figure(figsize=(8,6))

    plt.hist(
        data,
        bins=bins,
        range=range,
        histtype="step",
        linewidth=2,
        density=density,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(os.path.join(args.output, filename))
    plt.close()

################################################################################
# Suu masses
################################################################################

plt.figure(figsize=(8,6))

plt.hist(
    total_mass,
    bins=100,
    range=(0,5000),
    histtype="step",
    linewidth=2,
    label="All PF candidates",
)

plt.hist(
    truth_suu_mass,
    bins=100,
    range=(0,5000),
    histtype="step",
    linewidth=2,
    label="Truth labels",
)

plt.hist(
    algo_suu_mass,
    bins=100,
    range=(0,5000),
    histtype="step",
    linewidth=2,
    label="Algorithm labels",
)

plt.xlabel(r"Reconstructed $m_{Suu}$ [GeV]")
plt.ylabel("Events")
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(args.output, "suu_mass.png"))
plt.close()

################################################################################
# Chi masses
################################################################################

plt.figure(figsize=(8,6))

plt.hist(
    np.concatenate([truth_chi0_mass, truth_chi1_mass]),
    bins=100,
    range=(0,1500),
    histtype="step",
    linewidth=2,
    label="Truth",
)

plt.hist(
    np.concatenate([algo_chi0_mass, algo_chi1_mass]),
    bins=100,
    range=(0,1500),
    histtype="step",
    linewidth=2,
    label="Algorithm",
)

plt.xlabel(r"Reconstructed $m_{\chi}$ [GeV]")
plt.ylabel("Events")
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(args.output, "chi_mass.png"))
plt.close()

################################################################################
# Energy fractions
################################################################################

plt.figure(figsize=(7,5))

plt.hist(
    truth_chi0_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label=r"Truth $\chi_0$",
)

plt.hist(
    truth_chi1_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label=r"Truth $\chi_1$",
)

plt.hist(
    algo_chi0_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label=r"Algo $\chi_0$",
)

plt.hist(
    algo_chi1_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label=r"Algo $\chi_1$",
)

plt.xlabel("Fraction of visible event energy")
plt.ylabel("Events")
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(args.output, "chi_energy_fraction.png"))
plt.close()

################################################################################
# Total chi-system energy fraction
################################################################################

plt.figure(figsize=(7,5))

plt.hist(
    truth_total_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label="Truth",
)

plt.hist(
    algo_total_frac,
    bins=50,
    range=(0,1),
    histtype="step",
    linewidth=2,
    label="Algorithm",
)

plt.xlabel("Fraction of visible energy assigned to χ system")
plt.ylabel("Events")
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(args.output, "total_chi_fraction.png"))
plt.close()

################################################################################
# Scatter: reconstructed chi masses
################################################################################

plt.figure(figsize=(6,6))

plt.scatter(
    truth_chi0_mass,
    truth_chi1_mass,
    s=5,
    alpha=0.4,
    label="Truth",
)

plt.xlabel(r"$m(\chi_0)$ [GeV]")
plt.ylabel(r"$m(\chi_1)$ [GeV]")

plt.tight_layout()

plt.savefig(os.path.join(args.output, "truth_chi_scatter.png"))
plt.close()


plt.figure(figsize=(6,6))

plt.scatter(
    algo_chi0_mass,
    algo_chi1_mass,
    s=5,
    alpha=0.4,
    label="Algorithm",
)

plt.xlabel(r"$m(\chi_0)$ [GeV]")
plt.ylabel(r"$m(\chi_1)$ [GeV]")

plt.tight_layout()

plt.savefig(os.path.join(args.output, "algo_chi_scatter.png"))
plt.close()


print(f"\nSaved plots to {args.output}/")
