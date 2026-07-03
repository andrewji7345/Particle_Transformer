"""
Preprocess ROOT ntuples for the Suu -> chi chi particle transformer.

This script

1. Reads the ROOT tree using uproot
2. Builds per-particle features
3. Pads/truncates to MAX_PARTICLES
4. Creates particle masks
5. Extracts generator truth
6. Splits into train/validation/test
7. Saves everything as PyTorch tensors

Example:

python preprocess_particles.py \
    /eos/uscms/store/user/aji/rootfiles_particleTransformer/WbWb_4000_1000.root \
    /ptfiles/WbWb_4000_1000.pt
"""

import argparse

import awkward as ak
import numpy as np
import torch
import uproot

from sklearn.model_selection import train_test_split

##############################################################
# Configuration, branch lists
##############################################################

MAX_PARTICLES = 128

EPS = 1e-6

PARTICLE_BRANCHES = [
    "particle_pt",
    "particle_eta",
    "particle_phi",
    "particle_energy",
    "particle_charge",
    "particle_pdgId",
    "particle_dxy",
    "particle_dz",
    "particle_truthLabel",
]

GLOBAL_BRANCHES = [
    "HT",
    "MET_pt",
    "MET_phi",
    "rho",
    "nPV",
    "PV_x",
    "PV_y",
    "PV_z",
]

TARGET_BRANCHES = [
    "gen_Suu_px",
    "gen_Suu_py",
    "gen_Suu_pz",
    "gen_Suu_E",
    "gen_chi0_px",
    "gen_chi0_py",
    "gen_chi0_pz",
    "gen_chi0_E",
    "gen_chi1_px",
    "gen_chi1_py",
    "gen_chi1_pz",
    "gen_chi1_E",
]

##############################################################
# Helper functions
##############################################################

def pad_array(array, pad_value=0.0):
    """
    Pad or truncate an awkward array to MAX_PARTICLES.

    Input:
        ak array, shape = (NEvents, NParticles)
        pad_value

    Output:
        np array, shape = (Nevents, MAX_PARTICLES)
    """

    padded = ak.pad_none(
        array,
        MAX_PARTICLES,
        axis=1,
        clip=True,
    )

    padded = ak.fill_none(
        padded,
        pad_value,
    )

    return ak.to_numpy(padded)

def build_mask(pt):
    """
    Builds particle mask in case of padding.

    Input:
        ak array, jagged shape = (NEvents, NParticles)

    Output:
        np array, shape = (NEvents, MAX_PARTICLES)
    """

    counts = ak.num(pt)

    mask = np.zeros(
        (len(counts), MAX_PARTICLES),
        dtype=np.bool_,
    )

    for i, n in enumerate(counts):
        mask[i, : min(n, MAX_PARTICLES)] = True

    return mask

def particle_flags(pdgid):
    """
    Returns particle flags given a pdgID.

    Input:
        pdgID

    Output:
        (
        isElectron,
        isMuon,
        isPhoton,
        isChargedHadron,
        isNeutralHadron,
        )
    """

    absid = np.abs(pdgid)
    isElectron = (absid == 11).astype(np.float32)
    isMuon = (absid == 13).astype(np.float32)
    isPhoton = (absid == 22).astype(np.float32)

    isChargedHadron = (
        (absid == 211) |
        (absid == 321) |
        (absid == 2212)
    ).astype(np.float32)

    isNeutralHadron = (
        (absid == 130) |
        (absid == 2112)
    ).astype(np.float32)

    return (
        isElectron,
        isMuon,
        isPhoton,
        isChargedHadron,
        isNeutralHadron,
    )

def build_features(
    pt,
    eta,
    phi,
    energy,
    charge,
    pdgid,
    dxy,
    dz,
):
    """
    Builds features used in the transformer.

    Input:
        kinematic variables

    Output:
        np array of features
    """

    logpt = np.log(pt + EPS)
    logE = np.log(energy + EPS)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    tanhdxy = np.tanh(dxy)
    tanhdz = np.tanh(dz)

    (
        isElectron,
        isMuon,
        isPhoton,
        isCH,
        isNH,
    ) = particle_flags(pdgid)

    features = np.stack(

        [

            logpt,
            eta,
            sinphi,
            cosphi,
            logE,

            charge,

            isElectron,
            isMuon,
            isPhoton,
            isCH,
            isNH,

            tanhdxy,
            tanhdz,

        ],

        axis=-1,

    )

    return features

def load_root_file(filename, treename="Events"):
    """
    Load ROOT tree and return uproot arrays.
    """

    file = uproot.open(filename)
    tree = file[treename]

    arrays = tree.arrays(
        PARTICLE_BRANCHES + GLOBAL_BRANCHES + TARGET_BRANCHES,
        library="ak",
    )

    return arrays

def process_events(arrays):
    """
    For a data array, find the features, mask, raw kinematics, and global features

    Input:
        ak arrays

    Output:
        X_particles: (Nevents, MAX_PARTICLES, 13)
        mask:        (Nevents, MAX_PARTICLES)
        raw_pt:      raw
        raw_eta:     raw
        raw_phi:     raw
        raw_E:       raw
        global_features: HT, MET, etc.
    )
    """

    ##########################################################
    # Extract particle-level branches
    ##########################################################

    pt     = arrays["particle_pt"]
    eta    = arrays["particle_eta"]
    phi    = arrays["particle_phi"]
    energy = arrays["particle_energy"]

    charge = arrays["particle_charge"]
    pdgid  = arrays["particle_pdgId"]

    dxy    = arrays["particle_dxy"]
    dz     = arrays["particle_dz"]

    features = build_features(
        pt,
        eta,
        phi,
        energy,
        charge,
        pdgid,
        dxy,
        dz,
    )

    ##########################################################
    # Pad/truncate features
    ##########################################################

    X_particles = ak.pad_none(
        features,
        MAX_PARTICLES,
        axis=1,
        clip=True,
    )

    X_particles = ak.fill_none(X_particles, 0.0)
    X_particles = ak.to_numpy(X_particles).astype(np.float32)

    ##########################################################
    # Build particle mask
    ##########################################################

    mask = build_mask(pt)

    ##########################################################
    # Save raw kinematics (used later)
    ##########################################################

    raw_pt  = pad_array(pt)
    raw_eta = pad_array(eta)
    raw_phi = pad_array(phi)
    raw_E   = pad_array(energy)

    raw_pt  = raw_pt.astype(np.float32)
    raw_eta = raw_eta.astype(np.float32)
    raw_phi = raw_phi.astype(np.float32)
    raw_E   = raw_E.astype(np.float32)

    ##########################################################
    # Global features
    ##########################################################

    global_features = np.stack(
        [
            arrays["HT"],
            arrays["MET_pt"],
            arrays["MET_phi"],
            arrays["rho"],
            arrays["nPV"],
            arrays["PV_x"],
            arrays["PV_y"],
            arrays["PV_z"],
        ],
        axis=1,
    ).astype(np.float32)

    return (
        X_particles,
        mask,
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        global_features,
    )

def build_targets(arrays):
    """
    Get the generator targets.

    Input:
        ak arrays

    Output:
        dict of ak arrays
    """

    targets = {}

    targets["Suu"] = np.stack(
        [
            arrays["gen_Suu_px"],
            arrays["gen_Suu_py"],
            arrays["gen_Suu_pz"],
            arrays["gen_Suu_E"],
            arrays["gen_Suu_pt"],
            arrays["gen_Suu_eta"],
            arrays["gen_Suu_phi"],
            arrays["gen_Suu_mass"],
        ],
        axis=1,
    ).astype(np.float32)

    targets["chi0"] = np.stack(
        [
            arrays["gen_chi0_px"],
            arrays["gen_chi0_py"],
            arrays["gen_chi0_pz"],
            arrays["gen_chi0_E"],
            arrays["gen_chi0_pt"],
            arrays["gen_chi0_eta"],
            arrays["gen_chi0_phi"],
            arrays["gen_chi0_mass"],
        ],
        axis=1,
    ).astype(np.float32)

    targets["chi1"] = np.stack(
        [
            arrays["gen_chi1_px"],
            arrays["gen_chi1_py"],
            arrays["gen_chi1_pz"],
            arrays["gen_chi1_E"],
            arrays["gen_chi1_pt"],
            arrays["gen_chi1_eta"],
            arrays["gen_chi1_phi"],
            arrays["gen_chi1_mass"],
        ],
        axis=1,
    ).astype(np.float32)

    return targets

def make_splits(n_events, seed=42):
    """
    Get indices for splitting dataset.

    Input:
        NEvents

    Output:
        train_idx, val_idx, test_idx
    """


    idx = np.arange(n_events)

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.1765,  # 15% of full dataset
        random_state=seed,
        shuffle=True,
    )

    return train_idx, val_idx, test_idx

def to_torch(x, dtype=torch.float32):
    """
    Convert to torch.
    """

    return torch.tensor(x, dtype=dtype)

def main(args):
    """
    Run preprocessing.
    """

    n_events = X_particles.shape[0]

    ##########################################################
    # Load ROOT
    ##########################################################

    arrays = load_root_file(args.input, args.tree)

    ##########################################################
    # Process particles + globals
    ##########################################################

    (
        X_particles,
        mask,
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        global_features,
    ) = process_events(arrays)

    ##########################################################
    # Build targets
    ##########################################################

    targets = build_targets(arrays)

    ##########################################################
    # Train/val/test split
    ##########################################################

    train_idx, val_idx, test_idx = make_splits(n_events)

    ##########################################################
    # Package dataset
    ##########################################################

    dataset = {
        "particles": X_particles,
        "mask": mask,

        "raw_pt": raw_pt,
        "raw_eta": raw_eta,
        "raw_phi": raw_phi,
        "raw_E": raw_E,

        "global": global_features,

        "targets": targets,

        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    ##########################################################
    # Convert to torch
    ##########################################################

    dataset_torch = {
        "particles": to_torch(dataset["particles"]),
        "mask": torch.tensor(dataset["mask"], dtype=torch.bool),

        "raw_pt": to_torch(dataset["raw_pt"]),
        "raw_eta": to_torch(dataset["raw_eta"]),
        "raw_phi": to_torch(dataset["raw_phi"]),
        "raw_E": to_torch(dataset["raw_E"]),

        "global": to_torch(dataset["global"]),

        "targets": {
            "Suu": to_torch(dataset["targets"]["Suu"]),
            "chi0": to_torch(dataset["targets"]["chi0"]),
            "chi1": to_torch(dataset["targets"]["chi1"]),
        },

        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx": torch.tensor(val_idx, dtype=torch.long),
        "test_idx": torch.tensor(test_idx, dtype=torch.long),
    }

    ##########################################################
    # Save
    ##########################################################

    torch.save(dataset_torch, args.output)

    print(f"[INFO] Saved dataset to {args.output}")
    print(f"[INFO] Events: {n_events}")
    print(f"[INFO] Particle shape: {dataset_torch['particles'].shape}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input", 
        type=str, 
        default="/eos/uscms/store/user/aji/rootfiles_particleTransformer/WbWb_4000_1000.root", 
        help="Input ROOT file"
    )
    parser.add_argument(
        "output", 
        type=str, 
        default="ptfiles/WbWb_4000_1000.pt", 
        help="Output .pt file"
    )

    parser.add_argument(
        "--tree",
        type=str,
        default="Events",
        help="ROOT TTree name",
    )

    args = parser.parse_args()

    main(args)