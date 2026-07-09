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

    absid = abs(pdgid)

    isElectron = ak.values_astype(absid == 11, np.float32)
    isMuon     = ak.values_astype(absid == 13, np.float32)
    isPhoton   = ak.values_astype(absid == 22, np.float32)

    isChargedHadron = ak.values_astype(
        (absid == 211) |
        (absid == 321) |
        (absid == 2212),
        np.float32
    )

    isNeutralHadron = ak.values_astype(
        (absid == 130) |
        (absid == 2112),
        np.float32
    )

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
        ak padded, filled array of features
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
            pad_array(logpt),
            pad_array(eta),
            pad_array(sinphi),
            pad_array(cosphi),
            pad_array(logE),
            pad_array(charge),
            pad_array(isElectron),
            pad_array(isMuon),
            pad_array(isPhoton),
            pad_array(isCH),
            pad_array(isNH),
            pad_array(tanhdxy),
            pad_array(tanhdz),
        ],
        axis=-1,
    )

    return features

def load_root_file(filename, treename="particleTransformerNtuplizer/Events"):
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
    For data arrays, find the features, mask, raw kinematics, and global features

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
    # Extract particle-level branches, ordering by pt
    ##########################################################

    order = ak.argsort(arrays["particle_pt"], ascending=False)

    pt     = arrays["particle_pt"][order]
    eta    = arrays["particle_eta"][order]
    phi    = arrays["particle_phi"][order]
    energy = arrays["particle_energy"][order]

    charge = arrays["particle_charge"][order]
    pdgid  = arrays["particle_pdgId"][order]

    dxy    = arrays["particle_dxy"][order]
    dz     = arrays["particle_dz"][order]

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

    X_particles = ak.to_numpy(features).astype(np.float32)

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
            np.asarray(arrays["HT"]),
            np.asarray(arrays["MET_pt"]),
            np.asarray(arrays["MET_phi"]),
            np.asarray(arrays["rho"]),
            np.asarray(arrays["nPV"]),
            np.asarray(arrays["PV_x"]),
            np.asarray(arrays["PV_y"]),
            np.asarray(arrays["PV_z"]),
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
            np.asarray(arrays["gen_Suu_px"]),
            np.asarray(arrays["gen_Suu_py"]),
            np.asarray(arrays["gen_Suu_pz"]),
            np.asarray(arrays["gen_Suu_E"]),
            np.asarray(arrays["gen_Suu_pt"]),
            np.asarray(arrays["gen_Suu_eta"]),
            np.asarray(arrays["gen_Suu_phi"]),
            np.asarray(arrays["gen_Suu_mass"]),
        ],
        axis=1,
    ).astype(np.float32)

    targets["chi0"] = np.stack(
        [
            np.asarray(arrays["gen_chi0_px"]),
            np.asarray(arrays["gen_chi0_py"]),
            np.asarray(arrays["gen_chi0_pz"]),
            np.asarray(arrays["gen_chi0_E"]),
            np.asarray(arrays["gen_chi0_pt"]),
            np.asarray(arrays["gen_chi0_eta"]),
            np.asarray(arrays["gen_chi0_phi"]),
            np.asarray(arrays["gen_chi0_mass"]),
        ],
        axis=1,
    ).astype(np.float32)

    targets["chi1"] = np.stack(
        [
            np.asarray(arrays["gen_chi1_px"]),
            np.asarray(arrays["gen_chi1_py"]),
            np.asarray(arrays["gen_chi1_pz"]),
            np.asarray(arrays["gen_chi1_E"]),
            np.asarray(arrays["gen_chi1_pt"]),
            np.asarray(arrays["gen_chi1_eta"]),
            np.asarray(arrays["gen_chi1_phi"]),
            np.asarray(arrays["gen_chi1_mass"]),
        ],
        axis=1,
    ).astype(np.float32)

    order = ak.argsort(arrays["particle_pt"], ascending=False)

    truthLabel = arrays["particle_truthLabel"][order]

    truthLabel = ak.where(
        (truthLabel >= 1) & (truthLabel <= 11),
        1,
        truthLabel,
    )

    truthLabel = ak.where(
        (truthLabel >= 12) & (truthLabel <= 22),
        2,
        truthLabel,
    )

    targets["truthLabel"] = pad_array(truthLabel, pad_value = -2).astype(np.int64)

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

    shard_id = 0
    all_train, all_val, all_test = [], [], []

    # Iterate over shards; cannot process all particles from all events at once
    for arrays in uproot.iterate(
        f"{args.input}:{args.tree}",
        step_size=5000,
        library="ak",
    ):

        # Process particles + globals
        (
            X_particles,
            mask,
            raw_pt,
            raw_eta,
            raw_phi,
            raw_E,
            global_features,
        ) = process_events(arrays)

        # Build targets
        targets = build_targets(arrays)

        # Make train/val/test splits
        n_events = X_particles.shape[0]
        train_idx, val_idx, test_idx = make_splits(n_events)

        # Dict of datasets
        dataset_torch = {
            "particles": torch.tensor(X_particles),
            "mask": torch.tensor(mask, dtype=torch.bool),

            "raw_pt": torch.tensor(raw_pt),
            "raw_eta": torch.tensor(raw_eta),
            "raw_phi": torch.tensor(raw_phi),
            "raw_E": torch.tensor(raw_E),

            "global": torch.tensor(global_features),

            "suu": torch.tensor(targets["Suu"]),
            "chi0": torch.tensor(targets["chi0"]),
            "chi1": torch.tensor(targets["chi1"]),
            "truthLabel": torch.tensor(targets["truthLabel"]),

            "train_idx": torch.tensor(train_idx),
            "val_idx": torch.tensor(val_idx),
            "test_idx": torch.tensor(test_idx),
        }

        # Save
        out_file = args.output.replace(".pt", f"_shard{shard_id:04d}.pt")
        torch.save(dataset_torch, out_file)

        print(f"[INFO] Wrote {out_file} with {n_events} events")

        shard_id += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", 
        type=str, 
        default="/eos/uscms/store/user/aji/rootfiles_particleTransformer/WbWb_4000_1000.root", # for testing
        #default="/eos/uscms/store/user/aji/rootfiles_particleTransformer/WbWb_all.root", # for realsies
        help="Input ROOT file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="ptfiles/WbWb_4000_1000.pt", # for testing
        #default="ptfiles/WbWb_all.pt", # for realsies
        help="Output .pt file"
    )

    parser.add_argument(
        "--tree",
        type=str,
        default="particleTransformerNtuplizer/Events",
        help="ROOT TTree name",
    )

    args = parser.parse_args()

    main(args)