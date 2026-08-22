"""
train.py

Particle Transformer implementation for Suu -> chichi reconstruction.

Architecture:
    Particle features
        ↓
    ParticleEmbedding
        ↓
    PairwiseFeatureBuilder
        ↓
    InteractionEmbedding
        ↓
    4 x ParticleAttentionBlock
        ↓
    ParticleClassifier
        ↓
    Per-particle logits:
        bkg / chi0 / chi1
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from tqdm.auto import tqdm

###########################################################################
# Model hyperparameters
###########################################################################

EMBED_DIM = 96

NUM_HEADS = 6
HEAD_DIM = 16

NUM_LAYERS = 4

MLP_RATIO = 4

DROPOUT = 0.1

INTERACTION_DIM = 16
NUM_CLASSES = 3

DATA_MODES = [
    "all_pf",
    "all_ak_constituents",
]

MIN_CHI_PT_FRACTION = 0.25
BKG_EMPTY_PT_SCALE = 0.05

LAMBDA_MASS      = 1.0
LAMBDA_ENTROPY   = 0.2
LAMBDA_NONEMPTY  = 0.5
LAMBDA_SPLIT     = 1.0
LAMBDA_BKG       = 0.5

###########################################################################
# Particle embedding
###########################################################################

class ParticleEmbedding(nn.Module):
    """
    Embed the input particle features into the transformer latent space.

    Input:
        (B, N, F)

    Output:
        (B, N, EMBED_DIM)

    B is the batch size, N is the number of particles.
    F is the number of per-particle input features.

    Following the ParT paper:
        Linear(F -> 256)
        LayerNorm
        GELU
        Linear(256 -> 512)
        LayerNorm
        GELU
        Linear(512 -> 256)
    """

    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, EMBED_DIM),
        )

    def forward(self, x):
        """
        Input:
            x: (B, N, F)

        Output:
            Tensor: (B, N, EMBED_DIM)
        """
        return self.embedding(x)

###########################################################################
# Pairwise feature builder
###########################################################################

class PairwiseFeatureBuilder(nn.Module):
    """
    Compute the pairwise interaction features used by the transformer.

    Input:
        puppi_pt  : (B, N)
        puppi_eta : (B, N)
        puppi_phi : (B, N)
        puppi_px  : (B, N)
        puppi_py  : (B, N)
        puppi_pz  : (B, N)
        puppi_E   : (B, N)
        mask      : (B, N)

    Output:
        pair_features : (B, N, N, 4)

    Feature ordering:
        0 : DeltaR = sqrt((y_a - y_b)^2 + (phi_a - phi_b)^2)
        1 : kt = min(p_{T,a}, p_{T,b}) * DeltaR
        2 : z = min(p_{T,a}, p_{T,b}) / (p_{T,a} + p_{T,b})
        3 : m^2 = (E_a + E_b)^2 - |vec{p_a} + vec{p_b}|^2
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def delta_phi(phi1, phi2):
        """
        Compute wrapped Deltaphi in [-pi, pi].
        """

        return (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi

    def forward(
        self,
        puppi_pt,
        puppi_eta,
        puppi_phi,
        puppi_px,
        puppi_py,
        puppi_pz,
        puppi_E,
        mask,
    ):

        B, N = puppi_pt.shape

        ############################################################
        # Construct pairwise tensors
        ############################################################

        pt_i = puppi_pt.unsqueeze(2)
        pt_j = puppi_pt.unsqueeze(1)

        eta_i = puppi_eta.unsqueeze(2)
        eta_j = puppi_eta.unsqueeze(1)

        phi_i = puppi_phi.unsqueeze(2)
        phi_j = puppi_phi.unsqueeze(1)

        px_i = puppi_px.unsqueeze(2)
        px_j = puppi_px.unsqueeze(1)

        py_i = puppi_py.unsqueeze(2)
        py_j = puppi_py.unsqueeze(1)

        pz_i = puppi_pz.unsqueeze(2)
        pz_j = puppi_pz.unsqueeze(1)

        E_i = puppi_E.unsqueeze(2)
        E_j = puppi_E.unsqueeze(1)

        deta = eta_i - eta_j

        dphi = self.delta_phi(phi_i, phi_j)

        deltaR = torch.sqrt(deta**2 + dphi**2 + 1e-8)

        kt = torch.minimum(pt_i, pt_j) * deltaR

        z = torch.minimum(pt_i, pt_j) / (pt_i + pt_j + 1e-8)

        # Use the exact stored PUPPI Cartesian four-vectors. This remains valid
        # for massive candidates and avoids reconstructing pz from eta.
        m2 = (
            (E_i + E_j) ** 2
            - (px_i + px_j) ** 2
            - (py_i + py_j) ** 2
            - (pz_i + pz_j) ** 2
        )

        ############################################################
        # Stack features (use log)
        ############################################################

        pair_features = torch.stack(
            (
                torch.log(torch.clamp(deltaR, min=1e-8)),
                torch.log(torch.clamp(kt, min=1e-8)),
                torch.log(torch.clamp(z, min=1e-8)),
                torch.log(torch.clamp(m2, min=1e-8)),
            ),
            dim=-1,
        )

        ############################################################
        # Zero out padded particles
        ############################################################

        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1))

        pair_features = pair_features * pair_mask.unsqueeze(-1)

        return pair_features

###########################################################################
# Interaction embedding
###########################################################################

class InteractionEmbedding(nn.Module):
    """
    Encode the pairwise interaction variables into an attention bias.

    Input:
        pair_features : (B, N, N, 4)

    Output:
        attn_bias : (B, NUM_HEADS, N, N)

    The intermediate representation has INTERACTION_DIM channels
    before being projected to one scalar attention bias per head.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=4,
                out_channels=64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                64,
                INTERACTION_DIM,
                kernel_size=1,
            ),
        )

        # Convert the learned interaction embedding into
        # one scalar attention bias per attention head.
        self.head_projection = nn.Linear(
            INTERACTION_DIM,
            NUM_HEADS,
        )

    def forward(self, pair_features):
        """
        Input:
            (B, N, N, 4)

        Output:
            (B, NUM_HEADS, N, N)
        """

        # Conv2d expects channels first.
        x = pair_features.permute(
            0,
            3,
            1,
            2,
        )

        # (B, 4, N, N) -> (B, 16, N, N)
        x = self.encoder(x)

        # Move channels back to the end.
        x = x.permute(
            0,
            2,
            3,
            1,
        )

        # (B, N, N, 16) -> (B, N, N, NUM_HEADS)
        x = self.head_projection(x)

        # MultiheadAttention wants (B, NUM_HEADS, N, N)
        x = x.permute(
            0,
            3,
            1,
            2,
        )

        return x

###########################################################################
# Particle Multi-Head Attention
###########################################################################

class ParticleMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with an additive learned attention bias.

    Input:
        x         : (B, N, EMBED_DIM)
        attn_bias : (B, NUM_HEADS, N, N)
        mask      : (B, N)

    Output:
        (B, N, EMBED_DIM)
    """

    def __init__(self):

        super().__init__()

        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM

        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, attn_bias, mask):

        B, N, _ = x.shape

        ############################################################
        # Project to Q, K, V
        ############################################################

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        ############################################################
        # Split into attention heads
        ############################################################

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        ############################################################
        # Normalized dot-product attention + interaction bias
        ############################################################

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(self.head_dim)

        scores = scores + attn_bias

        ############################################################
        # Mask padded particles
        ############################################################

        key_mask = mask.unsqueeze(1).unsqueeze(2)

        scores = scores.masked_fill(
            ~key_mask,
            float(-1e9),
        )

        ############################################################
        # Softmax, dropout
        ############################################################

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(attention)

        ############################################################
        # Weighted sum
        ############################################################

        output = torch.matmul(
            attention,
            V,
        )

        ############################################################
        # Merge attention heads
        ############################################################

        output = output.transpose(1, 2)

        output = output.reshape(
            B,
            N,
            EMBED_DIM,
        )

        output = output.masked_fill(
            ~mask.unsqueeze(-1),
            0.,
        )

        return output

###########################################################################
# Particle Attention Block
###########################################################################

class ParticleAttentionBlock(nn.Module):
    """
    One Particle Transformer encoder block. Described in Figure 3b in Qu 2022.
    """

    def __init__(self):

        super().__init__()

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.attention = ParticleMultiHeadAttention()

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.norm3 = nn.LayerNorm(EMBED_DIM)

        self.fc1 = nn.Linear(
            EMBED_DIM,
            EMBED_DIM * MLP_RATIO,
        )

        self.norm4 = nn.LayerNorm(EMBED_DIM * MLP_RATIO)

        self.fc2 = nn.Linear(
            EMBED_DIM * MLP_RATIO,
            EMBED_DIM,
        )

    def forward(
        self,
        x,
        attn_bias,
        mask,
    ):

        # Self-attention
        x = x + self.norm2(
            self.attention(
                self.norm1(x),
                attn_bias,
                mask,
            )
        )

        # Feed-forward network
        y = self.norm3(x)

        y = self.fc1(y)

        y = F.gelu(y)

        y = self.norm4(y)

        y = self.fc2(y)

        x = x + y

        return x

###########################################################################
# Particle Transformer
###########################################################################

class ParticleTransformer(nn.Module):
    """
    Particle Transformer for per-particle classification.

    Input:
        particles : (B, N, F)
            Per-particle input features
            (eta, phi, log(pt), log(E), charge, ...)
        puppi_pt  : (B, N)
        puppi_eta : (B, N)
        puppi_phi : (B, N)
        puppi_px  : (B, N)
        puppi_py  : (B, N)
        puppi_pz  : (B, N)
        puppi_E   : (B, N)
        mask      : (B, N)
            Boolean mask indicating valid particles.

    Output:
        dict      : logits : (B, N, NUM_CLASSES)
    """

    def __init__(self, input_dim):

        super().__init__()

        # Particle embedding
        self.particle_embedding = ParticleEmbedding(input_dim)

        # Pairwise feature computation
        self.pair_builder = PairwiseFeatureBuilder()

        # Interaction embedding
        self.interaction_embedding = InteractionEmbedding()

        # Transformer encoder
        self.blocks = nn.ModuleList([ParticleAttentionBlock() for _ in range(NUM_LAYERS)])

        # Final particle classifier
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(
        self,
        particles,
        puppi_pt,
        puppi_eta,
        puppi_phi,
        puppi_px,
        puppi_py,
        puppi_pz,
        puppi_E,
        mask,
    ):
        # Particle embedding
        x = self.particle_embedding(particles)

        # Pairwise interaction variables
        pair_features = self.pair_builder(
            puppi_pt,
            puppi_eta,
            puppi_phi,
            puppi_px,
            puppi_py,
            puppi_pz,
            puppi_E,
            mask,
        )

        # Learned attention bias
        attn_bias = self.interaction_embedding(pair_features)

        # Transformer encoder
        for block in self.blocks:
            x = block(
                x,
                attn_bias,
                mask,
            )

        # Per-particle classification
        logits = self.classifier(x)

        probabilities = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
        }

###########################################################################
# Dataset
###########################################################################

class ParticleTransformerDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Particle Transformer.

    Each item contains everything needed for one forward pass.

    Expected training fields from the v3 preprocessing schema:

        {
            "particles"       : Tensor(N,F),
            "mask"            : Tensor(N),
            "puppi_pt"        : Tensor(N),
            "puppi_eta"       : Tensor(N),
            "puppi_phi"       : Tensor(N),
            "puppi_px"        : Tensor(N),
            "puppi_py"        : Tensor(N),
            "puppi_pz"        : Tensor(N),
            "puppi_E"         : Tensor(N),
            "truthLabel"      : Tensor(N),
            "algorithmLabel"  : Tensor(N),
            "algorithmCAIndex": Tensor(N),
        }
    """

    ITEM_KEYS = (
        "particles",
        "mask",
        "puppi_pt",
        "puppi_eta",
        "puppi_phi",
        "puppi_px",
        "puppi_py",
        "puppi_pz",
        "puppi_E",
        "truthLabel",
        "truthAncestryMask",
        "truthHasChi0",
        "truthHasChi1",
        "truthMixed",
        "algorithmLabel",
        "algorithmAKIndex",
        "algorithmCAIndex",
        "run",
        "lumi",
        "event",
        "split",
        "chi0",
        "chi1",
    )

    def __init__(
        self,
        dataset,
        indices,
    ):

        super().__init__()

        self.dataset = dataset
        self.metadata = dataset["metadata"]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        event_idx = self.indices[idx]
        return {key: self.dataset[key][event_idx] for key in self.ITEM_KEYS}

###########################################################################
# Dataset loading helper
###########################################################################

def validate_dataset_shard(dataset, path):
    """Validate one preprocessed shard before any tensors are concatenated."""

    required = set(ParticleTransformerDataset.ITEM_KEYS) | {
        "metadata",
        "split_hash",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")

    metadata = dataset["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError(f"{path}: metadata must be a dictionary")

    schema_version = metadata.get("schema_version")
    expected_version = "particle_transformer_preprocessor_v3"
    if schema_version != expected_version:
        raise ValueError(
            f"{path}: expected schema {expected_version}, got {schema_version!r}"
        )

    n_events = dataset["particles"].shape[0]
    for key in required - {"metadata"}:
        value = dataset[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{path}: {key} must be a tensor")
        if value.ndim == 0 or value.shape[0] != n_events:
            raise ValueError(
                f"{path}: {key} is not aligned with its {n_events} events"
            )

    feature_names = metadata.get("features", {}).get("particle_names")
    if feature_names is None:
        raise KeyError(f"{path}: metadata is missing particle feature names")
    if dataset["particles"].shape[-1] != len(feature_names):
        raise ValueError(
            f"{path}: particles has {dataset['particles'].shape[-1]} features "
            f"but metadata defines {len(feature_names)}"
        )


def validate_shard_compatibility(datasets, shard_paths):
    """Ensure all shards describe the same model input and selection."""

    reference = datasets[0]["metadata"]
    reference_features = reference["features"]["particle_names"]
    reference_selection = reference["selection"]

    for dataset, path in zip(datasets[1:], shard_paths[1:]):
        metadata = dataset["metadata"]
        if metadata["features"]["particle_names"] != reference_features:
            raise ValueError(f"{path}: particle feature schema differs across shards")
        for key in ("mode", "Nparticles", "teacher"):
            if metadata["selection"].get(key) != reference_selection.get(key):
                raise ValueError(f"{path}: selection metadata differs for {key}")


def resolve_shard_paths(dataset_name, pt_dir="ptfiles"):
    """Resolve a dataset stub or one preprocessed shard filename.

    Passing any ``<dataset>_shardNNNN.pt`` file selects every numbered shard
    belonging to that dataset. A path in ``--input`` takes precedence over
    ``--pt-dir``; the historical dataset-stub interface remains supported.
    """

    input_path = Path(dataset_name)
    if input_path.parent != Path("."):
        search_dir = input_path.parent
    else:
        search_dir = Path(pt_dir)

    input_name = input_path.name
    shard_match = re.fullmatch(r"(.+)_shard\d+\.pt", input_name)
    if shard_match:
        dataset_stub = shard_match.group(1)
    elif input_name.endswith(".pt"):
        raise ValueError(
            f"Input filename {input_name!r} does not end in _shardNNNN.pt"
        )
    else:
        dataset_stub = input_name

    shard_paths = sorted(search_dir.glob(f"{dataset_stub}_shard*.pt"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found matching "
            f"{search_dir / (dataset_stub + '_shard*.pt')}"
        )

    return shard_paths


def load_particle_datasets(dataset_name, pt_dir="ptfiles"):
    """
    Load all shards for a dataset and construct train/val/test datasets.
    """

    shard_paths = resolve_shard_paths(dataset_name, pt_dir)

    datasets = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in shard_paths
    ]

    for dataset, path in zip(datasets, shard_paths):
        validate_dataset_shard(dataset, path)
    validate_shard_compatibility(datasets, shard_paths)

    # Keys that should NOT simply be concatenated
    index_keys = {
        "train_idx",
        "val_idx",
        "test_idx",
        "metadata",
    }

    merged = {}

    for key in datasets[0].keys():
        if key in index_keys:
            continue
        if not torch.is_tensor(datasets[0][key]):
            raise TypeError(f"Unexpected non-tensor dataset field: {key}")
        if any(key not in dataset for dataset in datasets):
            raise KeyError(f"Dataset field {key} is missing from one or more shards")
        merged[key] = torch.cat([dataset[key] for dataset in datasets], dim=0)

    # Split codes are derived from event identity in preprocessing and remain
    # correct regardless of shard size or shard ordering. Do not combine the
    # shard-local *_idx tensors.
    split = merged["split"]
    unexpected_split_codes = set(split.unique().tolist()) - {0, 1, 2}
    if unexpected_split_codes:
        raise ValueError(f"Unexpected split codes: {unexpected_split_codes}")

    merged["train_idx"] = torch.nonzero(split == 0, as_tuple=False).flatten()
    merged["val_idx"] = torch.nonzero(split == 1, as_tuple=False).flatten()
    merged["test_idx"] = torch.nonzero(split == 2, as_tuple=False).flatten()
    merged["metadata"] = {
        "schema_version": datasets[0]["metadata"]["schema_version"],
        "features": datasets[0]["metadata"]["features"],
        "selection": datasets[0]["metadata"]["selection"],
        "shards": [dataset["metadata"] for dataset in datasets],
    }

    train_dataset = ParticleTransformerDataset(merged, merged["train_idx"])
    val_dataset = ParticleTransformerDataset(merged, merged["val_idx"])
    test_dataset = ParticleTransformerDataset(merged, merged["test_idx"])

    return (train_dataset, val_dataset, test_dataset)

###########################################################################
# Cross entropy loss used for student training.
###########################################################################

class PretrainingLoss(nn.Module):

    """
    Cross entropy against the selected algorithm labels.
    """

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(
        self,
        logits,
        slimmed_labels,
    ):

        B, N = slimmed_labels.shape

        swapped_labels = slimmed_labels.clone()
        swapped_labels[slimmed_labels == 1] = 2
        swapped_labels[slimmed_labels == 2] = 1

        particle_loss = self.ce(
            logits.reshape(-1, NUM_CLASSES),
            slimmed_labels.reshape(-1),
        ).reshape(B, N)

        swapped_particle_loss = self.ce(
            logits.reshape(-1, NUM_CLASSES),
            swapped_labels.reshape(-1),
        ).reshape(B, N)

        valid_mask = (slimmed_labels != -1)

        event_loss = (
            particle_loss * valid_mask
        ).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        swapped_event_loss = (
            swapped_particle_loss * valid_mask
        ).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        event_loss = torch.minimum(event_loss, swapped_event_loss)

        loss = event_loss.mean()

        return {
            "loss": loss,
            "event_loss": event_loss,
        }

###########################################################################
# Two-body loss
###########################################################################

class TwoBodyLoss(nn.Module):

    """
    Differentiable event-level reconstruction loss.

    The loss uses no truth-level chi assignment or truth-level mass:

        L = L_mass + L_entropy + L_split + L_nonempty + L_bkg

    The individual terms encourage equal reconstructed chi masses, confident
    particle assignments, coherent assignment of signal-like CA8 jets, two
    nonempty chi candidates, and diffuse transverse background activity.
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        probabilities,
        puppi_px,
        puppi_py,
        puppi_pz,
        puppi_E,
        slimmed_CA8indices,
        mask,
        epoch,
        num_epochs,
    ):

        eps = 1e-8

        mask = mask.float()

        # Use the exact stored PUPPI Cartesian four-vectors. In particular, do
        # not reconstruct pz from eta or px/py from pt and phi.
        px = puppi_px
        py = puppi_py
        pz = puppi_pz

        # Predicted probabilities
        p_bkg  = probabilities[...,0] * mask
        p_chi0 = probabilities[...,1] * mask
        p_chi1 = probabilities[...,2] * mask

        # Reconstructed chi four-vectors
        chi0_px = torch.sum(p_chi0 * px, dim=1)
        chi0_py = torch.sum(p_chi0 * py, dim=1)
        chi0_pz = torch.sum(p_chi0 * pz, dim=1)
        chi0_E  = torch.sum(p_chi0 * puppi_E, dim=1)

        chi1_px = torch.sum(p_chi1 * px, dim=1)
        chi1_py = torch.sum(p_chi1 * py, dim=1)
        chi1_pz = torch.sum(p_chi1 * pz, dim=1)
        chi1_E  = torch.sum(p_chi1 * puppi_E, dim=1)

        # Probability-weighted background transverse momentum.  The
        # longitudinal component is deliberately excluded from the background
        # isotropy term because a pp collision is not isotropic along the beam.
        bkg_px = torch.sum(p_bkg * px, dim=1)
        bkg_py = torch.sum(p_bkg * py, dim=1)

        # Chi masses
        chi0_mass2 = (
            chi0_E**2
            - chi0_px**2
            - chi0_py**2
            - chi0_pz**2
        )

        chi1_mass2 = (
            chi1_E**2
            - chi1_px**2
            - chi1_py**2
            - chi1_pz**2
        )

        chi0_mass = torch.sqrt(torch.clamp(chi0_mass2, min=0.))
        chi1_mass = torch.sqrt(torch.clamp(chi1_mass2, min=0.))

        # Equal mass loss
        # L_mass = |m_chi0 - m_chi1| / (m_chi0 + m_chi1)
        mass_loss = (torch.abs(chi0_mass - chi1_mass) / (chi0_mass + chi1_mass + eps))

        # Assignment entropy
        # ------------------
        # Low entropy favors decisive per-particle assignments.  Its original
        # tuned coefficient is retained, then multiplied by an epoch-dependent
        # weight so assignments can move more freely early in training.
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=-1)
        entropy_loss = (entropy * mask).sum(dim=1) / (mask.sum(dim=1) + eps)
        entropy_weight = get_entropy_weight(epoch, num_epochs)

        # Conditional CA8 jet coherence
        # ----------------------------
        # For each jet, first condition its chi0/chi1 probabilities on the jet
        # being signal-like.  The coherence penalty is zero when that
        # conditional probability selects one chi, and 0.5 for a 50/50 split.
        # Multiplication by the mean signal probability makes an all-background
        # jet contribute zero instead of forcing it into a chi candidate.
        B, N, _ = probabilities.shape
        device = probabilities.device

        chi_probs = probabilities[..., 1:]
        labels = slimmed_CA8indices

        # Flatten the particle dimension and retain particles belonging to a
        # reconstructed CA8 jet.
        flat_probs = chi_probs.reshape(-1, 2)
        flat_labels = labels.reshape(-1)

        # Event index for every particle
        event_ids = torch.arange(B, device=device).repeat_interleave(N)

        # Remove padded particles and particles not in a slimmed-flow CA8 jet.
        valid = (flat_labels >= 0) & mask.bool().reshape(-1)

        flat_probs = flat_probs[valid]
        flat_labels = flat_labels[valid]
        event_ids = event_ids[valid]

        split_loss = torch.zeros(B, device=device)

        if flat_labels.numel() > 0:
            # Build globally unique jet indices using only valid CA8 indices.
            max_jets = int(flat_labels.max().item()) + 1
            global_jets = event_ids * max_jets + flat_labels
            num_global_jets = B * max_jets

            jet_sum = torch.zeros(num_global_jets, 2, device=device)
            jet_sum.index_add_(0, global_jets, flat_probs)

            jet_count = torch.zeros(num_global_jets, device=device)
            jet_count.index_add_(
                0,
                global_jets,
                torch.ones_like(global_jets, dtype=torch.float),
            )

            jet_signal_sum = jet_sum.sum(dim=1)
            jet_conditional = jet_sum / (jet_signal_sum.unsqueeze(1) + eps)
            jet_coherence = 1.0 - (jet_conditional ** 2).sum(dim=1)
            jet_signal_fraction = jet_signal_sum / jet_count.clamp(min=1)
            jet_loss = jet_signal_fraction * jet_coherence

            valid_jets = jet_count >= 2
            jet_events = torch.arange(num_global_jets, device=device) // max_jets

            split_loss.index_add_(
                0,
                jet_events[valid_jets],
                jet_loss[valid_jets],
            )

            jets_per_event = torch.zeros(B, device=device)
            jets_per_event.index_add_(
                0,
                jet_events[valid_jets],
                torch.ones_like(jet_events[valid_jets], dtype=torch.float),
            )
            split_loss = split_loss / jets_per_event.clamp(min=1)

        # Nonempty chi candidates
        # -----------------------
        # Each chi should carry at least 25% of the event's scalar pT.  Below
        # that threshold a squared hinge rises smoothly from zero to one.  This
        # prevents the equal-mass loss from being minimized by two empty chis
        # without requiring equal particle multiplicities or a target mass.
        particle_pt = torch.sqrt(px**2 + py**2 + eps)
        event_scalar_pt = torch.sum(mask * particle_pt, dim=1)
        chi0_scalar_pt = torch.sum(p_chi0 * particle_pt, dim=1)
        chi1_scalar_pt = torch.sum(p_chi1 * particle_pt, dim=1)

        chi0_pt_fraction = chi0_scalar_pt / (event_scalar_pt + eps)
        chi1_pt_fraction = chi1_scalar_pt / (event_scalar_pt + eps)

        chi0_deficit = torch.relu(
            (MIN_CHI_PT_FRACTION - chi0_pt_fraction) / MIN_CHI_PT_FRACTION
        )
        chi1_deficit = torch.relu(
            (MIN_CHI_PT_FRACTION - chi1_pt_fraction) / MIN_CHI_PT_FRACTION
        )
        nonempty_loss = chi0_deficit.square() + chi1_deficit.square()

        # Smooth transverse background isotropy
        # --------------------------------------
        # The anisotropy ratio is gated off as the background becomes empty,
        # where its direction is undefined.  A separate bounded exponential
        # rises smoothly toward one at empty background, avoiding the previous
        # sharp divergence while still discouraging background collapse.
        bkg_vector_pt = torch.sqrt(bkg_px**2 + bkg_py**2 + eps)
        bkg_scalar_pt = torch.sum(p_bkg * particle_pt, dim=1)
        bkg_pt_fraction = bkg_scalar_pt / (event_scalar_pt + eps)

        bkg_gate = bkg_pt_fraction / (bkg_pt_fraction + BKG_EMPTY_PT_SCALE)
        bkg_anisotropy = bkg_vector_pt / (
            bkg_scalar_pt + BKG_EMPTY_PT_SCALE * event_scalar_pt + eps
        )
        bkg_empty_penalty = torch.exp(-bkg_pt_fraction / BKG_EMPTY_PT_SCALE)
        bkg_loss = bkg_gate * bkg_anisotropy + bkg_empty_penalty

        # Combine
        event_loss = (
            LAMBDA_MASS * mass_loss
            + entropy_weight * LAMBDA_ENTROPY * entropy_loss
            + LAMBDA_SPLIT * split_loss
            + LAMBDA_NONEMPTY * nonempty_loss
            + LAMBDA_BKG * bkg_loss
        )

        losses = {
            "event_loss":     event_loss,
            "loss":           event_loss.mean(),
            "mass_loss":      (LAMBDA_MASS * mass_loss).mean(),
            "entropy_loss":   (entropy_weight * LAMBDA_ENTROPY * entropy_loss).mean(),
            "split_loss":     (LAMBDA_SPLIT * split_loss).mean(),
            "nonempty_loss":  (LAMBDA_NONEMPTY * nonempty_loss).mean(),
            "bkg_loss":       (LAMBDA_BKG * bkg_loss).mean(),
            "entropy_weight": entropy_weight,
        }

        return losses
    
###########################################################################
# Epoch-dependent loss weights
###########################################################################

def get_entropy_weight(epoch, num_epochs):
    """Return 0.25 through 25% of training, then reach 1.0 at 75%."""

    if num_epochs <= 1:
        return 1.0

    training_fraction = epoch / (num_epochs - 1)
    ramp_fraction = (training_fraction - 0.25) / 0.50
    ramp_fraction = min(max(ramp_fraction, 0.0), 1.0)

    return 0.25 + 0.75 * ramp_fraction


def get_loss_weights(epoch, num_epochs):
    """
    Smoothly transition from student supervision to two-body optimization.

    Epoch 0:
        CE = 1.0
        TwoBody = 0.0

    Final epoch:
        CE = 0.1
        TwoBody = 1.0
    """

    transition_epochs = max(1, int(0.4 * num_epochs))

    progress = min(epoch / transition_epochs, 1.0)

    ce_weight = 1.0 - 0.9 * progress
    twobody_weight = progress

    return ce_weight, twobody_weight

###########################################################################
# Combined loss
###########################################################################

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce_loss = PretrainingLoss()
        self.two_body_loss = TwoBodyLoss()

    def forward(
        self,
        epoch,
        num_epochs,
        logits,
        probabilities,
        puppi_px,
        puppi_py,
        puppi_pz,
        puppi_E,
        algorithm_labels,
        algorithm_CAindices,
        mask,
    ):

        ce_losses = self.ce_loss(
            logits,
            algorithm_labels,
        )

        twobody_losses = self.two_body_loss(
            probabilities,
            puppi_px,
            puppi_py,
            puppi_pz,
            puppi_E,
            algorithm_CAindices,
            mask,
            epoch,
            num_epochs,
        )

        ce_weight, twobody_weight = get_loss_weights(epoch, num_epochs)

        loss = (ce_weight * ce_losses["loss"] + twobody_weight * twobody_losses["loss"])

        losses = {
            "loss": loss,

            "ce_loss": ce_losses["loss"],
            "twobody_loss": twobody_losses["loss"],

            "ce_weight": ce_weight,
            "twobody_weight": twobody_weight,
            "entropy_weight": twobody_losses["entropy_weight"],

            "event_loss": ce_weight * ce_losses["event_loss"] + twobody_weight * twobody_losses["event_loss"],

            "mass_loss": twobody_losses["mass_loss"],
            "entropy_loss": twobody_losses["entropy_loss"],
            "split_loss": twobody_losses["split_loss"],
            "nonempty_loss": twobody_losses["nonempty_loss"],
            "bkg_loss": twobody_losses["bkg_loss"],
        }

        return losses

###########################################################################
# Training utilities
###########################################################################

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    trainMode="student",
    epoch=None,
    num_epochs=None,
):
    """
    Train for one epoch.

    Return average loss over the epoch.
    """

    model.train()

    total_losses = {}
    total_events = 0

    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs}"
    else:
        desc = "Training"

    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:

        # Move tensors to device
        particles = batch["particles"].to(device)
        mask = batch["mask"].to(device)

        puppi_pt = batch["puppi_pt"].to(device)
        puppi_eta = batch["puppi_eta"].to(device)
        puppi_phi = batch["puppi_phi"].to(device)
        puppi_px = batch["puppi_px"].to(device)
        puppi_py = batch["puppi_py"].to(device)
        puppi_pz = batch["puppi_pz"].to(device)
        puppi_E = batch["puppi_E"].to(device)

        algorithm_labels = batch["algorithmLabel"].to(device)
        algorithm_CAindices = batch["algorithmCAIndex"].to(device)

        # Forward pass
        outputs = model(
            particles,
            puppi_pt,
            puppi_eta,
            puppi_phi,
            puppi_px,
            puppi_py,
            puppi_pz,
            puppi_E,
            mask,
        )

        logits = outputs["logits"]
        probabilities = outputs["probabilities"]

        # Compute loss
        # logits: (B,N,NUM_CLASSES)

        if trainMode == "student":
            losses = criterion(
                logits,
                algorithm_labels,
            )
        elif trainMode == "student_to_scratch":
            losses = criterion(
                epoch,
                num_epochs,
                logits,
                probabilities,
                puppi_px,
                puppi_py,
                puppi_pz,
                puppi_E,
                algorithm_labels,
                algorithm_CAindices,
                mask,
            )
        else:
            losses = criterion(
                probabilities,
                puppi_px,
                puppi_py,
                puppi_pz,
                puppi_E,
                algorithm_CAindices,
                mask,
                epoch,
                num_epochs,
            )

        # Backpropagation
        optimizer.zero_grad()

        losses["loss"].backward()

        optimizer.step()

        # Accumulate statistics
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

        postfix = {"loss": f"{losses['loss'].item():.3f}"}

        pbar.set_postfix(postfix)

    for key in total_losses:

        if "weight" in key or key == "event_loss":
            continue

        total_losses[key] /= total_events

    return total_losses

def validate(
    model,
    loader,
    criterion,
    device,
    trainMode="student",
    epoch=None,
    num_epochs=None,
):
    """
    Evaluate model on validation set.
    """

    model.eval()

    total_losses = {}
    total_events = 0

    with torch.no_grad():

        if epoch is not None and num_epochs is not None:
            desc = f"Epoch {epoch+1}/{num_epochs}"
        else:
            desc = "Training"

        pbar = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:

            particles = batch["particles"].to(device)
            mask = batch["mask"].to(device)

            puppi_pt = batch["puppi_pt"].to(device)
            puppi_eta = batch["puppi_eta"].to(device)
            puppi_phi = batch["puppi_phi"].to(device)
            puppi_px = batch["puppi_px"].to(device)
            puppi_py = batch["puppi_py"].to(device)
            puppi_pz = batch["puppi_pz"].to(device)
            puppi_E = batch["puppi_E"].to(device)

            algorithm_labels = batch["algorithmLabel"].to(device)
            algorithm_CAindices = batch["algorithmCAIndex"].to(device)

            outputs = model(
                particles,
                puppi_pt,
                puppi_eta,
                puppi_phi,
                puppi_px,
                puppi_py,
                puppi_pz,
                puppi_E,
                mask,
            )

            logits = outputs["logits"]
            probabilities = outputs["probabilities"]

            if trainMode == "student":
                losses = criterion(
                    logits,
                    algorithm_labels,
                )
            elif trainMode == "student_to_scratch":
                losses = criterion(
                    epoch,
                    num_epochs,
                    logits,
                    probabilities,
                    puppi_px,
                    puppi_py,
                    puppi_pz,
                    puppi_E,
                    algorithm_labels,
                    algorithm_CAindices,
                    mask,
                )
            else:
                losses = criterion(
                    probabilities,
                    puppi_px,
                    puppi_py,
                    puppi_pz,
                    puppi_E,
                    algorithm_CAindices,
                    mask,
                    epoch,
                    num_epochs,
                )

            batch_size = particles.shape[0]

            for key, value in losses.items():

                if "weight" in key:
                    total_losses[key] = value
                    continue

                if value.ndim > 0:
                    continue

                else:
                    total_losses[key] = total_losses.get(key, 0.0) + value.item() * batch_size

            total_events += batch_size

            postfix = {"loss": f"{losses['loss'].item():.3f}"}

            pbar.set_postfix(postfix)

        for key in total_losses:

            if "weight" in key or key == "event_loss":
                continue

            total_losses[key] /= total_events

    return total_losses

###########################################################################
# Build model dependent on mode
###########################################################################

def build_model(
    input_dim,
    checkpoint_path,
    feature_names,
    load_checkpoint=False,
):
    """
    Construct a ParticleTransformer and optionally load a checkpoint.
    """

    model = ParticleTransformer(input_dim=input_dim)

    if load_checkpoint:

        print(f"Loading checkpoint {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_input_dim = checkpoint.get("input_dim")
        if checkpoint_input_dim is not None and checkpoint_input_dim != input_dim:
            raise ValueError(
                f"Checkpoint expects {checkpoint_input_dim} particle features, "
                f"but this dataset provides {input_dim}"
            )

        checkpoint_features = checkpoint.get("particle_feature_names")
        if checkpoint_features is not None and checkpoint_features != feature_names:
            raise ValueError(
                "Checkpoint and dataset particle feature schemas do not match"
            )

        model.load_state_dict(checkpoint["model_state_dict"])

    return model

###########################################################################
# Main training function
###########################################################################

def train(
    dataset_path,
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    output_path="WbWb_4000_1000_slimmed_all_ak_constituents",
    trainMode="student",
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

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"Dataset {dataset_path!r} must have nonempty train and validation splits; "
            f"found train={len(train_dataset)}, validation={len(val_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Finished loading data")

    # Determine input feature dimension
    sample = train_dataset[0]

    input_dim = sample["particles"].shape[-1]
    feature_names = train_dataset.metadata["features"]["particle_names"]

    if input_dim != len(feature_names):
        raise ValueError(
            f"Dataset contains {input_dim} particle features, but metadata "
            f"defines {len(feature_names)}"
        )

    print(
        "Dataset schema: "
        f"{train_dataset.metadata['schema_version']}, "
        f"mode={train_dataset.metadata['selection']['mode']}, "
        f"Nparticles={train_dataset.metadata['selection']['Nparticles']}, "
        f"features={input_dim}"
    )

    # Model
    tmp_path = f"checkpoints/{output_path}.pt"

    if trainMode == "student_to_scratch":
        checkpoint_path = tmp_path.replace(
            "_student_to_scratch",
            "_student",
        )
    else:
        checkpoint_path = tmp_path

    model = build_model(
        input_dim=input_dim,
        checkpoint_path=checkpoint_path,
        feature_names=feature_names,
        load_checkpoint=trainMode == "student_to_scratch",
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Training loop
    best_val_loss = float("inf")
    all_train_losses = [None] * epochs
    all_val_losses = [None] * epochs

    for epoch in range(epochs):
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            trainMode,
            epoch,
            epochs,
        )

        val_losses = validate(
            model,
            val_loader,
            criterion,
            device,
            trainMode,
            epoch,
            epochs,
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| train loss = {train_losses['loss']:.3f} "
            f"| val loss = {val_losses['loss']:.3f} "
        )

        all_train_losses[epoch] = train_losses
        all_val_losses[epoch] = val_losses

        state_dict = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )

        checkpoint = {
                "state_epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "input_dim": input_dim,
                "particle_feature_names": feature_names,
                "dataset_schema_version": train_dataset.metadata["schema_version"],
                "dataset_selection": train_dataset.metadata["selection"],
                "dataset_name": dataset_path,
                "train_mode": trainMode,
            }

        # Keep every epoch for later comparisons.
        torch.save(
            checkpoint,
            "checkpoints/" + output_path + f"_epoch{epoch}.pt",
        )

        # Also maintain an unnumbered best-validation checkpoint. This is the
        # path consumed by student-to-scratch training.
        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            torch.save(checkpoint, tmp_path)

    # Save losses
    torch.save(
        {
            "epoch": torch.arange(1, epochs+1),
            "train_loss": all_train_losses,
            "val_loss": all_val_losses,
            "best_val_loss": best_val_loss,
            "input_dim": input_dim,
            "particle_feature_names": feature_names,
            "dataset_schema_version": train_dataset.metadata["schema_version"],
            "dataset_selection": train_dataset.metadata["selection"],
            "dataset_name": dataset_path,
            "train_mode": trainMode,
        },
        "checkpoints/" + output_path + "_losses.pt"
    )

###########################################################################
# Main
###########################################################################

def main(args):

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

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
        
        print("------------------Train------------------")
        print("Dataset path: " + job["dataset_path"])
        print("Output path: " + job["output_path"])
        print("Train mode: " + job["trainMode"])

        train(
            dataset_path=job["dataset_path"],
            pt_dir=args.pt_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_path=job["output_path"],
            trainMode=job["trainMode"],
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a Particle Transformer."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="WbWb_4000_1000_slimmed_all_ak_constituents",
        help=(
            "Preprocessed dataset stub or any _shardNNNN.pt filename; a shard "
            "filename loads all shards with the same dataset prefix"
        ),
    )

    parser.add_argument(
        "--pt-dir",
        type=Path,
        default=Path("ptfiles"),
        help="Directory containing the preprocessed .pt shards",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
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

    args = parser.parse_args()

    main(args)
