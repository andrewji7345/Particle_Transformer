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
    8 x ParticleAttentionBlock
        ↓
    ParticleClassifier
        ↓
    Per-particle logits:
        bkg / chi0 / chi1
"""

import math
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

FRAC_OCCUPANCY = 0.1

LAMBDA_MASS      = 1.0
LAMBDA_ENTROPY   = 0.2
LAMBDA_OCCUPANCY = 0.5
LAMBDA_SPLIT     = 1.0
LAMBDA_BKG       = 0.5

###########################################################################
# Particle embedding
###########################################################################

class ParticleEmbedding(nn.Module):
    """
    Embed the raw particle features into the transformer latent space.

    Input:
        (B, N, F)

    Output:
        (B, N, EMBED_DIM)

    B is the batch size, N is the number of particles.
    F is the number of per-particle input features.

    Following the ParT paper:
        Linear(F -> 128)
        LayerNorm
        GELU
        Linear(128 -> 512)
        LayerNorm
        GELU
        Linear(512 -> 128)
    """

    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 512),
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
        raw_pt  : (B, N)
        raw_eta : (B, N)
        raw_phi : (B, N)
        raw_E   : (B, N)
        mask    : (B, N)

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
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        mask,
    ):

        B, N = raw_pt.shape

        ############################################################
        # Construct pairwise tensors
        ############################################################

        pt_i = raw_pt.unsqueeze(2)
        pt_j = raw_pt.unsqueeze(1)

        eta_i = raw_eta.unsqueeze(2)
        eta_j = raw_eta.unsqueeze(1)

        phi_i = raw_phi.unsqueeze(2)
        phi_j = raw_phi.unsqueeze(1)

        E_i = raw_E.unsqueeze(2)
        E_j = raw_E.unsqueeze(1)

        deta = eta_i - eta_j

        dphi = self.delta_phi(phi_i, phi_j)

        deltaR = torch.sqrt(deta**2 + dphi**2 + 1e-8)

        kt = torch.minimum(pt_i, pt_j) * deltaR

        z = torch.minimum(pt_i, pt_j) / (pt_i + pt_j + 1e-8)

        m2 = (2.0 * E_i * E_j * (torch.cosh(deta) - torch.cos(dphi)))

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
        raw_pt    : (B, N)
        raw_eta   : (B, N)
        raw_phi   : (B, N)
        raw_E     : (B, N)
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
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        mask,
    ):
        # Particle embedding
        x = self.particle_embedding(particles)

        # Pairwise interaction variables
        pair_features = self.pair_builder(
            raw_pt,
            raw_eta,
            raw_phi,
            raw_E,
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

    Expected input dictionary format:

        {
            "particles"          : Tensor(N,F),
            "mask"               : Tensor(N),
            "raw_pt"             : Tensor(N),
            "raw_eta"            : Tensor(N),
            "raw_phi"            : Tensor(N),
            "raw_E"              : Tensor(N),
            "truthLabel"         : Tensor(N),
            "algorithmLabel"     : Tensor(N),
            "algorithmCA8Label"  : Tensor(N),
        }
    """

    def __init__(
        self,
        dataset,
        indices,
    ):

        super().__init__()

        self.particles = dataset["particles"]
        self.mask = dataset["mask"]

        self.raw_pt = dataset["raw_pt"]
        self.raw_eta = dataset["raw_eta"]
        self.raw_phi = dataset["raw_phi"]
        self.raw_E = dataset["raw_E"]

        self.truth_labels = dataset["truthLabel"]
        self.algorithm_labels = dataset["algorithmLabel"]
        self.algorithm_CA8labels = dataset["algorithmCA8Label"]

        # Only keep requested split indices
        self.indices = indices


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, idx):

        event_idx = self.indices[idx]

        return {

            "particles":
                self.particles[event_idx],

            "mask":
                self.mask[event_idx],

            "raw_pt":
                self.raw_pt[event_idx],

            "raw_eta":
                self.raw_eta[event_idx],

            "raw_phi":
                self.raw_phi[event_idx],

            "raw_E":
                self.raw_E[event_idx],

            "truthLabel":
                self.truth_labels[event_idx],

            "algorithmLabel":
                self.algorithm_labels[event_idx],
                
            "algorithmCA8Label":
                self.algorithm_CA8labels[event_idx],
        }

###########################################################################
# Dataset loading helper
###########################################################################

def load_particle_datasets(dataset_name, pt_dir="ptfiles"):
    """
    Load all shards for a dataset and construct train/val/test datasets.
    """

    pt_dir = Path(pt_dir)

    shard_paths = sorted(pt_dir.glob(f"{dataset_name}_shard*.pt"))

    if len(shard_paths) == 0:
        raise FileNotFoundError(
            f"No shards found matching {dataset_name}_shard*.pt"
        )

    datasets = []

    for path in shard_paths:
        datasets.append(torch.load(path, weights_only=False))

    # Keys that should NOT simply be concatenated
    index_keys = {
        "train_idx",
        "val_idx",
        "test_idx",
    }

    merged = {}

    for key in datasets[0].keys():
        if key not in index_keys:
            merged[key] = torch.cat(
                [d[key] for d in datasets],
                dim=0,
            )

    # Merge indices with proper offsets
    train_idx = []
    val_idx = []
    test_idx = []

    offset = 0

    for d in datasets:

        n_events = d["particles"].shape[0]

        train_idx.append(d["train_idx"] + offset)
        val_idx.append(d["val_idx"] + offset)
        test_idx.append(d["test_idx"] + offset)

        offset += n_events

    merged["train_idx"] = torch.cat(train_idx)
    merged["val_idx"] = torch.cat(val_idx)
    merged["test_idx"] = torch.cat(test_idx)

    train_dataset = ParticleTransformerDataset(merged, merged["train_idx"])
    val_dataset = ParticleTransformerDataset(merged, merged["val_idx"])
    test_dataset = ParticleTransformerDataset(merged, merged["test_idx"])

    return (train_dataset, val_dataset, test_dataset)

###########################################################################
# Cross entropy loss (used in pretraining)
###########################################################################

class PretrainingLoss(nn.Module):

    """
    Cross entropy against algorithm labels.
    """

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(
        self,
        logits,
        algorithm_labels,
    ):

        B, N = algorithm_labels.shape

        swapped_labels = algorithm_labels.clone()
        swapped_labels[algorithm_labels == 1] = 2
        swapped_labels[algorithm_labels == 2] = 1

        particle_loss = self.ce(
            logits.reshape(-1, NUM_CLASSES),
            algorithm_labels.reshape(-1),
        ).reshape(B, N)

        swapped_particle_loss = self.ce(
            logits.reshape(-1, NUM_CLASSES),
            swapped_labels.reshape(-1),
        ).reshape(B, N)

        valid_mask = (algorithm_labels != -1)

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

    L = L_mass + L_entropy + L_split + L_occupancy + L_bkg_anisotropy
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        probabilities,
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        algorithm_CA8labels,
        mask,
    ):

        eps = 1e-8

        mask = mask.float()

        # Particle four-vectors
        px = raw_pt * torch.cos(raw_phi)
        py = raw_pt * torch.sin(raw_phi)
        pz = raw_pt * torch.sinh(raw_eta)

        # Predicted probabilities
        p_bkg  = probabilities[...,0] * mask
        p_chi0 = probabilities[...,1] * mask
        p_chi1 = probabilities[...,2] * mask

        # Reconstructed chi four-vectors
        chi0_px = torch.sum(p_chi0 * px, dim=1)
        chi0_py = torch.sum(p_chi0 * py, dim=1)
        chi0_pz = torch.sum(p_chi0 * pz, dim=1)
        chi0_E  = torch.sum(p_chi0 * raw_E, dim=1)

        chi1_px = torch.sum(p_chi1 * px, dim=1)
        chi1_py = torch.sum(p_chi1 * py, dim=1)
        chi1_pz = torch.sum(p_chi1 * pz, dim=1)
        chi1_E  = torch.sum(p_chi1 * raw_E, dim=1)

        # Reconstructed bkg four-vector

        bkg_px = torch.sum(p_bkg * px, dim=1)
        bkg_py = torch.sum(p_bkg * py, dim=1)
        bkg_pz = torch.sum(p_bkg * pz, dim=1)
        bkg_E  = torch.sum(p_bkg * raw_E, dim=1)

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

        # Entropy loss
        # L_entropy = - (1/n_particles) Sum (p ln(p))
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=-1)

        entropy_loss = (entropy * mask).sum(dim=1) / (mask.sum(dim=1) + eps)

        # Jet splitting loss
        # L_split = 1 - 1/(n_particles^2) |p|^2
        # Equivalent to average pairwise agreement between CA8 jet constituents
        B, N, _ = probabilities.shape
        device = probabilities.device

        # Ignore bkg label
        probs = probabilities[..., 1:]
        labels = algorithm_CA8labels

        # Flatten particle dimension
        flat_probs = probs.reshape(-1, 2)
        flat_labels = labels.reshape(-1)

        # Event index for every particle
        event_ids = torch.arange(B, device=device).repeat_interleave(N)

        # Remove padded particles & particles not in a CA8 jet
        valid = flat_labels >= 0

        flat_probs = flat_probs[valid]
        flat_labels = flat_labels[valid]
        event_ids = event_ids[valid]

        # Build globally unique jet indices
        max_jets = int(labels.max().item()) + 1
        global_jets = event_ids * max_jets + flat_labels

        num_global_jets = B * max_jets

        # Sum probabilities for each jet
        jet_sum = torch.zeros(num_global_jets, 2, device=device)
        jet_sum.index_add_(0, global_jets, flat_probs)

        # Number of particles in each jet
        jet_count = torch.zeros(num_global_jets, device=device)
        jet_count.index_add_(0, global_jets, torch.ones_like(global_jets, dtype=torch.float))

        # Mean assignment vector for each jet
        jet_mean = jet_sum / jet_count.clamp(min=1).unsqueeze(1)

        # 1 - ||mean||^2
        jet_loss = 1.0 - (jet_mean ** 2).sum(dim=1)

        # Ignore jets with fewer than two particles
        valid_jets = jet_count >= 2

        # Event index for every global jet
        jet_events = torch.arange(num_global_jets, device=device) // max_jets

        split_loss = torch.zeros(B, device=device)

        split_loss.index_add_(0, jet_events[valid_jets], jet_loss[valid_jets])

        jets_per_event = torch.zeros(B, device=device)
        jets_per_event.index_add_(0, jet_events[valid_jets], torch.ones_like(jet_events[valid_jets], dtype=torch.float))

        split_loss = split_loss / jets_per_event.clamp(min=1)

        # occupancy loss
        # L_occupancy = e^(-N_chi0 / N_occupancy) + e^(-N_chi1 / N_occupancy)
        valid_particles = mask.sum(dim=1)

        n_occupancy = FRAC_OCCUPANCY * valid_particles.clamp(min=1)

        chi0_occ = p_chi0.sum(dim=1)
        chi1_occ = p_chi1.sum(dim=1)

        occupancy_loss = (torch.exp(-chi0_occ / (n_occupancy + eps)) + torch.exp(-chi1_occ / (n_occupancy + eps)))

        # bkg anisotropy loss
        bkg_vector = torch.sqrt(bkg_px**2 + bkg_py**2 + bkg_pz**2 + eps)

        particle_p = torch.sqrt(px**2 + py**2 + pz**2 + eps)

        bkg_scalar = torch.sum(p_bkg * particle_p, dim=1)

        bkg_loss = bkg_vector / (bkg_scalar + eps)

        # Combine
        event_loss = (
            LAMBDA_MASS * mass_loss
            + LAMBDA_ENTROPY * entropy_loss
            + LAMBDA_SPLIT * split_loss
            + LAMBDA_OCCUPANCY * occupancy_loss
            + LAMBDA_BKG * bkg_loss
        )

        losses = {
            "event_loss":     event_loss,
            "loss":           event_loss.mean(),
            "mass_loss":      (LAMBDA_MASS * mass_loss).mean(),
            "entropy_loss":   (LAMBDA_ENTROPY * entropy_loss).mean(),
            "split_loss":     (LAMBDA_SPLIT * split_loss).mean(),
            "occupancy_loss":     (LAMBDA_OCCUPANCY * occupancy_loss).mean(),
            "bkg_loss":       (LAMBDA_BKG * bkg_loss).mean(),
        }

        return losses
    
###########################################################################
# Get loss weights if using combined loss
###########################################################################
    
def get_loss_weights(epoch, num_epochs):
    """
    Smoothly transition from CE pretraining to two-body optimization.

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
        raw_pt,
        raw_eta,
        raw_phi,
        raw_E,
        algorithm_labels,
        algorithm_CA8labels,
        mask,
    ):

        ce_losses = self.ce_loss(
            logits,
            algorithm_labels,
        )

        twobody_losses = self.two_body_loss(
            probabilities,
            raw_pt,
            raw_eta,
            raw_phi,
            raw_E,
            algorithm_CA8labels,
            mask,
        )

        ce_weight, twobody_weight = get_loss_weights(epoch, num_epochs)

        loss = (ce_weight * ce_losses["loss"] + twobody_weight * twobody_losses["loss"])

        losses = {
            "loss": loss,

            "ce_loss": ce_losses["loss"],
            "twobody_loss": twobody_losses["loss"],

            "ce_weight": ce_weight,
            "twobody_weight": twobody_weight,

            "event_loss": ce_weight * ce_losses["event_loss"] + twobody_weight * twobody_losses["event_loss"],

            "mass_loss": twobody_losses["mass_loss"],
            "entropy_loss": twobody_losses["entropy_loss"],
            "split_loss": twobody_losses["split_loss"],
            "occupancy_loss": twobody_losses["occupancy_loss"],
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
    trainMode="pretrain",
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

        raw_pt = batch["raw_pt"].to(device)
        raw_eta = batch["raw_eta"].to(device)
        raw_phi = batch["raw_phi"].to(device)
        raw_E = batch["raw_E"].to(device)

        algorithm_labels = batch["algorithmLabel"].to(device)
        algorithm_CA8labels = batch["algorithmCA8Label"].to(device)

        # Forward pass
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

        # Compute loss
        # logits: (B,N,NUM_CLASSES)

        if trainMode == "pretrain":
            losses = criterion(
                logits,
                algorithm_labels,
            )
        elif trainMode == "combined":
            losses = criterion(
                epoch,
                num_epochs,
                logits,
                probabilities,
                raw_pt,
                raw_eta,
                raw_phi,
                raw_E,
                algorithm_labels,
                algorithm_CA8labels,
                mask,
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

        postfix = {"loss": f"{losses["loss"].item():.3f}"}

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
    trainMode="pretrain",
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

            raw_pt = batch["raw_pt"].to(device)
            raw_eta = batch["raw_eta"].to(device)
            raw_phi = batch["raw_phi"].to(device)
            raw_E = batch["raw_E"].to(device)

            algorithm_labels = batch["algorithmLabel"].to(device)
            algorithm_CA8labels = batch["algorithmCA8Label"].to(device)

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

            if trainMode == "pretrain":
                losses = criterion(
                    logits,
                    algorithm_labels,
                )
            elif trainMode == "combined":
                losses = criterion(
                    epoch,
                    num_epochs,
                    logits,
                    probabilities,
                    raw_pt,
                    raw_eta,
                    raw_phi,
                    raw_E,
                    algorithm_labels,
                    algorithm_CA8labels,
                    mask,
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

            for key, value in losses.items():

                if "weight" in key:
                    total_losses[key] = value
                    continue

                if value.ndim > 0:
                    continue

                else:
                    total_losses[key] = total_losses.get(key, 0.0) + value.item() * batch_size

            total_events += batch_size

            postfix = {"loss": f"{losses["loss"].item():.3f}"}

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
    trainMode,
    checkpoint_path,
):
    """
    Construct a ParticleTransformer and optionally
    load pretrained weights.
    """

    model = ParticleTransformer(input_dim=input_dim)

    if trainMode == "use_pretrained" or trainMode == "combined":

        print(f"Loading checkpoint {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

    return model

###########################################################################
# Main training function
###########################################################################

def train(
    dataset_path,
    epochs=50,
    batch_size=64,
    learning_rate=1e-4,
    output_path="test_model",
    trainMode="pretrain",
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

    # Model
    tmp_path = f"checkpoints/{output_path}.pt"

    if trainMode == "use_pretrained":
        checkpoint_path = tmp_path.replace(
            "_use_pretrained",
            "_pretrain",
        )
    elif trainMode == "combined":
        checkpoint_path = tmp_path.replace(
            "_combined",
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
    elif trainMode == "combined":
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
            f"| train loss = {train_losses["loss"]:.3f} "
            f"| val loss = {val_losses["loss"]:.3f} "
        )

        all_train_losses[epoch] = train_losses
        all_val_losses[epoch] = val_losses

        # Save best checkpoint
        if val_losses["loss"] < best_val_loss:

            best_val_loss = val_losses["loss"]

            state_dict = (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            )

            torch.save({
                    "state_epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "checkpoints/" + output_path + ".pt",
            )

    # Save losses
    torch.save(
        {
            "epoch": torch.arange(1, epochs+1),
            "train_loss": all_train_losses,
            "val_loss": all_val_losses,
        },
        "checkpoints/" + output_path + "_losses.pt"
    )

###########################################################################
# Main
###########################################################################

def main(args):

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # Assign data modes, pretraining modes
    dataModes = (
        ["all_pf", "ak8_constituents", "ak4_constituents", "all_constituents"]
        if args.dataMode == "all"
        else [args.dataMode]
    )

    trainModes = (
        ["pretrain", "use_pretrained", "no_use_pretrained", "combined"]
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
        
        print("------------------Train------------------")
        print("Dataset path: " + job["dataset_path"])
        print("Output path: " + job["output_path"])
        print("Train mode: " + job["trainMode"])

        train(
            dataset_path=job["dataset_path"],
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
        default="WbWb_4000_1000",
        help="Input preprocessed filename stub",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
            "combined",
            "all"
        ],
        default="no_use_pretrained",
        help="Either pretrain, use_pretrained, no_use_pretrained, combined, or all",
    )

    args = parser.parse_args()

    main(args)
