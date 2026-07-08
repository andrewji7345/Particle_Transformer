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
        chi0 / chi1 / ISR
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

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
# Particle embedding
###########################################################################

class ParticleEmbedding(nn.Module):
    """
    Embed the raw particle features into the transformer latent space.
    Used in each Particle Attention Block.

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
        0 : \DeltaR = \sqrt((y_a - y_b)^2 + (\phi_a - \phi_b)^2)
        1 : kt = min(p_{T,a}, p_{T,b}) * \DeltaR
        2 : z = min(p_{T,a}, p_{T,b}) / (p_{T,a} + p_{T,b})
        3 : m^2 = (E_a + E_b)^2 - |\vec{p_a} + \vec{p_b}|^2
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def delta_phi(phi1, phi2):
        """
        Compute wrapped \Delta\phi in [-\pi, \pi].
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
                torch.log(deltaR),
                torch.log(kt),
                torch.log(z),
                torch.log(m2),
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

        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

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
            float("-inf"),
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

        output = self.out_proj(output)

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

        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * MLP_RATIO),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM * MLP_RATIO, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(
        self,
        x,
        attn_bias,
        mask,
    ):

        # Self-attention
        x = x + self.attention(
            self.norm1(x),
            attn_bias,
            mask,
        )

        # Feed-forward network
        x = x + self.mlp(self.norm2(x))

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

        return {
            "logits": logits,
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
            "particles"       : Tensor(N,F),
            "mask"            : Tensor(N),
            "raw_pt"          : Tensor(N),
            "raw_eta"         : Tensor(N),
            "raw_phi"         : Tensor(N),
            "raw_E"           : Tensor(N),
            "truthLabel"      : Tensor(N),
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

        self.labels = dataset["truthLabel"]

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

            "labels":
                self.labels[event_idx],
        }

###########################################################################
# Dataset loading helper
###########################################################################

def load_particle_datasets(path):
    """
    Load preprocessed tensors and construct train/val/test datasets.
    """

    dataset = torch.load(
        path,
        weights_only=False,
    )

    train_dataset = ParticleTransformerDataset(
        dataset,
        dataset["train_idx"],
    )

    val_dataset = ParticleTransformerDataset(
        dataset,
        dataset["val_idx"],
    )

    test_dataset = ParticleTransformerDataset(
        dataset,
        dataset["test_idx"],
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )

###########################################################################
# Training utilities
###########################################################################

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    """
    Train for one epoch.

    Return average loss over the epoch.
    """

    model.train()

    total_loss = 0.0
    total_events = 0

    for batch in loader:

        # Move tensors to device
        particles = batch["particles"].to(device)
        mask = batch["mask"].to(device)

        raw_pt = batch["raw_pt"].to(device)
        raw_eta = batch["raw_eta"].to(device)
        raw_phi = batch["raw_phi"].to(device)
        raw_E = batch["raw_E"].to(device)

        labels = batch["labels"].to(device)

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

        # Compute loss
        # logits: (B,N,3)
        # labels: (B,N)

        loss = criterion(
            logits.reshape(-1, NUM_CLASSES),
            labels.reshape(-1),
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Accumulate statistics
        batch_size = particles.shape[0]

        total_loss += loss.item() * batch_size

        total_events += batch_size

    return total_loss / total_events

def validate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate model on validation set.
    """

    model.eval()

    total_loss = 0.0
    total_events = 0

    with torch.no_grad():

        for batch in loader:

            particles = batch["particles"].to(device)
            mask = batch["mask"].to(device)

            raw_pt = batch["raw_pt"].to(device)
            raw_eta = batch["raw_eta"].to(device)
            raw_phi = batch["raw_phi"].to(device)
            raw_E = batch["raw_E"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                particles,
                raw_pt,
                raw_eta,
                raw_phi,
                raw_E,
                mask,
            )

            logits = outputs["logits"]

            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                labels.reshape(-1),
            )

            batch_size = particles.shape[0]

            total_loss += loss.item() * batch_size

            total_events += batch_size

    return total_loss / total_events

###########################################################################
# Main training function
###########################################################################

def train(
    dataset_path,
    epochs=50,
    batch_size=64,
    learning_rate=1e-4,
    output_path,
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

    # Determine input feature dimension
    sample = train_dataset[0]

    input_dim = sample["particles"].shape[-1]

    # Model
    model = ParticleTransformer(input_dim=input_dim)

    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| train loss = {train_loss:.5f} "
            f"| val loss = {val_loss:.5f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                output_path,
            )

            print("Saved best model.")

###########################################################################
# Main
###########################################################################

def main(args):

    train(
        dataset_path=args.input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_path=args.output,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a Particle Transformer."
    )

    parser.add_argument(
        "input",
        type=str,
        default="ptfiles/WbWb_4000_1000_shard0000.pt",
        help="Input preprocessed .pt file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="checkpoints/particle_transformer.pt",
        help="Output checkpoint file",
    )

    args = parser.parse_args()

    main(args)