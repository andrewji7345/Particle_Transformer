import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main(args):
    # Load checkpoint
    checkpoint = torch.load(args.loss_checkpoint, map_location="cpu")

    required_keys = [
        "train_loss",
        "val_loss",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain '{key}'.")

    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    # Create figure and two y axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Loss axis
    loss_train_line, = ax1.plot(
        epochs,
        train_loss,
        linewidth=2,
        label="Training Loss",
    )

    loss_val_line = None
    if val_loss is not None:
        loss_val_line, = ax1.plot(
            epochs,
            val_loss,
            linewidth=2,
            label="Validation Loss",
        )

        best_epoch = torch.argmin(torch.tensor(val_loss)).item() + 1
        best_loss = min(val_loss)

        ax1.scatter(
            best_epoch,
            best_loss,
            s=60,
            marker="*",
            label=f"Best Validation ({best_epoch})",
        )

        print(f"Best validation loss : {best_loss:.6f}")
        print(f"Best epoch           : {best_epoch}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Combine legends from both axes
    lines = [
        loss_train_line,
        loss_val_line,
    ]

    lines = [line for line in lines if line is not None]

    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="center right",
    )

    plt.title("Training History")
    plt.tight_layout()

    png_file = output_dir / "training_history.png"
    pdf_file = output_dir / "training_history.pdf"

    plt.savefig(png_file, dpi=300)
    plt.savefig(pdf_file)

    print(f"Saved {png_file}")
    print(f"Saved {pdf_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loss-checkpoint",
        type=Path,
        default=Path("checkpoints/test_model_losses.pt"),
        help="Training history checkpoint",
    )

    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=Path("checkpoints/test_model.pt"),
        help="Model checkpoint",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots",
    )

    args = parser.parse_args()

    main(args)