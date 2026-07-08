import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main(args):
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "train_loss" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'train_loss'.")

    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint.get("val_loss", None)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))

    epochs = range(1, len(train_loss) + 1)

    plt.plot(
        epochs,
        train_loss,
        linewidth=2,
        label="Training Loss",
    )

    if val_loss is not None:
        plt.plot(
            epochs,
            val_loss,
            linewidth=2,
            label="Validation Loss",
        )

        best_epoch = val_loss.index(min(val_loss)) + 1
        best_loss = min(val_loss)

        plt.scatter(
            best_epoch,
            best_loss,
            s=60,
            marker="*",
            label=f"Best Validation ({best_epoch})",
        )

        print(f"Best validation loss : {best_loss:.6f}")
        print(f"Best epoch           : {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    png_file = output_dir / "loss.png"
    pdf_file = output_dir / "loss.pdf"

    plt.savefig(png_file, dpi=300)
    plt.savefig(pdf_file)

    print(f"Saved {png_file}")
    print(f"Saved {pdf_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/model.pt"),
        help="Training checkpoint",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots",
    )

    args = parser.parse_args()

    main(args)