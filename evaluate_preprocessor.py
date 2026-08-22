#!/usr/bin/env python3
"""Plot preprocessing truncation for all-PF and ntuplizer-selected-AK inputs."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MODES = (
    "all_pf",
    "all_ak_constituents",
)
MODE_PATTERN = "|".join(re.escape(mode) for mode in sorted(MODES, key=len, reverse=True))
SHARD_PATTERN = re.compile(
    rf"^(?P<dataset>.+)_(?P<mode>{MODE_PATTERN})_shard(?P<shard>\d+)\.pt$"
)
LABELS = {
    "all_pf": "All PF",
    "all_ak_constituents": "Selected AK constituents",
}
EVENT_KEYS = ("run", "lumi", "event")
PARTICLE_KEYS = ("mask", "puppi_pt", "puppi_px", "puppi_py", "puppi_pz", "puppi_E")
TRUNCATION_KEYS = (
    "truncation_n_before_puppi_filter",
    "truncation_n_negative_puppi_weight_excluded",
    "truncation_n_before",
    "truncation_n_retained",
    "truncation_n_dropped",
    "truncation_was_applied",
    "truncation_puppi_pt_before",
    "truncation_puppi_pt_retained",
    "truncation_puppi_pt_retained_fraction",
    "truncation_puppi_energy_before",
    "truncation_puppi_energy_retained",
    "truncation_puppi_energy_retained_fraction",
)
LOAD_KEYS = EVENT_KEYS + PARTICLE_KEYS + TRUNCATION_KEYS


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def percentile(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def discover(pt_dir):
    """Return {dataset: {mode: [ordered shard paths]}}."""
    found = {}
    for path in pt_dir.glob("*.pt"):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match:
            key = (int(match["shard"]), path)
            found.setdefault(match["dataset"], {}).setdefault(match["mode"], []).append(key)
    return {
        dataset: {
            mode: [path for _, path in sorted(shards)]
            for mode, shards in by_mode.items()
        }
        for dataset, by_mode in found.items()
    }


def dataset_from_selector(selector):
    """Resolve a historical dataset prefix or any generated shard filename."""

    name = Path(selector).name
    match = SHARD_PATTERN.fullmatch(name)
    if match:
        return match["dataset"]
    if name.endswith(".pt"):
        raise ValueError(
            f"Shard filename {name!r} does not match "
            "<dataset>_<data-mode>_shardNNNN.pt"
        )
    return name


def torch_load(path):
    """Use mmap so slicing 1,000 events does not eagerly page in a large shard."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def load_mode(paths, expected_mode, max_events):
    chunks = {key: [] for key in LOAD_KEYS}
    metadata = None
    remaining = max_events
    used = 0

    for path in paths:
        if remaining == 0:
            break
        shard = torch_load(path)
        missing = sorted(set(LOAD_KEYS + ("metadata",)) - set(shard))
        if missing:
            raise KeyError(f"{path} is missing keys: {missing}")

        current = shard["metadata"]
        if current.get("schema_version") != "particle_transformer_preprocessor_v3":
            raise ValueError(f"{path}: unsupported schema {current.get('schema_version')!r}")
        if current.get("selection", {}).get("mode") != expected_mode:
            raise ValueError(f"{path}: mode disagrees with its filename")
        if metadata is None:
            metadata = current
        elif current["selection"].get("Nparticles") != metadata["selection"].get("Nparticles"):
            raise ValueError(f"{path}: inconsistent Nparticles")

        take = min(remaining, int(shard["event"].shape[0]))
        if take:
            for key in LOAD_KEYS:
                chunks[key].append(shard[key][:take].clone())
            remaining -= take
            used += 1

    if metadata is None or not chunks["event"]:
        raise RuntimeError(f"No events loaded for {expected_mode}")
    return {
        "tensors": {key: torch.cat(parts, dim=0) for key, parts in chunks.items()},
        "metadata": metadata,
        "n_events": max_events - remaining,
        "n_shards": used,
    }


def truncation_rows(data):
    rows = []
    for mode in MODES:
        if mode not in data:
            continue
        item = data[mode]
        tensors = item["tensors"]
        before = numpy(tensors["truncation_n_before"])
        dropped = numpy(tensors["truncation_n_dropped"])
        negative = numpy(tensors["truncation_n_negative_puppi_weight_excluded"])
        pt_fraction = numpy(tensors["truncation_puppi_pt_retained_fraction"])
        energy_fraction = numpy(tensors["truncation_puppi_energy_retained_fraction"])
        rows.append(
            {
                "mode": mode,
                "events": item["n_events"],
                "Nparticles": item["metadata"]["selection"]["Nparticles"],
                "fraction_events_truncated": float(np.mean(dropped > 0)),
                "median_particles_before": float(np.median(before)),
                "p95_particles_before": percentile(before, 95),
                "mean_particles_dropped": float(np.mean(dropped)),
                "p95_particles_dropped": percentile(dropped, 95),
                "fraction_events_negative_puppi_excluded": float(np.mean(negative > 0)),
                "mean_negative_puppi_excluded": float(np.mean(negative)),
                "median_pt_retained_fraction": float(np.median(pt_fraction)),
                "p05_pt_retained_fraction": percentile(pt_fraction, 5),
                "fraction_events_below_99pct_pt": float(np.mean(pt_fraction < 0.99)),
                "median_energy_retained_fraction": float(np.median(energy_fraction)),
                "p05_energy_retained_fraction": percentile(energy_fraction, 5),
                "fraction_events_below_99pct_energy": float(np.mean(energy_fraction < 0.99)),
            }
        )
    return rows


def plot_truncation(data, output_dir):
    modes = [mode for mode in MODES if mode in data]
    labels = [LABELS[mode] for mode in modes]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(modes)))
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    truncated = [
        np.mean(numpy(data[mode]["tensors"]["truncation_n_dropped"]) > 0)
        for mode in modes
    ]
    axes[0, 0].bar(labels, truncated, color=colors)
    axes[0, 0].set_ylabel("Fraction of events")
    axes[0, 0].set_title("Events affected by top-N truncation")

    largest = max(int(numpy(data[mode]["tensors"]["truncation_n_before"]).max()) for mode in modes)
    bins = np.arange(largest + 2) - 0.5
    for mode, label, color in zip(modes, labels, colors):
        axes[0, 1].hist(
            numpy(data[mode]["tensors"]["truncation_n_before"]),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
    for limit in {data[mode]["metadata"]["selection"]["Nparticles"] for mode in modes}:
        axes[0, 1].axvline(limit, color="black", linestyle="--", alpha=0.6)
    axes[0, 1].set(xlabel="Particles before truncation", ylabel="Event density")
    axes[0, 1].set_title("Selected particle multiplicity")
    axes[0, 1].legend()

    for axis, key, title in (
        (axes[1, 0], "truncation_puppi_pt_retained_fraction", r"Retained PUPPI $p_T$"),
        (axes[1, 1], "truncation_puppi_energy_retained_fraction", "Retained PUPPI energy"),
    ):
        values = [numpy(data[mode]["tensors"][key]) for mode in modes]
        axis.boxplot(values, showfliers=False)
        axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
        axis.set(ylabel="Retained fraction", ylim=(0, 1.02))
        axis.set_title(title)

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "truncation_overview.png", dpi=200)
    plt.close(figure)


def evaluate(dataset, paths, output_root, max_events):
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for mode in MODES:
        if mode not in paths:
            print(f"[WARNING] {dataset}: no files for {mode}")
            continue
        data[mode] = load_mode(paths[mode], mode, max_events)
        print(f"[INFO] {dataset}: {data[mode]['n_events']} {mode} events")

    truncation = truncation_rows(data)
    with (output_dir / "truncation_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(truncation[0]))
        writer.writeheader()
        writer.writerows(truncation)
    plot_truncation(data, output_dir)

    with (output_dir / "summary.txt").open("w") as output:
        output.write(f"Dataset: {dataset}\nMaximum events per mode: {max_events}\n\n")
        output.write("TRUNCATION\n")
        for row in truncation:
            output.write(
                f"{row['mode']}: truncated={row['fraction_events_truncated']:.2%}, "
                f"p05 retained pT={row['p05_pt_retained_fraction']:.4f}, "
                f"p05 retained energy={row['p05_energy_retained_fraction']:.4f}\n"
            )
    print(f"[INFO] Wrote {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-dir", default="ptfiles")
    parser.add_argument("--output-dir", default="preprocessor_evaluation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset prefix or any generated shard filename/path; may be "
            "repeated to evaluate multiple datasets"
        ),
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every dataset family discovered in --pt-dir",
    )
    parser.add_argument("--max-events", type=int, default=1000)
    args = parser.parse_args()
    if args.max_events <= 0:
        parser.error("--max-events must be positive")

    pt_dir = Path(args.pt_dir)
    if not pt_dir.is_dir():
        parser.error(f"PT directory not found: {pt_dir}")
    found = discover(pt_dir)
    if args.all:
        datasets = sorted(found)
    elif args.dataset:
        try:
            datasets = list(dict.fromkeys(dataset_from_selector(item) for item in args.dataset))
        except ValueError as error:
            parser.error(str(error))
    else:
        parser.error("specify --dataset or --all")
    missing = sorted(set(datasets) - set(found))
    if missing:
        parser.error(f"Dataset prefix(es) not found: {missing}")
    if not datasets:
        parser.error("No <dataset>_<mode>_shardNNNN.pt files found")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        evaluate(dataset, found[dataset], output_root, args.max_events)


if __name__ == "__main__":
    main()
