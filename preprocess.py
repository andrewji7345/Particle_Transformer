"""
Preprocess ROOT ntuples for the Suu -> chi chi particle transformer.

The preprocessor uses the PUPPI four-vectors and the single algorithm teacher
stored by the ntuplizer, assigns deterministic dataset splits from the event
identity, and writes enough schema/truncation metadata to audit the resulting
shards.

Example:

python preprocess_particles.py \
    --input /path/to/WbWb_4000_1000_slimmed.root \
    --output ptfiles/WbWb_4000_1000_slimmed.pt \
    --data-mode all_ak_constituents \
    --Nparticles 256

To preprocess every ROOT file and both particle representations:

python preprocess.py \
    --all-input \
    --all-data-modes \
    --output ptfiles \
    --root-dir /path/to/root/files
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot


EPS = 1.0e-6
SCHEMA_VERSION = "particle_transformer_preprocessor_v3"

FEATURE_NAMES = [
    "log_puppi_pt",
    "puppi_eta",
    "sin_puppi_phi",
    "cos_puppi_phi",
    "log_puppi_energy",
    "charge",
    "is_electron",
    "is_muon",
    "is_photon",
    "is_charged_hadron",
    "is_neutral_hadron",
    "tanh_dxy",
    "tanh_dz",
    "puppi_weight",
    "from_pv",
    "pv_association_quality",
]

GLOBAL_FEATURE_NAMES = [
    "HT",
    "MET_pt",
    "MET_phi",
    "rho",
    "nPV",
    "PV_x",
    "PV_y",
    "PV_z",
]

GENERATOR_P4_FIELDS = [
    "px",
    "py",
    "pz",
    "E",
    "pt",
    "eta",
    "phi",
    "mass",
]

MODE_NAMES = [
    "all_pf",
    "all_ak_constituents",
]

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test",
}


def pad_array(array, Nparticles, pad_value=0.0):
    """Pad or truncate a jagged particle array to ``Nparticles``."""

    padded = ak.pad_none(array, Nparticles, axis=1, clip=True)
    padded = ak.fill_none(padded, pad_value)
    return ak.to_numpy(padded)


def build_padding_mask(reference, Nparticles):
    """Return True for real particles and False for padded entries."""

    real_particles = ak.ones_like(reference, dtype=np.bool_)
    return pad_array(
        real_particles,
        Nparticles,
        pad_value=False,
    ).astype(np.bool_)


def particle_flags(pdgid):
    """Build the five particle-type indicator features from PDG ID."""

    absid = abs(pdgid)

    is_electron = ak.values_astype(absid == 11, np.float32)
    is_muon = ak.values_astype(absid == 13, np.float32)
    is_photon = ak.values_astype(absid == 22, np.float32)

    is_charged_hadron = ak.values_astype(
        (absid == 211) | (absid == 321) | (absid == 2212),
        np.float32,
    )
    is_neutral_hadron = ak.values_astype(
        (absid == 130) | (absid == 2112),
        np.float32,
    )

    return (
        is_electron,
        is_muon,
        is_photon,
        is_charged_hadron,
        is_neutral_hadron,
    )


def build_mode_particle_mask(arrays, mode):
    """Build an input mask from the ntuplizer-defined algorithm membership."""

    if mode == "all_pf":
        return ak.ones_like(arrays["particle_puppi_pt"], dtype=np.bool_)

    if mode == "all_ak_constituents":
        # A non-negative index means the particle belongs to one of the AK
        # jets that passed the threshold in this ntuple's selected algorithm.
        return arrays["particle_algorithmAKIndex"] >= 0

    raise ValueError(f"Invalid data mode: {mode}")


def prepare_particle_selection(arrays, mode):
    """Apply the mode and PUPPI filters, then sort particles by PUPPI pT."""

    mode_mask = build_mode_particle_mask(arrays, mode)

    # This is deliberately '< 0', rather than '<= 0', to match the requested
    # dataset definition. Zero-weight candidates remain present with a zero p4.
    valid_puppi_weight = arrays["particle_puppiWeight"] >= 0.0
    particle_mask = mode_mask & valid_puppi_weight

    selected_pt = arrays["particle_puppi_pt"][particle_mask]
    order = ak.argsort(selected_pt, axis=1, ascending=False)

    return {
        "mode_mask": mode_mask,
        "particle_mask": particle_mask,
        "order": order,
    }


def select_and_order(array, selection):
    """Apply the common particle selection and PUPPI-pT order."""

    return array[selection["particle_mask"]][selection["order"]]


def build_features(
    pt,
    eta,
    phi,
    energy,
    charge,
    pdgid,
    dxy,
    dz,
    puppi_weight,
    from_pv,
    pv_association_quality,
    Nparticles,
):
    """Build and pad the transformer particle-feature tensor."""

    logpt = np.log(np.maximum(pt, 0.0) + EPS)
    log_energy = np.log(np.maximum(energy, 0.0) + EPS)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    tanhdxy = np.tanh(dxy)
    tanhdz = np.tanh(dz)

    (
        is_electron,
        is_muon,
        is_photon,
        is_charged_hadron,
        is_neutral_hadron,
    ) = particle_flags(pdgid)

    features = np.stack(
        [
            pad_array(logpt, Nparticles),
            pad_array(eta, Nparticles),
            pad_array(sinphi, Nparticles),
            pad_array(cosphi, Nparticles),
            pad_array(log_energy, Nparticles),
            pad_array(charge, Nparticles),
            pad_array(is_electron, Nparticles),
            pad_array(is_muon, Nparticles),
            pad_array(is_photon, Nparticles),
            pad_array(is_charged_hadron, Nparticles),
            pad_array(is_neutral_hadron, Nparticles),
            pad_array(tanhdxy, Nparticles),
            pad_array(tanhdz, Nparticles),
            pad_array(puppi_weight, Nparticles),
            pad_array(from_pv, Nparticles),
            pad_array(pv_association_quality, Nparticles),
        ],
        axis=-1,
    )

    return features.astype(np.float32)


def retained_fraction(retained, total):
    """Compute a safe retained fraction, defining an empty total as 1."""

    return np.divide(
        retained,
        total,
        out=np.ones_like(retained, dtype=np.float32),
        where=total > 0.0,
    ).astype(np.float32)


def build_truncation_diagnostics(selection, ordered_pt, ordered_energy, Nparticles):
    """Build event-level diagnostics for the selection and top-pT truncation."""

    n_before_puppi_filter = ak.to_numpy(
        ak.sum(selection["mode_mask"], axis=1)
    ).astype(np.int64)
    n_before_truncation = ak.to_numpy(ak.num(ordered_pt, axis=1)).astype(np.int64)
    n_retained = np.minimum(n_before_truncation, Nparticles).astype(np.int64)
    n_truncated = (n_before_truncation - n_retained).astype(np.int64)

    retained_pt = ordered_pt[:, :Nparticles]
    retained_energy = ordered_energy[:, :Nparticles]

    pt_before = ak.to_numpy(ak.sum(ordered_pt, axis=1)).astype(np.float32)
    pt_retained = ak.to_numpy(ak.sum(retained_pt, axis=1)).astype(np.float32)
    energy_before = ak.to_numpy(ak.sum(ordered_energy, axis=1)).astype(np.float32)
    energy_retained = ak.to_numpy(ak.sum(retained_energy, axis=1)).astype(np.float32)

    return {
        "truncation_n_before_puppi_filter": n_before_puppi_filter,
        "truncation_n_negative_puppi_weight_excluded": (
            n_before_puppi_filter - n_before_truncation
        ).astype(np.int64),
        "truncation_n_before": n_before_truncation,
        "truncation_n_retained": n_retained,
        "truncation_n_dropped": n_truncated,
        "truncation_was_applied": (n_truncated > 0),
        "truncation_puppi_pt_before": pt_before,
        "truncation_puppi_pt_retained": pt_retained,
        "truncation_puppi_pt_retained_fraction": retained_fraction(
            pt_retained,
            pt_before,
        ),
        "truncation_puppi_energy_before": energy_before,
        "truncation_puppi_energy_retained": energy_retained,
        "truncation_puppi_energy_retained_fraction": retained_fraction(
            energy_retained,
            energy_before,
        ),
    }


def process_events(arrays, selection, Nparticles):
    """Build particle inputs, exact stored PUPPI p4s, and global features."""

    puppi_pt = select_and_order(arrays["particle_puppi_pt"], selection)
    puppi_eta = select_and_order(arrays["particle_puppi_eta"], selection)
    puppi_phi = select_and_order(arrays["particle_puppi_phi"], selection)
    puppi_energy = select_and_order(arrays["particle_puppi_energy"], selection)
    puppi_px = select_and_order(arrays["particle_puppi_px"], selection)
    puppi_py = select_and_order(arrays["particle_puppi_py"], selection)
    puppi_pz = select_and_order(arrays["particle_puppi_pz"], selection)

    charge = select_and_order(arrays["particle_charge"], selection)
    pdgid = select_and_order(arrays["particle_pdgId"], selection)
    dxy = select_and_order(arrays["particle_dxy"], selection)
    dz = select_and_order(arrays["particle_dz"], selection)
    puppi_weight = select_and_order(arrays["particle_puppiWeight"], selection)
    from_pv = select_and_order(arrays["particle_fromPV"], selection)
    pv_association_quality = select_and_order(
        arrays["particle_pvAssociationQuality"],
        selection,
    )

    features = build_features(
        puppi_pt,
        puppi_eta,
        puppi_phi,
        puppi_energy,
        charge,
        pdgid,
        dxy,
        dz,
        puppi_weight,
        from_pv,
        pv_association_quality,
        Nparticles,
    )

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

    diagnostics = build_truncation_diagnostics(
        selection,
        puppi_pt,
        puppi_energy,
        Nparticles,
    )

    return {
        "particles": features,
        "mask": build_padding_mask(puppi_pt, Nparticles),
        # These are copied from the stored PUPPI four-vector branches. No p4 is
        # reconstructed from (pt, eta, phi) in this preprocessor.
        "puppi_px": pad_array(puppi_px, Nparticles).astype(np.float32),
        "puppi_py": pad_array(puppi_py, Nparticles).astype(np.float32),
        "puppi_pz": pad_array(puppi_pz, Nparticles).astype(np.float32),
        "puppi_E": pad_array(puppi_energy, Nparticles).astype(np.float32),
        # Directional/cylindrical coordinates remain useful for plots and
        # pairwise transformer inputs without replacing the stored Cartesian p4.
        "puppi_pt": pad_array(puppi_pt, Nparticles).astype(np.float32),
        "puppi_eta": pad_array(puppi_eta, Nparticles).astype(np.float32),
        "puppi_phi": pad_array(puppi_phi, Nparticles).astype(np.float32),
        "global": global_features,
        "diagnostics": diagnostics,
    }


def build_generator_target(arrays, name):
    """Build one event-level generator four-vector/kinematics target."""

    return np.stack(
        [np.asarray(arrays[f"gen_{name}_{field}"]) for field in GENERATOR_P4_FIELDS],
        axis=1,
    ).astype(np.float32)


def build_targets(arrays, selection, Nparticles):
    """Build generator, ancestry, and selected-algorithm targets."""

    truth_ancestry = select_and_order(arrays["particle_truthLabel"], selection)
    algorithm_label = select_and_order(
        arrays["particle_algorithmLabel"],
        selection,
    )
    algorithm_ak_index = select_and_order(
        arrays["particle_algorithmAKIndex"],
        selection,
    )
    algorithm_ca_index = select_and_order(
        arrays["particle_algorithmCAIndex"],
        selection,
    )

    has_chi0 = ak.any(
        (truth_ancestry >= 1) & (truth_ancestry <= 10),
        axis=2,
    )
    has_chi1 = ak.any(
        (truth_ancestry >= 11) & (truth_ancestry <= 20),
        axis=2,
    )
    mixed = has_chi0 & has_chi1

    # -2 is deliberately distinct from padding (-1). The ancestry bit mask is
    # also saved: bit 0 = chi0, bit 1 = chi1, so mixed ancestry is exactly 3.
    truth_label = ak.where(
        mixed,
        -2,
        ak.where(has_chi0, 1, ak.where(has_chi1, 2, 0)),
    )
    truth_ancestry_mask = (
        ak.values_astype(has_chi0, np.int64)
        + 2 * ak.values_astype(has_chi1, np.int64)
    )

    return {
        "suu": build_generator_target(arrays, "Suu"),
        "chi0": build_generator_target(arrays, "chi0"),
        "chi1": build_generator_target(arrays, "chi1"),
        "truthLabel": pad_array(
            truth_label,
            Nparticles,
            pad_value=-1,
        ).astype(np.int64),
        "truthAncestryMask": pad_array(
            truth_ancestry_mask,
            Nparticles,
            pad_value=-1,
        ).astype(np.int64),
        "truthHasChi0": pad_array(
            has_chi0,
            Nparticles,
            pad_value=False,
        ).astype(np.bool_),
        "truthHasChi1": pad_array(
            has_chi1,
            Nparticles,
            pad_value=False,
        ).astype(np.bool_),
        "truthMixed": pad_array(
            mixed,
            Nparticles,
            pad_value=False,
        ).astype(np.bool_),
        "algorithmLabel": pad_array(
            algorithm_label,
            Nparticles,
            pad_value=-1,
        ).astype(np.int64),
        "algorithmAKIndex": pad_array(
            algorithm_ak_index,
            Nparticles,
            pad_value=-1,
        ).astype(np.int64),
        "algorithmCAIndex": pad_array(
            algorithm_ca_index,
            Nparticles,
            pad_value=-1,
        ).astype(np.int64),
    }


def splitmix64(values):
    """Vectorized SplitMix64 finalizer for stable event-identity hashing."""

    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
        values = (
            (values ^ (values >> np.uint64(30)))
            * np.uint64(0xBF58476D1CE4E5B9)
        ) & mask
        values = (
            (values ^ (values >> np.uint64(27)))
            * np.uint64(0x94D049BB133111EB)
        ) & mask
        values = values ^ (values >> np.uint64(31))
    return values


def make_event_splits(arrays):
    """Assign stable 70/15/15 splits from (run, lumi, event)."""

    run = np.asarray(arrays["run"], dtype=np.uint64)
    lumi = np.asarray(arrays["lumi"], dtype=np.uint64)
    event = np.asarray(arrays["event"], dtype=np.uint64)

    with np.errstate(over="ignore"):
        identity_key = (
            event
            ^ (run * np.uint64(0xD6E8FEB86659FD93))
            ^ (lumi * np.uint64(0xA5A3564E27F8862B))
        )
    event_hash = splitmix64(identity_key)
    bucket = event_hash % np.uint64(10_000)

    split = np.full(len(event), 2, dtype=np.int8)
    split[bucket < 8_500] = 1
    split[bucket < 7_000] = 0

    # Retain a non-negative 63-bit hash for debugging on all PyTorch versions.
    split_hash = (event_hash & np.uint64(0x7FFFFFFFFFFFFFFF)).astype(np.int64)

    return {
        "run": run.astype(np.int64),
        "lumi": lumi.astype(np.int64),
        "event": event.astype(np.int64),
        "split": split,
        "split_hash": split_hash,
        "train_idx": np.flatnonzero(split == 0).astype(np.int64),
        "val_idx": np.flatnonzero(split == 1).astype(np.int64),
        "test_idx": np.flatnonzero(split == 2).astype(np.int64),
    }


def uniform_scalar(arrays, branch):
    """Return a Python scalar when an ntuple configuration branch is uniform."""

    values = np.asarray(arrays[branch])
    unique = np.unique(values)
    if len(unique) != 1:
        return None
    return unique[0].item()


def build_metadata(
    args, arrays, input_path, mode, shard_id, n_events, algorithm_mode
):
    """Build self-describing schema and provenance metadata for one shard."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "input": str(input_path),
            "tree": str(args.tree),
            "shard_id": int(shard_id),
            "n_events": int(n_events),
        },
        "selection": {
            "mode": mode,
            "Nparticles": int(args.Nparticles),
            "particle_order": "descending particle_puppi_pt",
            "puppi_weight_requirement": "particle_puppiWeight >= 0",
            "selected_jet_requirement": "ntuple particle_algorithmAKIndex >= 0",
            "teacher": algorithm_mode,
        },
        "features": {
            "particle_names": list(FEATURE_NAMES),
            "global_names": list(GLOBAL_FEATURE_NAMES),
            "particle_kinematics": "PUPPI only",
            "stored_p4_components": [
                "puppi_px",
                "puppi_py",
                "puppi_pz",
                "puppi_E",
            ],
        },
        "tensor_schema": {
            "particles": ["events", "Nparticles", len(FEATURE_NAMES)],
            "mask": ["events", "Nparticles"],
            "puppi_kinematics": ["events", "Nparticles"],
            "global": ["events", len(GLOBAL_FEATURE_NAMES)],
            "generator_targets": ["events", len(GENERATOR_P4_FIELDS)],
            "particle_targets": ["events", "Nparticles"],
            "event_identity_and_diagnostics": ["events"],
            "split_indices": ["events_in_split_within_this_shard"],
        },
        "truth": {
            "truthLabel_values": {
                "padding": -1,
                "mixed": -2,
                "neither": 0,
                "chi0": 1,
                "chi1": 2,
            },
            "truthAncestryMask_bits": {
                "bit_0": "chi0 ancestry",
                "bit_1": "chi1 ancestry",
                "mixed_value": 3,
                "padding": -1,
            },
        },
        "split": {
            "method": "SplitMix64(run, lumi, event) modulo 10000",
            "index_scope": "train_idx/val_idx/test_idx are local to this shard",
            "concatenation_guidance": "prefer the per-event split tensor",
            "fractions": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "codes": dict(SPLIT_NAMES),
        },
        "ntuplizer": {
            "isMC": uniform_scalar(arrays, "isMC"),
            "algorithmMode": algorithm_mode,
            "jetPtCut": uniform_scalar(arrays, "jetPtCut"),
            "akRadius": uniform_scalar(arrays, "akRadius"),
            "caRadius": uniform_scalar(arrays, "caRadius"),
            "cosThrust": uniform_scalar(arrays, "cosThrust"),
        },
    }


def required_branches():
    """Return branches required by this schema."""

    branches = {
        "run",
        "lumi",
        "event",
        "HT",
        "MET_pt",
        "MET_phi",
        "rho",
        "nPV",
        "PV_x",
        "PV_y",
        "PV_z",
        "isMC",
        "jetPtCut",
        "akRadius",
        "caRadius",
        "cosThrust",
        "particle_puppi_pt",
        "particle_puppi_eta",
        "particle_puppi_phi",
        "particle_puppi_energy",
        "particle_puppi_px",
        "particle_puppi_py",
        "particle_puppi_pz",
        "particle_charge",
        "particle_pdgId",
        "particle_dxy",
        "particle_dz",
        "particle_puppiWeight",
        "particle_fromPV",
        "particle_pvAssociationQuality",
        "particle_truthLabel",
        "particle_algorithmLabel",
        "particle_algorithmAKIndex",
        "particle_algorithmCAIndex",
    }
    for name in ("Suu", "chi0", "chi1"):
        branches.update(f"gen_{name}_{field}" for field in GENERATOR_P4_FIELDS)
    return branches


def algorithm_mode_from_input(input_path, tree_name):
    """Resolve the string-valued teacher from the ROOT file or its filename.

    Uproot may omit a ``std::string`` TBranch when it groups TTree iteration
    output into an Awkward record, so read that branch separately. Older files
    can still fall back to the mode encoded in their filename.
    """

    try:
        with uproot.open(input_path) as root_file:
            values = root_file[tree_name]["algorithmMode"].array(
                entry_stop=1,
                library="np",
            )
        if len(values):
            value = values[0]
            if isinstance(value, bytes):
                value = value.decode()
            value = str(value)
            if value in ("slimmed", "packed"):
                return value
    except (KeyError, TypeError, ValueError):
        pass

    stem_tokens = Path(input_path).stem.lower().replace("-", "_").split("_")
    matches = [name for name in ("slimmed", "packed") if name in stem_tokens]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Could not read algorithmMode from {input_path!s} or infer it from "
        "the filename; expected 'slimmed' or 'packed'"
    )


def discover_inputs(args):
    """Resolve one input path or every ROOT file in ``--root-dir``."""

    if not args.all_input and args.input != "all":
        return [Path(args.input)]

    inputs = sorted(Path(args.root_dir).glob("*.root"))
    if not inputs:
        raise FileNotFoundError(f"No ROOT files found in {args.root_dir}")
    return inputs


def output_prefix(output, input_path, multiple_inputs):
    """Give each input its own output prefix when processing many files."""

    output = Path(output)
    if not multiple_inputs:
        return output
    output_dir = output.parent if output.suffix else output
    return output_dir / f"{Path(input_path).stem}.pt"


def validate_schema(arrays):
    """Fail early with a useful error if the ntuple schema is incompatible."""

    missing = sorted(required_branches() - set(arrays.fields))
    if missing:
        joined = "\n  - ".join(missing)
        raise KeyError(f"Input ntuple is missing required branches:\n  - {joined}")


def tensor(array):
    """Convert a NumPy array to a PyTorch tensor without changing its dtype."""

    return torch.from_numpy(np.ascontiguousarray(array))


def output_filename(output, mode, shard_id):
    """Build a shard filename without relying on string replacement."""

    output = Path(output)
    suffix = output.suffix if output.suffix else ".pt"
    stem = output.stem if output.suffix else output.name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.parent / f"{stem}_{mode}_shard{shard_id:04d}{suffix}"


def make_dataset(processed, targets, identity, metadata):
    """Convert one processed shard to the dictionary written by torch.save."""

    dataset = {
        "particles": tensor(processed["particles"]),
        "mask": tensor(processed["mask"]),
        "puppi_px": tensor(processed["puppi_px"]),
        "puppi_py": tensor(processed["puppi_py"]),
        "puppi_pz": tensor(processed["puppi_pz"]),
        "puppi_E": tensor(processed["puppi_E"]),
        "puppi_pt": tensor(processed["puppi_pt"]),
        "puppi_eta": tensor(processed["puppi_eta"]),
        "puppi_phi": tensor(processed["puppi_phi"]),
        "global": tensor(processed["global"]),
    }

    dataset.update({name: tensor(value) for name, value in targets.items()})
    dataset.update({name: tensor(value) for name, value in identity.items()})
    dataset.update(
        {
            name: tensor(value)
            for name, value in processed["diagnostics"].items()
        }
    )
    dataset["metadata"] = metadata
    return dataset


def preprocess_file(args, input_path, output, modes):
    """Preprocess every shard and requested representation of one ROOT file."""

    algorithm_mode = algorithm_mode_from_input(input_path, args.tree)
    shard_ids = {mode: 0 for mode in modes}
    validated_schema = False

    print(f"[INFO] Preprocessing {input_path} ({algorithm_mode})")

    # Deliberately keep ROOT iteration outside the mode loop: every ROOT basket
    # is read once, then all requested representations are produced from the
    # same in-memory Awkward record array.
    for arrays in uproot.iterate(
        f"{input_path}:{args.tree}",
        expressions=sorted(required_branches()),
        step_size=args.step_size,
        library="ak",
    ):
        if not validated_schema:
            validate_schema(arrays)
            validated_schema = True

        identity = make_event_splits(arrays)

        for mode in modes:
            shard_id = shard_ids[mode]
            selection = prepare_particle_selection(arrays, mode)
            processed = process_events(arrays, selection, args.Nparticles)
            targets = build_targets(arrays, selection, args.Nparticles)
            n_events = processed["particles"].shape[0]
            metadata = build_metadata(
                args,
                arrays,
                input_path,
                mode,
                shard_id,
                n_events,
                algorithm_mode,
            )
            dataset = make_dataset(processed, targets, identity, metadata)

            out_file = output_filename(output, mode, shard_id)
            torch.save(dataset, out_file)
            print(f"[INFO] Wrote {out_file} with {n_events} events")

            shard_ids[mode] += 1

    if not validated_schema:
        raise RuntimeError(f"No events were read from input ROOT tree {input_path}")


def main(args):
    if args.Nparticles <= 0:
        raise ValueError("--Nparticles must be positive")

    inputs = discover_inputs(args)
    modes = (
        MODE_NAMES
        if args.all_data_modes or args.data_mode == "all"
        else [args.data_mode]
    )
    multiple_inputs = args.all_input or args.input == "all" or len(inputs) > 1
    for input_path in inputs:
        output = output_prefix(args.output, input_path, multiple_inputs)
        preprocess_file(args, input_path, output, modes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="/eos/uscms/store/user/aji/rootFiles_particleTransformer/WbWb_4000_1000_slimmed.root",
        help="Input ROOT file, or 'all' to scan --root-dir",
    )
    parser.add_argument(
        "--root-dir",
        default="/eos/uscms/store/user/aji/rootfiles_particleTransformer",
        help="Directory scanned when --all-input or --input all is used",
    )
    parser.add_argument(
        "--all-input",
        action="store_true",
        help="Preprocess every .root file found directly in --root-dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ptfiles/WbWb_4000_1000_slimmed.pt",
        help=(
            "Output .pt filename prefix; with --all-input or --input all, "
            "this is treated as an output directory (or its parent is used "
            "if it has a suffix)"
        ),
    )
    parser.add_argument(
        "--tree",
        type=str,
        default="particleTransformerNtuplizer/Events",
        help="ROOT TTree name",
    )
    parser.add_argument(
        "--data-mode",
        choices=MODE_NAMES + ["all"],
        default="all_ak_constituents",
        help="Particle representation to store, or 'all' for every representation",
    )
    parser.add_argument(
        "--all-data-modes",
        action="store_true",
        help="Write both all_pf and all_ak_constituents outputs",
    )
    parser.add_argument(
        "--Nparticles",
        type=int,
        default=256,
        help="Maximum number of particles retained per event",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5000,
        help="Number of ROOT events processed per shard",
    )

    main(parser.parse_args())
