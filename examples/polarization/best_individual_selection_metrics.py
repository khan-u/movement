#!/usr/bin/env python3
"""Test metrics for selecting the most trustworthy reference individual.

Given multiple individuals from a dataset, which metric best predicts
whose AP ordering is most anatomically correct?

Metrics tested:
1. R×M (current approach) - velocity confidence
2. PC1 Variance Ratio - how well-defined the body axis is geometrically
3. Mean Inverse Lateral Variance - skeleton stability across frames
4. Cross-Individual Agreement - consensus with other individuals
5. Skeleton Completeness - fraction of valid (non-NaN) keypoints

For each metric, we select the "best" individual and check GT accuracy.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from movement.io import load_poses as load_poses_module
from movement.kinematics.body_axis import (
    ValidateAPConfig,
    validate_ap,
)


class TeeOutput:
    """Context manager that duplicates stdout to both console and a file."""

    def __init__(self, filepath: Path):
        """Initialize with output file path."""
        self.filepath = Path(filepath)
        self.file = None
        self.original_stdout = None

    def __enter__(self):
        """Open file and redirect stdout."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.filepath, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout and close file."""
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
        return False

    def write(self, text):
        """Write to both stdout and file."""
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        """Flush both stdout and file."""
        self.original_stdout.flush()
        self.file.flush()


# =============================================================================
# Configuration (same as grid_search)
# =============================================================================

ROOT_PATH = Path(__file__).parent / "datasets" / "multi-animal"
SLP_DIR = ROOT_PATH / "slp"
OUTPUT_DIR = ROOT_PATH / "exports" / "reference_selection_test"
LOGS_DIR = OUTPUT_DIR / "logs"

GROUND_TRUTH = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": {
        0: 3,
        1: 2,
        2: 1,
    },
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": {
        0: 2,
        3: 1,
    },
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": {
        0: 6,
        5: 5,
        6: 4,
        7: 3,
        8: 2,
        9: 1,
    },
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": {
        0: 3,
        1: 2,
        6: 1,
    },
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": {
        1: 3,
        0: 2,
        2: 1,
    },
}

FILE_LABELS = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": "2Flies",
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": "2Mice",
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": "4Gerbils",
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": "5Mice",
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": "2Bees",
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class IndividualMetrics:
    """Metrics for a single individual."""

    individual: str
    # Core metrics
    rxm: float  # R × M (current approach)
    pc1_variance_ratio: float  # s[0] / s[1] from SVD
    mean_inv_lateral_var: float  # 1 / mean(lateral_std)
    skeleton_completeness: float  # fraction of valid keypoints
    # For cross-individual agreement (computed later)
    agreement_score: float = 0.0
    # GT accuracy
    gt_accuracy: float = 0.0  # pairwise accuracy against GT
    gt_ordering: list = None  # node ordering by PC1 projection


@dataclass
class DatasetResult:
    """Results for a single dataset."""

    file_stem: str
    label: str
    individuals: list[IndividualMetrics]
    gt_nodes: list[int]
    gt_ranks: dict[int, int]


# =============================================================================
# Metric Computation
# =============================================================================


def compute_gt_accuracy(
    avg_skeleton: np.ndarray,
    pc1: np.ndarray,
    anterior_sign: int,
    gt_ranks: dict[int, int],
) -> tuple[float, list[int]]:
    """Compute pairwise GT accuracy for an individual.

    Returns (accuracy_pct, ordering) where ordering is GT nodes sorted by
    their projection onto the inferred AP axis.
    """
    if avg_skeleton is None or pc1 is None or anterior_sign == 0:
        return 0.0, []

    avg_skeleton = np.array(avg_skeleton)
    pc1 = np.array(pc1)

    # Inferred AP axis
    pc1_norm = pc1 / np.linalg.norm(pc1)
    e_ap = anterior_sign * pc1_norm

    # Project GT nodes onto AP axis
    gt_ap_coords = {}
    for node_idx in gt_ranks:
        if node_idx < len(avg_skeleton):
            pos = avg_skeleton[node_idx]
            if not np.any(np.isnan(pos)):
                gt_ap_coords[node_idx] = np.dot(pos, e_ap)

    if len(gt_ap_coords) < 2:
        return 0.0, []

    # Pairwise accuracy
    correct = 0
    total = 0
    for a, b in itertools.combinations(gt_ap_coords.keys(), 2):
        total += 1
        ap_a, ap_b = gt_ap_coords[a], gt_ap_coords[b]
        gt_a, gt_b = gt_ranks[a], gt_ranks[b]

        if (ap_a > ap_b and gt_a > gt_b) or (ap_a < ap_b and gt_a < gt_b):
            correct += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Ordering (sorted by AP coordinate, posterior to anterior)
    ordering = sorted(gt_ap_coords.keys(), key=lambda x: gt_ap_coords[x])

    return accuracy, ordering


def compute_pc1_variance_ratio(avg_skeleton: np.ndarray) -> float:
    """Compute PC1/PC2 variance ratio from SVD.

    Higher ratio = clearer elongated body axis.
    """
    if avg_skeleton is None:
        return 0.0

    avg_skeleton = np.array(avg_skeleton)
    valid_mask = ~np.any(np.isnan(avg_skeleton), axis=1)

    if np.sum(valid_mask) < 2:
        return 0.0

    valid_rows = avg_skeleton[valid_mask]

    try:
        _u, s, _vt = np.linalg.svd(valid_rows, full_matrices=False)
        if len(s) >= 2 and s[1] > 0:
            return float(s[0] / s[1])
        elif len(s) >= 1:
            return float(s[0])  # Only one component
        else:
            return 0.0
    except Exception:
        return 0.0


def compute_mean_inv_lateral_var(lateral_std: np.ndarray) -> float:
    """Compute mean inverse lateral variance.

    Higher = more stable skeleton (lower variance).
    """
    if lateral_std is None:
        return 0.0

    lateral_std = np.array(lateral_std)
    valid = lateral_std[~np.isnan(lateral_std)]

    if len(valid) == 0:
        return 0.0

    # Avoid division by zero
    valid = np.maximum(valid, 1e-6)

    return float(np.mean(1.0 / valid))


def compute_skeleton_completeness(avg_skeleton: np.ndarray) -> float:
    """Compute fraction of valid (non-NaN) keypoints."""
    if avg_skeleton is None:
        return 0.0

    avg_skeleton = np.array(avg_skeleton)
    valid_mask = ~np.any(np.isnan(avg_skeleton), axis=1)

    return float(np.mean(valid_mask))


def compute_cross_individual_agreement(
    individuals: list[IndividualMetrics],
) -> None:
    """Compute agreement scores based on ordering consistency.

    For each individual, count how many others share the same GT node ordering.
    Modifies individuals in place.
    """
    n = len(individuals)
    if n < 2:
        for ind in individuals:
            ind.agreement_score = 1.0
        return

    # Compare orderings pairwise
    for i, ind_i in enumerate(individuals):
        if ind_i.gt_ordering is None or len(ind_i.gt_ordering) == 0:
            ind_i.agreement_score = 0.0
            continue

        matches = 0
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
            if ind_j.gt_ordering is None:
                continue

            # Check if orderings match
            if ind_i.gt_ordering == ind_j.gt_ordering:
                matches += 1

        # Agreement score = fraction of other individuals that agree
        ind_i.agreement_score = matches / (n - 1)


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_dataset(
    slp_path: Path,
    gt_ranks: dict[int, int],
    config: ValidateAPConfig,
) -> list[IndividualMetrics]:
    """Analyze all individuals in a dataset."""
    ds = load_poses_module.from_sleap_file(slp_path)
    keypoint_names = list(ds.coords["keypoints"].values)

    # Get individuals
    if "individuals" in ds.position.dims:
        individuals = list(ds.coords["individuals"].values)
    else:
        individuals = ["single"]

    # Use first two GT nodes as arbitrary pair for validate_ap
    gt_indices = list(gt_ranks.keys())
    if len(gt_indices) < 2:
        return []

    from_idx, to_idx = gt_indices[0], gt_indices[1]
    from_kp = keypoint_names[from_idx]
    to_kp = keypoint_names[to_idx]

    results = []

    for individual in individuals:
        try:
            if "individuals" in ds.position.dims:
                pos_data = ds.position.sel(individuals=individual)
            else:
                pos_data = ds.position

            # Run validate_ap to get all the data we need
            val = validate_ap(
                pos_data,
                from_node=from_kp,
                to_node=to_kp,
                config=config,
                verbose=False,
            )

            # Extract metrics
            r = val.get("resultant_length", 0.0)
            m = val.get("vote_margin", 0.0)
            rxm = r * m if not (np.isnan(r) or np.isnan(m)) else 0.0

            avg_skeleton = val.get("avg_skeleton")
            pc1 = val.get("PC1")
            anterior_sign = val.get("anterior_sign", 0)
            lateral_std = val.get("lateral_std")

            # Compute metrics
            pc1_var_ratio = compute_pc1_variance_ratio(avg_skeleton)
            inv_lat_var = compute_mean_inv_lateral_var(lateral_std)
            completeness = compute_skeleton_completeness(avg_skeleton)
            gt_acc, gt_ordering = compute_gt_accuracy(
                avg_skeleton, pc1, anterior_sign, gt_ranks
            )

            results.append(
                IndividualMetrics(
                    individual=str(individual),
                    rxm=rxm,
                    pc1_variance_ratio=pc1_var_ratio,
                    mean_inv_lateral_var=inv_lat_var,
                    skeleton_completeness=completeness,
                    gt_accuracy=gt_acc,
                    gt_ordering=gt_ordering,
                )
            )

        except Exception as e:
            print(f"  Warning: Failed for {individual}: {e}")
            continue

    # Compute cross-individual agreement
    compute_cross_individual_agreement(results)

    return results


def select_best_by_metric(
    individuals: list[IndividualMetrics],
    metric: str,
) -> IndividualMetrics | None:
    """Select the best individual by a given metric."""
    if not individuals:
        return None

    if metric == "rxm":
        return max(individuals, key=lambda x: x.rxm)
    elif metric == "pc1_variance_ratio":
        return max(individuals, key=lambda x: x.pc1_variance_ratio)
    elif metric == "mean_inv_lateral_var":
        return max(individuals, key=lambda x: x.mean_inv_lateral_var)
    elif metric == "agreement_score":
        return max(individuals, key=lambda x: x.agreement_score)
    elif metric == "skeleton_completeness":
        return max(individuals, key=lambda x: x.skeleton_completeness)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_analysis() -> dict[str, dict]:
    """Run the full analysis across all datasets."""
    config = ValidateAPConfig()

    metrics = [
        "rxm",
        "pc1_variance_ratio",
        "mean_inv_lateral_var",
        "agreement_score",
        "skeleton_completeness",
    ]

    # Results structure: {metric: {dataset: selected_individual_metrics}}
    results_by_metric = {m: {} for m in metrics}
    all_dataset_results = {}

    print("Analyzing datasets...")
    print()

    for slp_file in sorted(SLP_DIR.glob("*.slp")):
        file_stem = slp_file.stem
        if file_stem not in GROUND_TRUTH:
            continue

        label = FILE_LABELS.get(file_stem, file_stem[:15])
        gt_ranks = GROUND_TRUTH[file_stem]

        print(f"  {label}...")
        individuals = analyze_dataset(slp_file, gt_ranks, config)

        if not individuals:
            print("    No valid individuals found")
            continue

        all_dataset_results[file_stem] = DatasetResult(
            file_stem=file_stem,
            label=label,
            individuals=individuals,
            gt_nodes=list(gt_ranks.keys()),
            gt_ranks=gt_ranks,
        )

        # Select best by each metric
        for metric in metrics:
            best = select_best_by_metric(individuals, metric)
            if best:
                results_by_metric[metric][file_stem] = best

    return results_by_metric, all_dataset_results


def print_results(  # noqa: C901
    results_by_metric: dict[str, dict],
    all_dataset_results: dict[str, DatasetResult],
):
    """Print analysis results."""
    metrics = list(results_by_metric.keys())
    datasets = list(all_dataset_results.keys())

    print()
    print("=" * 80)
    print("REFERENCE SELECTION METRIC COMPARISON")
    print("=" * 80)
    print()
    print("For each metric, we select the 'best' individual and check GT acc.")
    print("Goal: Find metric that reliably selects individuals with 100% GT.")
    print()

    # Summary table
    print("-" * 79)
    print(f"{'Metric':<25} | {'100%':<6} | {'Mean':<6} | {'Datasets'}")
    print("-" * 79)

    for metric in metrics:
        accuracies = []
        perfect_count = 0

        for file_stem in datasets:
            if file_stem in results_by_metric[metric]:
                acc = results_by_metric[metric][file_stem].gt_accuracy
                accuracies.append(acc)
                if acc == 100.0:
                    perfect_count += 1

        mean_acc = np.mean(accuracies) if accuracies else 0.0
        n_datasets = len(accuracies)

        print(
            f"{metric:<25} | {perfect_count}/{n_datasets:<4} | "
            f"{mean_acc:>5.1f}% | ",
            end="",
        )

        # Per-dataset results
        for file_stem in datasets:
            label = FILE_LABELS.get(file_stem, file_stem[:8])[:8]
            if file_stem in results_by_metric[metric]:
                acc = results_by_metric[metric][file_stem].gt_accuracy
                symbol = "✓" if acc == 100.0 else f"{acc:.0f}%"
                print(f"{label}:{symbol} ", end="")
        print()

    print("-" * 80)
    print()

    # Detailed per-dataset breakdown
    print("=" * 80)
    print("DETAILED PER-DATASET BREAKDOWN")
    print("=" * 80)

    for file_stem, ds_result in all_dataset_results.items():
        label = ds_result.label
        print(f"\n{label} ({len(ds_result.individuals)} individuals):")
        print("-" * 60)

        # Header
        print(
            f"  {'Individual':<15} | {'R×M':<6} | {'PC1':<5} | "
            f"{'InvLat':<6} | {'Agree':<5} | {'Compl':<5} | {'GT'}"
        )
        print("  " + "-" * 65)

        # Sort by GT accuracy to see patterns
        sorted_inds = sorted(
            ds_result.individuals, key=lambda x: x.gt_accuracy, reverse=True
        )

        for ind in sorted_inds:
            pc1_vr = ind.pc1_variance_ratio
            inv_lat = ind.mean_inv_lateral_var
            agree = ind.agreement_score
            compl = ind.skeleton_completeness
            print(
                f"  {ind.individual:<15} | {ind.rxm:<6.3f} | "
                f"{pc1_vr:<5.2f} | {inv_lat:<6.2f} | "
                f"{agree:<5.2f} | {compl:<5.2f} | "
                f"{ind.gt_accuracy:>5.1f}%"
            )

        # Show which metric would select which individual
        print()
        print("  Selection by metric:")
        for metric in metrics:
            if file_stem in results_by_metric[metric]:
                selected = results_by_metric[metric][file_stem]
                if selected.gt_accuracy == 100.0:
                    acc_str = "✓"
                else:
                    acc_str = f"{selected.gt_accuracy:.0f}%"
                print(
                    f"    {metric:<25}: {selected.individual:<15} "
                    f"(GT: {acc_str})"
                )

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Find best metric
    best_metric = None
    best_perfect = -1
    best_mean = -1

    for metric in metrics:
        accuracies = [
            results_by_metric[metric][fs].gt_accuracy
            for fs in datasets
            if fs in results_by_metric[metric]
        ]
        perfect = sum(1 for a in accuracies if a == 100.0)
        mean_acc = np.mean(accuracies) if accuracies else 0.0

        is_better = perfect > best_perfect
        same_perfect = perfect == best_perfect
        is_equal_but_higher_mean = same_perfect and mean_acc > best_mean
        if is_better or is_equal_but_higher_mean:
            best_metric = metric
            best_perfect = perfect
            best_mean = mean_acc

    print(f"\nBest metric: {best_metric}")
    n_ds = len(datasets)
    print(f"  - Achieves 100% GT accuracy on {best_perfect}/{n_ds} datasets")
    print(f"  - Mean GT accuracy: {best_mean:.1f}%")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"reference_selection_{timestamp}.log"

    with TeeOutput(log_path):
        print("Reference Selection Metric Analysis")
        print(f"Started at {datetime.now().isoformat()}")
        print(f"Log file: {log_path}")
        print()

        results_by_metric, all_dataset_results = run_analysis()
        print_results(results_by_metric, all_dataset_results)

        print(f"\nLog saved to: {log_path}")
