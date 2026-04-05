#!/usr/bin/env python3
"""Exhaustive grid search for AP inference threshold optimization.

This script tests all combinations of:
- AP midpoint computation methods (6 variants)
- Lateral threshold methods (5 variants)
- Edge threshold methods (4 variants)
- d_norm normalization methods (3 variants)
- Effective score formulas (4 variants)
- Pair selection scoring methods (4 variants)

Goal: Maximize "Both nodes in GT" across all datasets.

Optimizations:
- Pre-loads all datasets once into memory
- Pre-computes PCA and frame selection (invariant to threshold params)
- Only re-runs filter cascade for each config (the fast part)
- Uses multiprocessing with shared pre-computed data
- Progress tracking with tqdm
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from movement.io import load_poses as load_poses_module
from movement.kinematics.body_axis import (
    FrameSelection,
    ValidateAPConfig,
    build_centered_skeletons,
    run_clustering_and_pca,
    run_motion_segmentation,
    select_tier2_frames,
    validate_ap,
)

# Utilities


class TeeOutput:
    """Context manager that duplicates stdout to both console and a file."""

    def __init__(self, filepath: Path):
        """Initialize with the target file path."""
        self.filepath = Path(filepath)
        self.file = None
        self.original_stdout = None

    def __enter__(self):
        """Open the file and redirect stdout."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.filepath, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout and close the file."""
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
        return False

    def write(self, text):
        """Write text to both stdout and file."""
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        """Flush both stdout and file."""
        self.original_stdout.flush()
        self.file.flush()


# Dataset Configuration

ROOT_PATH = Path(__file__).parent / "datasets" / "multi-animal"
SLP_DIR = ROOT_PATH / "slp"
OUTPUT_DIR = ROOT_PATH / "exports" / "grid_search"
LOGS_DIR = OUTPUT_DIR / "logs"

# Hand-curated ground truth AP rankings
# Format: {file_stem: {node_index: rank}}
# Convention: higher rank = more anterior (rank 1 = most posterior in subset)
GROUND_TRUTH = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": {
        0: 3,  # head (most anterior)
        1: 2,  # thorax
        2: 1,  # abdomen (most posterior)
    },
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": {
        0: 2,  # snout (anterior)
        3: 1,  # tail-base (posterior)
    },
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": {
        0: 6,  # nose (most anterior)
        5: 5,  # spine1
        6: 4,  # spine2
        7: 3,  # spine3
        8: 2,  # spine4
        9: 1,  # spine5 (most posterior)
    },
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": {
        0: 3,  # nose (most anterior)
        1: 2,  # neck
        6: 1,  # tail_base (most posterior)
    },
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": {
        1: 3,  # head (most anterior)
        0: 2,  # thorax
        2: 1,  # abdomen (most posterior)
    },
}

# Display labels for files
FILE_LABELS = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": "2Flies",
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": "2Mice",
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": "4Gerbils",
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": "5Mice",
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": "2Bees",
}


def compute_best_individual_with_pca(
    slp_path: Path, gt_nodes: list[int]
) -> tuple[str, float, dict | None]:
    """Compute best individual for a dataset using R×M selection.

    Runs validate_ap on each individual with an arbitrary GT pair and
    selects the individual with highest R×M = resultant_length × vote_margin.

    Returns (best_individual_name, best_rxm_value, best_validation_result).
    The validation result has PCA data for precomputation.
    """
    ds = load_poses_module.from_sleap_file(slp_path)
    keypoint_names = list(ds.coords["keypoints"].values)

    # Get individuals
    if "individuals" in ds.position.dims:
        individuals = list(ds.coords["individuals"].values)
    else:
        individuals = ["single"]

    # Use first two GT nodes as arbitrary pair
    if len(gt_nodes) < 2:
        return individuals[0] if individuals else "single", 0.0, None

    from_idx, to_idx = gt_nodes[0], gt_nodes[1]
    from_kp = keypoint_names[from_idx]
    to_kp = keypoint_names[to_idx]

    config = ValidateAPConfig()
    best_individual = individuals[0]
    best_rxm = -1.0
    best_val_result = None

    for individual in individuals:
        try:
            if "individuals" in ds.position.dims:
                pos_data = ds.position.sel(individuals=individual)
            else:
                pos_data = ds.position

            val = validate_ap(
                pos_data,
                from_node=from_kp,
                to_node=to_kp,
                config=config,
                verbose=False,
            )

            r = val.get("resultant_length", 0.0)
            m = val.get("vote_margin", 0.0)
            rxm = r * m if not (np.isnan(r) or np.isnan(m)) else 0.0

            if rxm > best_rxm:
                best_rxm = rxm
                best_individual = individual
                best_val_result = val

        except Exception:
            continue

    return str(best_individual), best_rxm, best_val_result


def discover_datasets() -> dict[str, dict]:
    """Discover datasets from SLP directory."""
    datasets = {}
    for slp_file in SLP_DIR.glob("*.slp"):
        file_stem = slp_file.stem
        if file_stem in GROUND_TRUTH:
            gt_ranks = GROUND_TRUTH[file_stem]
            # GT nodes sorted by rank (posterior to anterior)
            gt_nodes = sorted(gt_ranks.keys(), key=lambda x: gt_ranks[x])
            datasets[file_stem] = {
                "path": slp_file,
                "gt_nodes": gt_nodes,
                "gt_ranks": gt_ranks,
                "label": FILE_LABELS.get(file_stem, file_stem[:20]),
                "individual": None,
                "val_result": None,  # Cached validation result with PCA data
            }
    return datasets


def compute_all_best_individuals(datasets: dict[str, dict]) -> dict[str, dict]:
    """Compute best individual for each dataset using R×M selection.

    Also caches the validation result (containing PCA data) for later reuse.
    """
    print("Computing best individuals via R×M selection...")
    for _file_stem, ds_info in tqdm(datasets.items(), desc="R×M selection"):
        best_ind, rxm, val_result = compute_best_individual_with_pca(
            ds_info["path"], ds_info["gt_nodes"]
        )
        ds_info["individual"] = best_ind
        ds_info["best_rxm"] = rxm
        ds_info["val_result"] = val_result  # Cache for PCA reuse
        label = ds_info["label"]
        print(f"  {label}: {best_ind} (R×M={rxm:.3f})")
    return datasets


# Discover datasets at module load time (individuals computed later)
DATASETS = discover_datasets()


# Method Enums


class MidpointMethod:
    """AP midpoint computation methods."""

    GEOMETRIC_CENTER = "geometric_center"  # (pc1_min + pc1_max) / 2
    CENTROID = "centroid"  # mean of all PC1 projections
    MEDIAN = "median"  # median of PC1 projections
    WEIGHTED_LATERAL = "weighted_lateral"  # weighted by inverse lateral offset
    WEIGHTED_VARIANCE = "weighted_variance"  # weighted by inverse variance
    TRIMMED_MEAN = "trimmed_mean"  # exclude top/bottom 10%


class LateralThreshMethod:
    """Lateral threshold computation methods."""

    FIXED = "fixed"  # fixed threshold value
    PERCENTILE = "percentile"  # pass nodes below Nth percentile
    ADAPTIVE_MIN_K = "adaptive_min_k"  # relax until K nodes pass
    ZSCORE = "zscore"  # pass nodes within N std devs
    IQR = "iqr"  # pass nodes within Q1 - 1.5*IQR


class EdgeThreshMethod:
    """Edge threshold computation methods."""

    FIXED = "fixed"  # fixed threshold value
    PERCENTILE = "percentile"  # Nth percentile of midline_dist
    ADAPTIVE_MIN_K = "adaptive_min_k"  # relax until K distal pairs
    BODY_LENGTH_NORM = "body_length_norm"  # normalized by body length


class NormalizationMethod:
    """d_norm normalization methods."""

    BODY_WIDTH = "body_width"  # d / d_max (current)
    MIN_MAX = "min_max"  # (d - d_min) / (d_max - d_min)
    PERCENTILE_RANK = "percentile_rank"  # rank / n_nodes


class EffectiveScoreFormula:
    """Effective score computation formulas."""

    ADDITIVE = "additive"  # d + w1*lat + w2*long
    MULTIPLICATIVE = "multiplicative"  # d * (1 + w1*lat) * (1 + w2*long)
    MAX_BASED = "max_based"  # max(d, w1*lat, w2*long)
    RMS = "rms"  # sqrt(d^2 + (w1*lat)^2 + (w2*long)^2)


class PairScoringMethod:
    """Pair selection scoring methods."""

    MAX_SEPARATION = "max_separation"  # original
    WEIGHTED_LATERAL = "weighted_lateral"  # weight by inverse lateral offset
    WEIGHTED_VARIANCE = "weighted_variance"  # weight by inverse variance
    WEIGHTED_BOTH = "weighted_both"  # weight by both


# Method Implementations


def compute_midpoint(
    pc1_coords: np.ndarray,
    lateral_offsets: np.ndarray,
    lateral_std: np.ndarray,
    valid_mask: np.ndarray,
    method: str,
) -> float:
    """Compute AP midpoint using specified method."""
    valid_pc1 = pc1_coords[valid_mask]
    valid_lateral = lateral_offsets[valid_mask]
    valid_std = lateral_std[valid_mask]

    if len(valid_pc1) == 0:
        return np.nan

    if method == MidpointMethod.GEOMETRIC_CENTER:
        return (np.min(valid_pc1) + np.max(valid_pc1)) / 2

    elif method == MidpointMethod.CENTROID:
        return np.mean(valid_pc1)

    elif method == MidpointMethod.MEDIAN:
        return np.median(valid_pc1)

    elif method == MidpointMethod.WEIGHTED_LATERAL:
        # Weight by inverse lateral offset (favor centerline nodes)
        weights = 1.0 / (valid_lateral + 1e-6)
        weights = weights / np.sum(weights)
        return np.sum(valid_pc1 * weights)

    elif method == MidpointMethod.WEIGHTED_VARIANCE:
        # Weight by inverse variance (favor stable nodes)
        weights = 1.0 / (valid_std + 1e-6)
        weights = weights / np.sum(weights)
        return np.sum(valid_pc1 * weights)

    elif method == MidpointMethod.TRIMMED_MEAN:
        # Exclude top/bottom 10%
        n = len(valid_pc1)
        trim = max(1, int(0.1 * n))
        sorted_pc1 = np.sort(valid_pc1)
        if n > 2 * trim:
            return np.mean(sorted_pc1[trim:-trim])
        return np.mean(valid_pc1)

    else:
        raise ValueError(f"Unknown midpoint method: {method}")


def normalize_d(
    d_raw: np.ndarray,
    valid_mask: np.ndarray,
    method: str,
) -> np.ndarray:
    """Normalize lateral offsets using specified method."""
    d_valid = d_raw[valid_mask]
    d_norm = np.full_like(d_raw, np.nan)

    if len(d_valid) == 0:
        return d_norm

    if method == NormalizationMethod.BODY_WIDTH:
        d_max = np.nanmax(d_valid)
        if d_max > 0:
            d_norm[valid_mask] = d_valid / d_max
        else:
            d_norm[valid_mask] = 0.0

    elif method == NormalizationMethod.MIN_MAX:
        d_min = np.nanmin(d_valid)
        d_max = np.nanmax(d_valid)
        if d_max > d_min:
            d_norm[valid_mask] = (d_valid - d_min) / (d_max - d_min)
        else:
            d_norm[valid_mask] = 0.0

    elif method == NormalizationMethod.PERCENTILE_RANK:
        # Rank-based normalization
        ranks = np.argsort(np.argsort(d_valid))
        if len(d_valid) > 1:
            d_norm[valid_mask] = ranks / (len(d_valid) - 1)
        else:
            d_norm[valid_mask] = 0.0

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return d_norm


def compute_effective_score(
    d_norm: np.ndarray,
    lat_std_norm: np.ndarray,
    long_std_norm: np.ndarray,
    lat_weight: float,
    long_weight: float,
    formula: str,
) -> np.ndarray:
    """Compute effective score using specified formula."""
    # Handle NaN values
    d = np.nan_to_num(d_norm, nan=0.0)
    lat = np.nan_to_num(lat_std_norm, nan=0.0)
    long = np.nan_to_num(long_std_norm, nan=0.0)

    if formula == EffectiveScoreFormula.ADDITIVE:
        return d + lat_weight * lat + long_weight * long

    elif formula == EffectiveScoreFormula.MULTIPLICATIVE:
        return d * (1 + lat_weight * lat) * (1 + long_weight * long)

    elif formula == EffectiveScoreFormula.MAX_BASED:
        return np.maximum(np.maximum(d, lat_weight * lat), long_weight * long)

    elif formula == EffectiveScoreFormula.RMS:
        return np.sqrt(
            d**2 + (lat_weight * lat) ** 2 + (long_weight * long) ** 2
        )

    else:
        raise ValueError(f"Unknown effective score formula: {formula}")


def compute_effective_threshold(
    base_thresh: float,
    lat_weight: float,
    long_weight: float,
    formula: str,
) -> float:
    """Compute effective threshold consistent with the score formula."""
    if formula == EffectiveScoreFormula.ADDITIVE:
        return base_thresh * (1 + lat_weight + long_weight)

    elif formula == EffectiveScoreFormula.MULTIPLICATIVE:
        return base_thresh * (1 + lat_weight) * (1 + long_weight)

    elif formula == EffectiveScoreFormula.MAX_BASED:
        return base_thresh

    elif formula == EffectiveScoreFormula.RMS:
        return base_thresh * np.sqrt(1 + lat_weight**2 + long_weight**2)

    else:
        raise ValueError(f"Unknown formula: {formula}")


def apply_lateral_filter_custom(
    d_norm: np.ndarray,
    lat_std_norm: np.ndarray,
    long_std_norm: np.ndarray,
    valid_mask: np.ndarray,
    lat_weight: float,
    long_weight: float,
    thresh_method: str,
    thresh_param: float,
    formula: str,
) -> np.ndarray:
    """Apply lateral filter using specified method. Returns candidates."""
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return np.array([], dtype=int)

    # Compute effective scores
    scores = compute_effective_score(
        d_norm, lat_std_norm, long_std_norm, lat_weight, long_weight, formula
    )
    valid_scores = scores[valid_mask]

    if thresh_method == LateralThreshMethod.FIXED:
        eff_thresh = compute_effective_threshold(
            thresh_param, lat_weight, long_weight, formula
        )
        passing = valid_scores <= eff_thresh

    elif thresh_method == LateralThreshMethod.PERCENTILE:
        # thresh_param is the percentile (0-100)
        percentile_val = np.percentile(valid_scores, thresh_param)
        passing = valid_scores <= percentile_val

    elif thresh_method == LateralThreshMethod.ADAPTIVE_MIN_K:
        # thresh_param is min number of nodes to pass
        min_k = int(thresh_param)
        sorted_indices = np.argsort(valid_scores)
        passing = np.zeros(len(valid_scores), dtype=bool)
        passing[sorted_indices[:min_k]] = True

    elif thresh_method == LateralThreshMethod.ZSCORE:
        # thresh_param is number of std devs
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        if std_score > 0:
            passing = valid_scores <= mean_score + thresh_param * std_score
        else:
            passing = np.ones(len(valid_scores), dtype=bool)

    elif thresh_method == LateralThreshMethod.IQR:
        # thresh_param is IQR multiplier (typically 1.5)
        q1 = np.percentile(valid_scores, 25)
        q3 = np.percentile(valid_scores, 75)
        iqr = q3 - q1
        upper_bound = q3 + thresh_param * iqr
        passing = valid_scores <= upper_bound

    else:
        raise ValueError(f"Unknown lateral thresh method: {thresh_method}")

    return valid_idx[passing]


def apply_edge_filter_custom(  # noqa: C901
    pc1_coords: np.ndarray,
    midpoint: float,
    valid_mask: np.ndarray,
    candidates: np.ndarray,
    thresh_method: str,
    thresh_param: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply edge filter. Returns (distal_pairs, proximal_pairs)."""
    # Compute midline distances
    midline_dist = np.abs(pc1_coords - midpoint)
    valid_dist = midline_dist[valid_mask]

    d_max = 1.0 if len(valid_dist) == 0 else np.nanmax(valid_dist)

    if d_max > 0:
        midline_dist_norm = midline_dist / d_max
    else:
        midline_dist_norm = np.zeros_like(midline_dist)

    # Generate all candidate pairs
    pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            # Check opposite sides of midpoint
            c1, c2 = candidates[i], candidates[j]
            if (pc1_coords[c1] - midpoint) * (pc1_coords[c2] - midpoint) < 0:
                pairs.append((c1, c2))

    if len(pairs) == 0:
        return np.array([]), np.array([])

    pairs = np.array(pairs)

    # Determine threshold based on method
    if thresh_method == EdgeThreshMethod.FIXED:
        edge_thresh = thresh_param

    elif thresh_method == EdgeThreshMethod.PERCENTILE:
        # thresh_param is percentile
        candidate_dists = midline_dist_norm[candidates]
        edge_thresh = np.percentile(candidate_dists, thresh_param)

    elif thresh_method == EdgeThreshMethod.ADAPTIVE_MIN_K:
        # Relax threshold until at least K distal pairs exist
        min_k = int(thresh_param)
        for test_thresh in np.arange(0.9, 0.0, -0.05):
            n_distal = 0
            for c1, c2 in pairs:
                min_dist = min(midline_dist_norm[c1], midline_dist_norm[c2])
                if min_dist >= test_thresh:
                    n_distal += 1
            if n_distal >= min_k:
                edge_thresh = test_thresh
                break
        else:
            edge_thresh = 0.0

    elif thresh_method == EdgeThreshMethod.BODY_LENGTH_NORM:
        # thresh_param is fraction of body length
        pc1_valid = pc1_coords[valid_mask]
        body_length = np.nanmax(pc1_valid) - np.nanmin(pc1_valid)
        edge_thresh = thresh_param if body_length > 0 else 0.5

    else:
        raise ValueError(f"Unknown edge thresh method: {thresh_method}")

    # Classify pairs
    distal_pairs = []
    proximal_pairs = []
    for c1, c2 in pairs:
        min_dist = min(midline_dist_norm[c1], midline_dist_norm[c2])
        if min_dist >= edge_thresh:
            distal_pairs.append((c1, c2))
        else:
            proximal_pairs.append((c1, c2))

    return np.array(distal_pairs), np.array(proximal_pairs)


def score_pairs(
    pairs: np.ndarray,
    ap_coords: np.ndarray,
    lateral_offsets_norm: np.ndarray,
    lateral_std_norm: np.ndarray,
    method: str,
) -> np.ndarray:
    """Score pairs for selection using specified method."""
    if len(pairs) == 0:
        return np.array([])

    scores = np.zeros(len(pairs))

    for k, (i, j) in enumerate(pairs):
        sep = abs(ap_coords[i] - ap_coords[j])
        lo_i, lo_j = lateral_offsets_norm[i], lateral_offsets_norm[j]
        ls_i, ls_j = lateral_std_norm[i], lateral_std_norm[j]
        lat_i = 0 if np.isnan(lo_i) else lo_i
        lat_j = 0 if np.isnan(lo_j) else lo_j
        std_i = 0 if np.isnan(ls_i) else ls_i
        std_j = 0 if np.isnan(ls_j) else ls_j

        if method == PairScoringMethod.MAX_SEPARATION:
            scores[k] = sep

        elif method == PairScoringMethod.WEIGHTED_LATERAL:
            avg_lateral = (lat_i + lat_j) / 2
            scores[k] = sep * (1 - avg_lateral)

        elif method == PairScoringMethod.WEIGHTED_VARIANCE:
            avg_std = (std_i + std_j) / 2
            scores[k] = sep * (1 - avg_std)

        elif method == PairScoringMethod.WEIGHTED_BOTH:
            avg_lateral = (lat_i + lat_j) / 2
            avg_std = (std_i + std_j) / 2
            scores[k] = sep * (1 - avg_lateral) * (1 - avg_std)

        else:
            raise ValueError(f"Unknown pair scoring method: {method}")

    return scores


def select_best_pair(
    distal_pairs: np.ndarray,
    proximal_pairs: np.ndarray,
    ap_coords: np.ndarray,
    lateral_offsets_norm: np.ndarray,
    lateral_std_norm: np.ndarray,
    scoring_method: str,
) -> tuple[int, int, str]:
    """Select best pair. Returns (posterior_node, anterior_node, pair_type)."""
    # Prefer distal pairs, fall back to proximal
    if len(distal_pairs) > 0:
        pairs = distal_pairs
        pair_type = "distal"
    elif len(proximal_pairs) > 0:
        pairs = proximal_pairs
        pair_type = "proximal"
    else:
        return -1, -1, ""

    scores = score_pairs(
        pairs,
        ap_coords,
        lateral_offsets_norm,
        lateral_std_norm,
        scoring_method,
    )

    best_idx = np.argmax(scores)
    c1, c2 = pairs[best_idx]

    # Order by AP coordinate (posterior first)
    if ap_coords[c1] < ap_coords[c2]:
        return c1, c2, pair_type
    else:
        return c2, c1, pair_type


# Main Validation Function


@dataclass
class ValidationResult:
    """Result of a single validation run."""

    dataset: str
    suggested_posterior: int
    suggested_anterior: int
    pair_type: str
    posterior_in_gt: bool
    anterior_in_gt: bool
    both_in_gt: bool
    order_correct: bool | None  # None if not both in GT
    n_step1_candidates: int
    n_step2_pairs: int
    n_distal_pairs: int


@dataclass
class GridSearchConfig:
    """Configuration for a single grid search iteration."""

    midpoint_method: str
    lateral_thresh_method: str
    lateral_thresh_param: float
    edge_thresh_method: str
    edge_thresh_param: float
    norm_method: str
    score_formula: str
    pair_scoring_method: str
    lateral_var_weight: float
    longitudinal_var_weight: float


@dataclass
class PrecomputedData:
    """Pre-computed data for a dataset (invariant to threshold params)."""

    dataset_name: str
    gt_nodes: list[int]
    n_keypoints: int
    valid_mask: np.ndarray
    pc1_coords: np.ndarray
    ap_coords: np.ndarray
    lateral_offsets: np.ndarray
    lateral_std: np.ndarray
    longitudinal_std: np.ndarray
    lat_std_norm: np.ndarray
    long_std_norm: np.ndarray
    success: bool = True
    failure_reason: str = ""


def precompute_dataset(  # noqa: C901
    ds_path: Path,
    individual: str,
    gt_nodes: list[int],
    base_config: ValidateAPConfig,
) -> PrecomputedData:
    """Pre-compute PCA and postural data for a dataset (done once).

    This mirrors the logic in validate_ap() but extracts the invariant parts.
    """

    def _make_failure(reason: str, n_kp: int = 0) -> PrecomputedData:
        return PrecomputedData(
            dataset_name=ds_path.stem,
            gt_nodes=gt_nodes,
            n_keypoints=n_kp,
            valid_mask=np.array([]),
            pc1_coords=np.array([]),
            ap_coords=np.array([]),
            lateral_offsets=np.array([]),
            lateral_std=np.array([]),
            longitudinal_std=np.array([]),
            lat_std_norm=np.array([]),
            long_std_norm=np.array([]),
            success=False,
            failure_reason=reason,
        )

    try:
        # Load dataset
        ds = load_poses_module.from_sleap_file(ds_path)
        position = ds.position.sel(individuals=individual)
        n_keypoints = position.sizes["keypoints"]
        num_frames = position.sizes["time"]

        # Extract keypoints array (time, keypoints, space)
        keypoints = position.values  # (time, space, keypoints)
        # Transpose to (time, keypoints, space)
        keypoints = np.transpose(keypoints, (0, 2, 1))

        # Dummy log functions (we don't need logging during precomputation)
        def _noop(*args, **kwargs):
            pass

        # Run motion segmentation
        seg = run_motion_segmentation(
            keypoints,
            num_frames,
            base_config,
            _noop,  # log_info
            _noop,  # log_warning
        )
        if seg is None:
            return _make_failure("Motion segmentation failed", n_keypoints)

        # Tier-2 frame selection
        t2 = select_tier2_frames(
            seg["segments"],
            seg["tier2_valid"],
            num_frames,
            _noop,
            _noop,
        )
        if t2 is None:
            return _make_failure("Not enough tier-2 valid frames", n_keypoints)

        selected_frames, selected_seg_id, num_selected = t2

        # Build centered skeletons
        _selected_centroids, centered_skeletons = build_centered_skeletons(
            keypoints, selected_frames
        )

        # Bundle frame selection data
        frame_sel = FrameSelection(
            frames=selected_frames,
            seg_ids=selected_seg_id,
            segments=seg["segments"],
            bbox_centroids=seg["bbox_centroids"],
            count=num_selected,
        )

        # Postural clustering + PCA + anterior inference
        pca = run_clustering_and_pca(
            centered_skeletons,
            frame_sel,
            base_config,
            _noop,
            _noop,
        )
        if pca is None:
            return _make_failure("Primary cluster PCA failed", n_keypoints)

        pr = pca["primary_result"]

        # Extract results
        # valid_shape_rows is a boolean mask of valid keypoints
        valid_shape_rows = pr["valid_shape_rows"]
        valid_mask = valid_shape_rows.copy()

        # proj_pc1 = PC1 coordinates (longitudinal axis)
        # proj_pc2 = PC2 coordinates (lateral offset from midline)
        # lateral_offsets = absolute distance from midline (unsigned)
        pc1_coords = pr["proj_pc1"]
        lateral_offsets = np.abs(pr["proj_pc2"])
        lateral_std = pr["lateral_std"]
        longitudinal_std = pr["longitudinal_std"]
        anterior_sign = pr["anterior_sign"]

        # AP coordinates (oriented so positive = anterior)
        ap_coords = anterior_sign * pc1_coords

        # Pre-normalize variance (invariant to config)
        lat_std_valid = lateral_std[valid_mask]
        if len(lat_std_valid) > 0:
            lat_std_max = float(np.nanmax(lat_std_valid))
        else:
            lat_std_max = 1.0
        lat_std_norm = np.full(n_keypoints, np.nan)
        if lat_std_max > 0:
            lat_std_norm[valid_mask] = lateral_std[valid_mask] / lat_std_max
        else:
            lat_std_norm[valid_mask] = 0.0

        long_std_valid = longitudinal_std[valid_mask]
        if len(long_std_valid) > 0:
            long_std_max = float(np.nanmax(long_std_valid))
        else:
            long_std_max = 1.0
        long_std_norm = np.full(n_keypoints, np.nan)
        if long_std_max > 0:
            long_std_norm[valid_mask] = long_std_valid / long_std_max
        else:
            long_std_norm[valid_mask] = 0.0

        return PrecomputedData(
            dataset_name=ds_path.stem,
            gt_nodes=gt_nodes,
            n_keypoints=n_keypoints,
            valid_mask=valid_mask,
            pc1_coords=pc1_coords,
            ap_coords=ap_coords,
            lateral_offsets=lateral_offsets,
            lateral_std=lateral_std,
            longitudinal_std=longitudinal_std,
            lat_std_norm=lat_std_norm,
            long_std_norm=long_std_norm,
            success=True,
        )

    except Exception as e:
        import traceback

        return _make_failure(f"{str(e)}\n{traceback.format_exc()}", 0)


def precompute_from_cached_validation(  # noqa: C901
    dataset_name: str,
    gt_nodes: list[int],
    val_result: dict,
) -> PrecomputedData:
    """Extract precomputed data from a cached validate_ap() result.

    Avoids re-running PCA by reusing results from R×M selection pass.
    """

    def _make_failure(reason: str, n_kp: int = 0) -> PrecomputedData:
        return PrecomputedData(
            dataset_name=dataset_name,
            gt_nodes=gt_nodes,
            n_keypoints=n_kp,
            valid_mask=np.array([]),
            pc1_coords=np.array([]),
            ap_coords=np.array([]),
            lateral_offsets=np.array([]),
            lateral_std=np.array([]),
            longitudinal_std=np.array([]),
            lat_std_norm=np.array([]),
            long_std_norm=np.array([]),
            success=False,
            failure_reason=reason,
        )

    if val_result is None:
        return _make_failure("No cached validation result")

    # Extract data from validate_ap() result
    avg_skeleton = val_result.get("avg_skeleton")
    pc1 = val_result.get("PC1")
    pc2 = val_result.get("PC2")
    lateral_std = val_result.get("lateral_std")
    longitudinal_std = val_result.get("longitudinal_std")
    anterior_sign = val_result.get("anterior_sign", 0)

    if avg_skeleton is None or pc1 is None:
        return _make_failure("Missing PCA data in cached result")

    avg_skeleton = np.array(avg_skeleton)
    pc1 = np.array(pc1)
    pc2 = np.array(pc2) if pc2 is not None else np.array([0.0, 1.0])
    if lateral_std is not None:
        lateral_std = np.array(lateral_std)
    if longitudinal_std is not None:
        longitudinal_std = np.array(longitudinal_std)

    n_keypoints = len(avg_skeleton)

    # Derive valid_mask from avg_skeleton (nodes without NaN)
    valid_mask = ~np.any(np.isnan(avg_skeleton), axis=1)

    if np.sum(valid_mask) < 2:
        return _make_failure("Fewer than 2 valid keypoints", n_keypoints)

    # Compute PC1 coordinates (projection onto PC1)
    pc1_coords = np.full(n_keypoints, np.nan)
    pc1_coords[valid_mask] = avg_skeleton[valid_mask] @ pc1

    # Compute lateral offsets (absolute PC2 projection = distance from midline)
    lateral_offsets = np.full(n_keypoints, np.nan)
    lateral_offsets[valid_mask] = np.abs(avg_skeleton[valid_mask] @ pc2)

    # AP coordinates (oriented so positive = anterior)
    ap_coords = anterior_sign * pc1_coords

    # Handle lateral_std and longitudinal_std
    if lateral_std is None:
        lateral_std = np.zeros(n_keypoints)
    if longitudinal_std is None:
        longitudinal_std = np.zeros(n_keypoints)

    # Pre-normalize variance
    lat_std_valid = lateral_std[valid_mask]
    if len(lat_std_valid) > 0:
        lat_std_max = float(np.nanmax(lat_std_valid))
    else:
        lat_std_max = 1.0
    lat_std_norm = np.full(n_keypoints, np.nan)
    if lat_std_max > 0:
        lat_std_norm[valid_mask] = lateral_std[valid_mask] / lat_std_max
    else:
        lat_std_norm[valid_mask] = 0.0

    long_std_valid = longitudinal_std[valid_mask]
    if len(long_std_valid) > 0:
        long_std_max = float(np.nanmax(long_std_valid))
    else:
        long_std_max = 1.0
    long_std_norm = np.full(n_keypoints, np.nan)
    if long_std_max > 0:
        long_std_norm[valid_mask] = long_std_valid / long_std_max
    else:
        long_std_norm[valid_mask] = 0.0

    return PrecomputedData(
        dataset_name=dataset_name,
        gt_nodes=gt_nodes,
        n_keypoints=n_keypoints,
        valid_mask=valid_mask,
        pc1_coords=pc1_coords,
        ap_coords=ap_coords,
        lateral_offsets=lateral_offsets,
        lateral_std=lateral_std,
        longitudinal_std=longitudinal_std,
        lat_std_norm=lat_std_norm,
        long_std_norm=long_std_norm,
        success=True,
    )


def precompute_all_datasets(
    base_config: ValidateAPConfig,
) -> dict[str, PrecomputedData]:
    """Pre-compute data for all datasets.

    Uses cached validation results from R×M selection to avoid PCA.
    """
    global DATASETS

    if not DATASETS:
        print("  Warning: No datasets found! Check SLP_DIR path.")
        return {}

    # Compute best individuals if not done (caches PCA results)
    if any(ds_info["individual"] is None for ds_info in DATASETS.values()):
        DATASETS = compute_all_best_individuals(DATASETS)

    print("\nExtracting PCA data from cached validation results...")
    precomputed = {}

    for file_stem, ds_info in tqdm(DATASETS.items(), desc="Extracting PCA"):
        gt_nodes = ds_info["gt_nodes"]
        label = ds_info["label"]
        val_result = ds_info.get("val_result")

        # Use cached validation result instead of re-running PCA
        data = precompute_from_cached_validation(
            file_stem, gt_nodes, val_result
        )
        precomputed[file_stem] = data

        if data.success:
            print(f"  {label}: OK ({data.n_keypoints} keypoints)")
        else:
            print(f"  {label}: FAILED - {data.failure_reason}")

    return precomputed


def run_validation_fast(
    precomputed: PrecomputedData,
    config: GridSearchConfig,
) -> ValidationResult:
    """Run validation using pre-computed data (fast path)."""
    if not precomputed.success:
        return ValidationResult(
            dataset=precomputed.dataset_name,
            suggested_posterior=-1,
            suggested_anterior=-1,
            pair_type="",
            posterior_in_gt=False,
            anterior_in_gt=False,
            both_in_gt=False,
            order_correct=None,
            n_step1_candidates=0,
            n_step2_pairs=0,
            n_distal_pairs=0,
        )

    # Normalize lateral offsets (depends on config.norm_method)
    d_norm = normalize_d(
        precomputed.lateral_offsets, precomputed.valid_mask, config.norm_method
    )

    # Compute midpoint (depends on config.midpoint_method)
    midpoint = compute_midpoint(
        precomputed.pc1_coords,
        precomputed.lateral_offsets,
        precomputed.lateral_std,
        precomputed.valid_mask,
        config.midpoint_method,
    )

    # Step 1: Lateral filter
    candidates = apply_lateral_filter_custom(
        d_norm,
        precomputed.lat_std_norm,
        precomputed.long_std_norm,
        precomputed.valid_mask,
        config.lateral_var_weight,
        config.longitudinal_var_weight,
        config.lateral_thresh_method,
        config.lateral_thresh_param,
        config.score_formula,
    )

    n_step1_candidates = len(candidates)

    if n_step1_candidates < 2:
        return ValidationResult(
            dataset=precomputed.dataset_name,
            suggested_posterior=-1,
            suggested_anterior=-1,
            pair_type="",
            posterior_in_gt=False,
            anterior_in_gt=False,
            both_in_gt=False,
            order_correct=None,
            n_step1_candidates=n_step1_candidates,
            n_step2_pairs=0,
            n_distal_pairs=0,
        )

    # Step 2 & 3: Edge filter
    distal_pairs, proximal_pairs = apply_edge_filter_custom(
        precomputed.pc1_coords,
        midpoint,
        precomputed.valid_mask,
        candidates,
        config.edge_thresh_method,
        config.edge_thresh_param,
    )

    n_step2_pairs = len(distal_pairs) + len(proximal_pairs)
    n_distal_pairs = len(distal_pairs)

    if n_step2_pairs == 0:
        return ValidationResult(
            dataset=precomputed.dataset_name,
            suggested_posterior=-1,
            suggested_anterior=-1,
            pair_type="",
            posterior_in_gt=False,
            anterior_in_gt=False,
            both_in_gt=False,
            order_correct=None,
            n_step1_candidates=n_step1_candidates,
            n_step2_pairs=0,
            n_distal_pairs=0,
        )

    # Select best pair
    posterior, anterior, pair_type = select_best_pair(
        distal_pairs,
        proximal_pairs,
        precomputed.ap_coords,
        d_norm,
        precomputed.lat_std_norm,
        config.pair_scoring_method,
    )

    # Check against GT
    gt_nodes = precomputed.gt_nodes
    posterior_in_gt = posterior in gt_nodes
    anterior_in_gt = anterior in gt_nodes
    both_in_gt = posterior_in_gt and anterior_in_gt

    # Check order if both in GT
    order_correct = None
    if both_in_gt:
        gt_rank_posterior = gt_nodes.index(posterior)
        gt_rank_anterior = gt_nodes.index(anterior)
        order_correct = gt_rank_posterior < gt_rank_anterior

    return ValidationResult(
        dataset=precomputed.dataset_name,
        suggested_posterior=posterior,
        suggested_anterior=anterior,
        pair_type=pair_type,
        posterior_in_gt=posterior_in_gt,
        anterior_in_gt=anterior_in_gt,
        both_in_gt=both_in_gt,
        order_correct=order_correct,
        n_step1_candidates=n_step1_candidates,
        n_step2_pairs=n_step2_pairs,
        n_distal_pairs=n_distal_pairs,
    )


# Grid Search


def generate_param_grid() -> list[GridSearchConfig]:
    """Generate all parameter combinations to test."""
    configs = []

    # Method options
    midpoint_methods = [
        MidpointMethod.GEOMETRIC_CENTER,
        MidpointMethod.CENTROID,
        MidpointMethod.MEDIAN,
        MidpointMethod.WEIGHTED_LATERAL,
        MidpointMethod.WEIGHTED_VARIANCE,
        MidpointMethod.TRIMMED_MEAN,
    ]

    lateral_methods_params = [
        (LateralThreshMethod.FIXED, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        (LateralThreshMethod.PERCENTILE, [30, 40, 50, 60, 70, 80]),
        (LateralThreshMethod.ADAPTIVE_MIN_K, [2, 3, 4, 5]),
        (LateralThreshMethod.ZSCORE, [0.5, 1.0, 1.5, 2.0]),
        (LateralThreshMethod.IQR, [1.0, 1.5, 2.0]),
    ]

    edge_methods_params = [
        (EdgeThreshMethod.FIXED, [0.3, 0.4, 0.5, 0.6, 0.7]),
        (EdgeThreshMethod.PERCENTILE, [30, 40, 50, 60, 70]),
        (EdgeThreshMethod.ADAPTIVE_MIN_K, [1, 2, 3]),
        (EdgeThreshMethod.BODY_LENGTH_NORM, [0.3, 0.4, 0.5, 0.6]),
    ]

    norm_methods = [
        NormalizationMethod.BODY_WIDTH,
        NormalizationMethod.MIN_MAX,
        NormalizationMethod.PERCENTILE_RANK,
    ]

    score_formulas = [
        EffectiveScoreFormula.ADDITIVE,
        EffectiveScoreFormula.MULTIPLICATIVE,
        EffectiveScoreFormula.MAX_BASED,
        EffectiveScoreFormula.RMS,
    ]

    pair_scoring_methods = [
        PairScoringMethod.MAX_SEPARATION,
        PairScoringMethod.WEIGHTED_LATERAL,
        PairScoringMethod.WEIGHTED_VARIANCE,
        PairScoringMethod.WEIGHTED_BOTH,
    ]

    var_weights = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        (1.0, 0.5),
        (1.0, 1.0),
    ]

    # Generate all combinations
    for midpoint_method in midpoint_methods:
        for lateral_method, lateral_params in lateral_methods_params:
            for lateral_param in lateral_params:
                for edge_method, edge_params in edge_methods_params:
                    for edge_param in edge_params:
                        for norm_method in norm_methods:
                            for score_formula in score_formulas:
                                for pair_scoring in pair_scoring_methods:
                                    for lat_w, long_w in var_weights:
                                        lat_m = lateral_method
                                        cfg = GridSearchConfig(
                                            midpoint_method=midpoint_method,
                                            lateral_thresh_method=lat_m,
                                            lateral_thresh_param=lateral_param,
                                            edge_thresh_method=edge_method,
                                            edge_thresh_param=edge_param,
                                            norm_method=norm_method,
                                            score_formula=score_formula,
                                            pair_scoring_method=pair_scoring,
                                            lateral_var_weight=lat_w,
                                            longitudinal_var_weight=long_w,
                                        )
                                        configs.append(cfg)

    return configs


def score_results(
    results: dict[str, ValidationResult],
) -> tuple[int, int, int]:
    """Score results. Returns (both, one, order) counts."""
    both_count = sum(1 for r in results.values() if r.both_in_gt)
    one_count = sum(
        1 for r in results.values() if r.posterior_in_gt or r.anterior_in_gt
    )
    order_count = sum(1 for r in results.values() if r.order_correct is True)
    return both_count, one_count, order_count


# Global variable for precomputed data (used by multiprocessing workers)
_PRECOMPUTED_DATA: dict[str, PrecomputedData] = {}


def _init_worker(precomputed: dict[str, PrecomputedData]):
    """Initialize worker with precomputed data."""
    global _PRECOMPUTED_DATA
    _PRECOMPUTED_DATA = precomputed


def run_single_config_fast(
    config: GridSearchConfig,
) -> tuple[
    GridSearchConfig, dict[str, ValidationResult], tuple[int, int, int]
]:
    """Run single config using global precomputed data."""
    global _PRECOMPUTED_DATA
    results = {}
    for ds_name, precomputed in _PRECOMPUTED_DATA.items():
        results[ds_name] = run_validation_fast(precomputed, config)
    score = score_results(results)
    return config, results, score


def run_validation_all_datasets_fast(
    config: GridSearchConfig,
    precomputed_data: dict[str, PrecomputedData],
) -> dict[str, ValidationResult]:
    """Run validation on all datasets using pre-computed data (fast path)."""
    results = {}
    for ds_name, precomputed in precomputed_data.items():
        results[ds_name] = run_validation_fast(precomputed, config)
    return results


def run_grid_search(
    n_workers: int = 1,
    output_dir: Path | None = None,
) -> list[
    tuple[GridSearchConfig, dict[str, ValidationResult], tuple[int, int, int]]
]:
    """Run full grid search with optimized precomputation."""
    base_config = ValidateAPConfig()

    # Pre-compute data once for all datasets
    precomputed_data = precompute_all_datasets(base_config)

    configs = generate_param_grid()
    print(f"Generated {len(configs)} configurations to test")
    n_configs = len(configs)
    n_ds = len(precomputed_data)
    total = n_configs * n_ds
    print(f"Estimated evaluations: {n_configs} × {n_ds} = {total:,}")

    # Run grid search
    all_results = []

    if n_workers > 1:
        # Use multiprocessing with precomputed data in global variable
        print(f"Running with {n_workers} parallel workers...")
        init_args = (precomputed_data,)
        pool_args = dict(initializer=_init_worker, initargs=init_args)
        with mp.Pool(n_workers, **pool_args) as pool:
            imap_iter = pool.imap_unordered(
                run_single_config_fast, configs, chunksize=100
            )
            for result in tqdm(imap_iter, total=n_configs, desc="Grid search"):
                all_results.append(result)
    else:
        # Single-threaded mode
        print("Running single-threaded...")
        for config in tqdm(configs, desc="Grid search"):
            results = run_validation_all_datasets_fast(
                config, precomputed_data
            )
            score = score_results(results)
            all_results.append((config, results, score))

    # Sort by score (both_in_gt first, then one_in_gt, then order_correct)
    all_results.sort(key=lambda x: (x[2][0], x[2][1], x[2][2]), reverse=True)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save top results
        fname = f"grid_search_top_results_{timestamp}.json"
        top_results_path = output_dir / fname
        top_results = []
        for config, results, score in all_results[:100]:
            top_results.append(
                {
                    "config": {
                        "midpoint_method": config.midpoint_method,
                        "lateral_thresh_method": config.lateral_thresh_method,
                        "lateral_thresh_param": config.lateral_thresh_param,
                        "edge_thresh_method": config.edge_thresh_method,
                        "edge_thresh_param": config.edge_thresh_param,
                        "norm_method": config.norm_method,
                        "score_formula": config.score_formula,
                        "pair_scoring_method": config.pair_scoring_method,
                        "lateral_var_weight": config.lateral_var_weight,
                        "long_var_weight": config.longitudinal_var_weight,
                    },
                    "score": {
                        "both_in_gt": score[0],
                        "one_in_gt": score[1],
                        "order_correct": score[2],
                    },
                    "per_dataset": {
                        name: {
                            "suggested": [
                                int(r.suggested_posterior),
                                int(r.suggested_anterior),
                            ],
                            "pair_type": r.pair_type,
                            "both_in_gt": bool(r.both_in_gt),
                            "posterior_in_gt": bool(r.posterior_in_gt),
                            "anterior_in_gt": bool(r.anterior_in_gt),
                            "order_correct": (
                                r.order_correct
                                if r.order_correct is None
                                else bool(r.order_correct)
                            ),
                        }
                        for name, r in results.items()
                    },
                }
            )

        with open(top_results_path, "w") as f:
            json.dump(top_results, f, indent=2)

        print(f"Saved top 100 results to: {top_results_path}")

    return all_results


def print_top_results(
    all_results: list[
        tuple[
            GridSearchConfig, dict[str, ValidationResult], tuple[int, int, int]
        ]
    ],
    n: int = 20,
):
    """Print top N results."""
    print("\n" + "=" * 80)
    print(f"TOP {n} CONFIGURATIONS")
    print("=" * 80)

    for i, (config, results, score) in enumerate(all_results[:n]):
        print(f"\n--- Rank {i + 1} ---")
        print(f"Score: both={score[0]}/5, one={score[1]}/5, order={score[2]}")
        print("Config:")
        print(f"  midpoint: {config.midpoint_method}")
        lat_m = config.lateral_thresh_method
        lat_p = config.lateral_thresh_param
        print(f"  lateral: {lat_m} (param={lat_p})")
        edge_m = config.edge_thresh_method
        edge_p = config.edge_thresh_param
        print(f"  edge: {edge_m} (param={edge_p})")
        print(f"  norm: {config.norm_method}")
        print(f"  formula: {config.score_formula}")
        print(f"  pair_scoring: {config.pair_scoring_method}")
        lat_w = config.lateral_var_weight
        long_w = config.longitudinal_var_weight
        print(f"  weights: lat={lat_w}, long={long_w}")
        print("Per-dataset:")
        for file_stem, r in results.items():
            label = FILE_LABELS.get(file_stem, file_stem[:15])
            if r.both_in_gt:
                status = "BOTH"
            elif r.posterior_in_gt or r.anterior_in_gt:
                status = "ONE"
            else:
                status = "NONE"
            if r.both_in_gt:
                oc = "correct" if r.order_correct else "wrong"
                order = f", order={oc}"
            else:
                order = ""
            sp, sa = r.suggested_posterior, r.suggested_anterior
            print(f"  {label}: [{sp} -> {sa}] {status}{order}")


# Main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search for AP threshold optimization"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel workers"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output dir"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Top results to print"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Disable logging"
    )
    args = parser.parse_args()

    # Use default OUTPUT_DIR if not specified
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"grid_search_{timestamp}.log"

    # Run with or without logging to file
    if args.no_log:
        # No file logging
        print("Starting grid search for AP threshold optimization...")
        print(f"Using {args.workers} workers")

        start_time = time.time()
        all_results = run_grid_search(
            n_workers=args.workers,
            output_dir=output_dir,
        )
        elapsed = time.time() - start_time

        print(f"\nGrid search completed in {elapsed:.1f} seconds")
        print_top_results(all_results, n=args.top)
    else:
        # Log to both console and file
        with TeeOutput(log_path):
            print(f"Grid search started at {datetime.now().isoformat()}")
            print(f"Log file: {log_path}")
            print()
            print("Starting grid search for AP threshold optimization...")
            print(f"Using {args.workers} workers")

            start_time = time.time()
            all_results = run_grid_search(
                n_workers=args.workers,
                output_dir=output_dir,
            )
            elapsed = time.time() - start_time

            print(f"\nGrid search completed in {elapsed:.1f} seconds")
            print_top_results(all_results, n=args.top)
            print(f"\nLog saved to: {log_path}")
