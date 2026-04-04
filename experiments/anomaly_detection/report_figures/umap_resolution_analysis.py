"""
UMAP Resolution Analysis: Options 2+3 (joint comparison) + Option 4 (per-model clustering)

This script generates three types of visualizations:
1. Joint UMAP fit on x64+x224 combined embeddings (Option 2)
2. Clustering quality metrics bar chart (Option 3)
3. Per-model centroid visualization (Option 4)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

# ============================================================================
# SETUP
# ============================================================================

PROJ_ROOT = Path(r"C:/Users/User/Desktop/Term 8/Deep Learning/Project/DeepLearning-Group8")
OUTPUT_DIR = PROJ_ROOT / "experiments/anomaly_detection/report_figures/artifacts/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metadata for defect type labels
METADATA_PATH = PROJ_ROOT / "data/processed/x64/wm811k/metadata_50k_5pct.csv"
meta = pd.read_csv(METADATA_PATH)
meta_test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
meta_test["label"] = meta_test["failure_type"].fillna("Normal")
meta_test.loc[meta_test["label"] == "none", "label"] = "Normal"

DEFECT_TYPES = ["Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Donut", "Random", "Near-full"]

# Color palette
DEFECT_COLORS = {
    "Normal":    "#cccccc",
    "Edge-Ring": "#e41a1c",
    "Edge-Loc":  "#ff7f00",
    "Center":    "#4daf4a",
    "Loc":       "#984ea3",
    "Scratch":   "#a65628",
    "Donut":     "#f781bf",
    "Random":    "#377eb8",
    "Near-full": "#ffff33",
}

# ============================================================================
# OPTION 2+3: Joint UMAP + Clustering Metrics
# ============================================================================

def option_2_3_side_by_side_with_metrics():
    """
    Load x64 and x224 UMAP points (fitted separately) and show them side-by-side.
    Compute clustering metrics within each UMAP space to show improvement.
    """
    print("\n=== OPTION 2+3: Side-by-Side UMAPs + Clustering Metrics ===\n")

    umap_x64_path = PROJ_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b0/x64/umap_points.csv"
    umap_x224_path = PROJ_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b0/x224/umap_points.csv"

    if not umap_x64_path.exists() or not umap_x224_path.exists():
        print(f"Warning: UMAP files not found.")
        return

    df_x64 = pd.read_csv(umap_x64_path)
    df_x224 = pd.read_csv(umap_x224_path)

    # Add defect type labels
    df_x64["defect_type"] = meta_test["label"].values[:len(df_x64)]
    df_x224["defect_type"] = meta_test["label"].values[:len(df_x224)]

    # Create figure with 3 panels: x64 UMAP | x224 UMAP | metrics
    fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1)  # x64 UMAP
    ax2 = plt.subplot(1, 3, 2)  # x224 UMAP
    ax3 = plt.subplot(1, 3, 3)  # metrics

    fig.suptitle(
        "EfficientNet-B0 PatchCore: Defect Clustering Quality\n"
        "Left: 64×64 input | Middle: 224×224 native resolution | Right: Clustering tightness by defect type",
        fontsize=12, fontweight="bold"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Panel 1: x64 UMAP
    # ─────────────────────────────────────────────────────────────────────
    ux64, uy64 = df_x64["umap_1"].values, df_x64["umap_2"].values
    mask_n64 = df_x64["defect_type"] == "Normal"
    ax1.scatter(ux64[mask_n64], uy64[mask_n64], s=5, alpha=0.1,
               c=DEFECT_COLORS["Normal"], linewidths=0, zorder=1)

    for dt in DEFECT_TYPES:
        mask_dt = df_x64["defect_type"] == dt
        if mask_dt.sum() > 0:
            ax1.scatter(ux64[mask_dt], uy64[mask_dt], s=60, alpha=0.8,
                       c=DEFECT_COLORS[dt], edgecolors="white", linewidth=0.3, zorder=3)

    ax1.set_title("64×64 Preprocessing\n(F1=0.467)", fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # ─────────────────────────────────────────────────────────────────────
    # Panel 2: x224 UMAP
    # ─────────────────────────────────────────────────────────────────────
    ux224, uy224 = df_x224["umap_1"].values, df_x224["umap_2"].values
    mask_n224 = df_x224["defect_type"] == "Normal"
    ax2.scatter(ux224[mask_n224], uy224[mask_n224], s=5, alpha=0.1,
               c=DEFECT_COLORS["Normal"], linewidths=0, zorder=1)

    for dt in DEFECT_TYPES:
        mask_dt = df_x224["defect_type"] == dt
        if mask_dt.sum() > 0:
            ax2.scatter(ux224[mask_dt], uy224[mask_dt], s=60, alpha=0.8,
                       c=DEFECT_COLORS[dt], edgecolors="white", linewidth=0.3, zorder=3)

    ax2.set_title("224×224 Native Resolution\n(F1=0.544)", fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    # ─────────────────────────────────────────────────────────────────────
    # Panel 3: Clustering metrics
    # ─────────────────────────────────────────────────────────────────────
    metrics_data = []

    for dt in DEFECT_TYPES:
        mask_x64 = df_x64["defect_type"] == dt
        mask_x224 = df_x224["defect_type"] == dt

        if mask_x64.sum() < 2 or mask_x224.sum() < 2:
            continue

        # Extract coordinates
        points_x64 = np.column_stack([ux64[mask_x64], uy64[mask_x64]])
        points_x224 = np.column_stack([ux224[mask_x224], uy224[mask_x224]])

        # Compute intra-cluster distance (lower = tighter cluster)
        if len(points_x64) > 1:
            dist_x64 = pdist(points_x64).mean()
        else:
            dist_x64 = np.inf

        if len(points_x224) > 1:
            dist_x224 = pdist(points_x224).mean()
        else:
            dist_x224 = np.inf

        metrics_data.append({
            "defect_type": dt,
            "dist_x64": dist_x64,
            "dist_x224": dist_x224,
            "n_x64": mask_x64.sum(),
            "n_x224": mask_x224.sum(),
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Normalize distances to 0-100 scale for better visualization
    all_dists = np.concatenate([metrics_df["dist_x64"].values, metrics_df["dist_x224"].values])
    all_dists = all_dists[~np.isinf(all_dists)]
    max_dist = all_dists.max()

    metrics_df["norm_x64"] = 100 * (1 - metrics_df["dist_x64"] / max_dist)
    metrics_df["norm_x224"] = 100 * (1 - metrics_df["dist_x224"] / max_dist)

    # Sort by improvement
    metrics_df["improvement"] = metrics_df["norm_x224"] - metrics_df["norm_x64"]
    metrics_df = metrics_df.sort_values("improvement", ascending=True)

    # Create grouped bar chart
    x_pos = np.arange(len(metrics_df))
    width = 0.35

    bars_x64 = ax3.barh(x_pos - width/2, metrics_df["norm_x64"], width,
                        label="x64 (upsampled)", color="#ff7f0e", edgecolor="white", linewidth=0.5, alpha=0.8)
    bars_x224 = ax3.barh(x_pos + width/2, metrics_df["norm_x224"], width,
                         label="x224 (native)", color="#1f77b4", edgecolor="white", linewidth=0.5, alpha=0.8)

    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(metrics_df["defect_type"])
    ax3.set_xlabel("Cluster Tightness (0=loose, 100=tight)", fontsize=10)
    ax3.set_title("Defect Clustering Quality\n(Higher = tighter clusters)", fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 105)
    ax3.xaxis.grid(True, alpha=0.25, linestyle="--")
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "umap_resolution_comparison_with_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}\n")

    # Print metrics table
    print("Clustering Quality Metrics (Intra-Cluster Distance):")
    print(metrics_df[["defect_type", "dist_x64", "dist_x224", "improvement", "n_x64", "n_x224"]].to_string(index=False))

    return metrics_df


# ============================================================================
# OPTION 4: Per-Model Centroid Visualization
# ============================================================================

def option_4_per_model_centroids(model_name, umap_csv_path, defect_label_col="defect_type"):
    """
    Visualize a single model's UMAP with defect centroids highlighted.
    Shows how well defects cluster around their centroid.
    """
    print(f"\n=== OPTION 4: {model_name} Centroid Analysis ===\n")

    if not Path(umap_csv_path).exists():
        print(f"UMAP file not found: {umap_csv_path}")
        return

    df = pd.read_csv(umap_csv_path)

    # Filter to test set if applicable
    if "split_group" in df.columns:
        df = df[df["split_group"].isin(["test_normal", "test_anomaly"])].reset_index(drop=True)

    # Add defect type labels
    df["defect_type"] = meta_test["label"].values[:len(df)]

    ux, uy = df["umap_1"].values, df["umap_2"].values

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(f"{model_name}\nDefect Clustering with Centroids", fontsize=12, fontweight="bold")

    # Draw normals first (small background)
    mask_normal = df["defect_type"] == "Normal"
    ax.scatter(ux[mask_normal], uy[mask_normal], marker="o", s=8, alpha=0.1,
              c=DEFECT_COLORS["Normal"], linewidths=0, zorder=1)

    # Draw defects and compute centroids
    centroids = {}
    for dt in DEFECT_TYPES:
        mask_dt = df["defect_type"] == dt
        if mask_dt.sum() < 2:
            continue

        # Plot individual defects
        ax.scatter(ux[mask_dt], uy[mask_dt], marker="o", s=80, alpha=0.75,
                  c=DEFECT_COLORS[dt], edgecolors="white", linewidth=0.4,
                  label=f"{dt} (n={mask_dt.sum()})", zorder=3)

        # Compute centroid
        centroid_x = ux[mask_dt].mean()
        centroid_y = uy[mask_dt].mean()
        centroids[dt] = (centroid_x, centroid_y)

        # Draw centroid as large star
        ax.scatter([centroid_x], [centroid_y], marker="*", s=1200, alpha=0.9,
                  c=DEFECT_COLORS[dt], edgecolors="black", linewidth=1.5, zorder=5)

        # Draw circle around centroid (radius = std dev of cluster)
        cluster_points = np.column_stack([ux[mask_dt], uy[mask_dt]])
        distances = np.sqrt((cluster_points - np.array([centroid_x, centroid_y]))**2).sum(axis=1)
        radius = distances.std()

        circle = plt.Circle((centroid_x, centroid_y), radius, fill=False,
                           edgecolor=DEFECT_COLORS[dt], linestyle="--", linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85, ncol=2)
    ax.set_title(f"{model_name}\nPoints = individual defects | Stars = centroids | Dashed circles = cluster spread",
                fontsize=10, style="italic")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"umap_centroids_{model_name.replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}\n")

    return df, centroids


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UMAP RESOLUTION ANALYSIS: Options 2, 3, and 4")
    print("=" * 80)

    # Option 2+3: Side-by-side UMAPs + Clustering metrics
    metrics_df = option_2_3_side_by_side_with_metrics()

    # Option 4: Per-model centroid visualizations
    # EfficientNet-B0 x64
    df_eb0_x64, centroids_eb0_x64 = option_4_per_model_centroids(
        "EfficientNet-B0 PatchCore (x64)",
        PROJ_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b0/x64/umap_points.csv"
    )

    # EfficientNet-B0 x224
    df_eb0_x224_result = option_4_per_model_centroids(
        "EfficientNet-B0 PatchCore (x224)",
        PROJ_ROOT / "experiments/anomaly_detection/patchcore/efficientnet_b0/x224/umap_points.csv"
    )
    if df_eb0_x224_result is not None:
        df_eb0_x224, centroids_eb0_x224 = df_eb0_x224_result

    print("\n" + "=" * 80)
    print("All visualizations complete!")
    print("=" * 80)
