from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "REPORT.md"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "report_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_overall_comparison_table(report_text: str) -> pd.DataFrame:
    lines = report_text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("| experiment"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("Could not find overall comparison table in REPORT.md")

    table_lines: list[str] = []
    for line in lines[start_idx:]:
        if not line.strip().startswith("|"):
            break
        table_lines.append(line)

    cleaned_lines = [table_lines[0]]
    cleaned_lines.extend(line for line in table_lines[2:])
    csv_like = "\n".join(cleaned_lines)
    df = pd.read_csv(StringIO(csv_like), sep="|", engine="python")
    df = df.drop(columns=[df.columns[0], df.columns[-1]])
    df.columns = [column.strip() for column in df.columns]
    df = df.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))

    numeric_columns = [
        "val-threshold precision",
        "val-threshold recall",
        "val-threshold F1",
        "AUROC",
        "AUPRC",
        "best sweep F1",
    ]
    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace("`", "", regex=False).astype(float)
    return df


def infer_family(row: pd.Series) -> str:
    model = row["model"]
    experiment = row["experiment"]
    if experiment.startswith("TS-"):
        return "Teacher-Distillation"
    if "PatchCore" in model:
        return "PatchCore"
    if "Autoencoder" in model:
        return "Autoencoder"
    if model == "VAE":
        return "VAE"
    if model == "Deep SVDD":
        return "SVDD"
    if "Backbone" in model:
        return "Backbone Baseline"
    return "Other"


def infer_autoencoder_subfamily(row: pd.Series) -> str:
    model = row["model"]
    experiment = row["experiment"]
    if "Residual Autoencoder" in model:
        return "Residual"
    if "BatchNorm + Dropout" in model:
        return "BatchNorm + Dropout"
    if "BatchNorm" in model:
        return "BatchNorm"
    if "128x128" in experiment:
        return "Resolution Variant"
    return "Plain Autoencoder"


def infer_patchcore_source(row: pd.Series) -> str:
    model = row["model"]
    experiment = row["experiment"]
    if "AE-BN Backbone" in model:
        return "AE-BN Encoder"
    if "WideResNet50-2" in model or "PatchCore-WideRes50" in experiment:
        return "WideResNet50-2"
    if "ResNet18" in model:
        return "ResNet18"
    if "ResNet50" in model:
        return "ResNet50"
    return "Other"


def infer_ts_backbone(row: pd.Series) -> str:
    model = row["model"]
    experiment = row["experiment"]
    if "WideResNet50-2" in model or "TS-WideRes50" in experiment:
        return "WideResNet50-2"
    if "ResNet50" in model or "TS-Res50" in experiment:
        return "ResNet50"
    if "ResNet18" in model or "TS-Res18" in experiment:
        return "ResNet18"
    return "Other"


def shorten_label(label: str, max_len: int = 28) -> str:
    return label if len(label) <= max_len else f"{label[:max_len - 3]}..."


def save_overall_plot(df: pd.DataFrame) -> None:
    colors = {
        "Teacher-Distillation": "#0f766e",
        "Autoencoder": "#b45309",
        "PatchCore": "#2563eb",
        "VAE": "#7c3aed",
        "SVDD": "#dc2626",
        "Backbone Baseline": "#4b5563",
        "Other": "#6b7280",
    }
    plot_df = df.copy()
    plot_df["color"] = plot_df["family"].map(colors).fillna(colors["Other"])

    top_df = plot_df.head(min(12, len(plot_df))).copy().iloc[::-1]

    plt.close("all")
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.09])

    ax_rank = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    ax_rank.barh(
        top_df["experiment"].map(shorten_label),
        top_df["val-threshold F1"],
        color=top_df["color"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_rank.set_title("Top Experiments by Validation-Threshold F1", pad=12)
    ax_rank.set_xlabel("Validation-Threshold F1")
    ax_rank.set_xlim(0, max(0.68, plot_df["val-threshold F1"].max() + 0.03))
    ax_rank.grid(axis="x", alpha=0.25, linestyle="--")

    for y_pos, (_, row) in enumerate(top_df.iterrows()):
        ax_rank.text(
            row["val-threshold F1"] + 0.008,
            y_pos,
            f"F1 {row['val-threshold F1']:.3f} | PR {row['AUPRC']:.3f}",
            va="center",
            fontsize=8.5,
        )

    sizes = 30 + (plot_df["AUROC"] - plot_df["AUROC"].min()) / max(
        plot_df["AUROC"].max() - plot_df["AUROC"].min(),
        1e-9,
    ) * 180
    for family, family_df in plot_df.groupby("family", sort=False):
        ax_scatter.scatter(
            family_df["AUPRC"],
            family_df["val-threshold F1"],
            s=sizes[family_df.index],
            c=family_df["color"],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
            label=family,
        )

    for _, row in plot_df.head(min(8, len(plot_df))).iterrows():
        ax_scatter.annotate(
            shorten_label(row["experiment"], max_len=20),
            (row["AUPRC"], row["val-threshold F1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax_scatter.set_title("All Experiments: AUPRC vs Validation-Threshold F1", pad=12)
    ax_scatter.set_xlabel("AUPRC")
    ax_scatter.set_ylabel("Validation-Threshold F1")
    ax_scatter.grid(alpha=0.25, linestyle="--")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for family, color in colors.items()
        if family in plot_df["family"].values
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(3, len(legend_handles)),
        frameon=False,
        title="Experiment Family",
    )

    fig.suptitle(
        "Overall Experiment Comparison\nLeft: top 12 by deployment-style F1 | Right: all runs, point size scaled by AUROC",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.14, hspace=0.18)

    output_path = OUTPUT_DIR / "overall_experiment_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved {output_path}")


def save_autoencoder_family_plot(df: pd.DataFrame) -> None:
    ae_df = df[df["family"] == "Autoencoder"].copy()
    if ae_df.empty:
        raise ValueError("No autoencoder-family rows found in overall comparison table")

    ae_df["subfamily"] = ae_df.apply(infer_autoencoder_subfamily, axis=1)
    ae_df = ae_df.sort_values("val-threshold F1", ascending=False).reset_index(drop=True)
    ae_df["plot_index"] = ae_df.index

    subgroup_colors = {
        "Plain Autoencoder": "#8b5e34",
        "BatchNorm": "#c2410c",
        "BatchNorm + Dropout": "#ea580c",
        "Residual": "#1d4ed8",
        "Resolution Variant": "#6b7280",
    }
    ae_df["color"] = ae_df["subfamily"].map(subgroup_colors).fillna("#6b7280")
    ranked_df = ae_df.iloc[::-1]

    plt.close("all")
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 0.09])

    ax_rank = fig.add_subplot(grid[0, 0])
    ax_tradeoff = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    ax_rank.barh(
        ranked_df["experiment"].map(lambda value: shorten_label(value, max_len=24)),
        ranked_df["val-threshold F1"],
        color=ranked_df["color"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_rank.set_title("Autoencoder Family Ranked by Validation-Threshold F1", pad=12)
    ax_rank.set_xlabel("Validation-Threshold F1")
    ax_rank.set_xlim(0, max(0.72, ae_df["val-threshold F1"].max() + 0.04))
    ax_rank.grid(axis="x", alpha=0.25, linestyle="--")

    for y_pos, (_, row) in enumerate(ranked_df.iterrows()):
        ax_rank.text(
            row["val-threshold F1"] + 0.008,
            y_pos,
            f"P {row['val-threshold precision']:.3f} | R {row['val-threshold recall']:.3f}",
            va="center",
            fontsize=8.2,
        )

    sizes = 50 + (ae_df["AUPRC"] - ae_df["AUPRC"].min()) / max(
        ae_df["AUPRC"].max() - ae_df["AUPRC"].min(),
        1e-9,
    ) * 230
    for subgroup, subgroup_df in ae_df.groupby("subfamily", sort=False):
        ax_tradeoff.scatter(
            subgroup_df["val-threshold recall"],
            subgroup_df["val-threshold precision"],
            s=sizes[subgroup_df["plot_index"]],
            c=subgroup_df["color"],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.45,
            label=subgroup,
        )

    for _, row in ae_df.head(min(7, len(ae_df))).iterrows():
        ax_tradeoff.annotate(
            shorten_label(row["experiment"], max_len=18),
            (row["val-threshold recall"], row["val-threshold precision"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax_tradeoff.set_title("Precision-Recall Tradeoff Within the AE Family", pad=12)
    ax_tradeoff.set_xlabel("Validation-Threshold Recall")
    ax_tradeoff.set_ylabel("Validation-Threshold Precision")
    ax_tradeoff.set_xlim(
        max(0.25, ae_df["val-threshold recall"].min() - 0.05),
        min(0.75, ae_df["val-threshold recall"].max() + 0.04),
    )
    ax_tradeoff.set_ylim(
        max(0.24, ae_df["val-threshold precision"].min() - 0.05),
        min(0.45, ae_df["val-threshold precision"].max() + 0.03),
    )
    ax_tradeoff.grid(alpha=0.25, linestyle="--")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=subgroup,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for subgroup, color in subgroup_colors.items()
        if subgroup in ae_df["subfamily"].values
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(3, len(legend_handles)),
        frameon=False,
        title="Autoencoder Subfamily",
    )

    fig.suptitle(
        "Autoencoder Experiment Family Comparison\nLeft: ranked by deployment-style F1 | Right: precision-recall tradeoff, point size scaled by AUPRC",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.16, hspace=0.18)

    output_path = OUTPUT_DIR / "autoencoder_family_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved {output_path}")


def save_baseline_family_plot(df: pd.DataFrame) -> None:
    baseline_df = df[df["family"].isin(["VAE", "SVDD", "Backbone Baseline"])].copy()
    if baseline_df.empty:
        raise ValueError("No baseline-family rows found in overall comparison table")

    colors = {
        "VAE": "#7c3aed",
        "SVDD": "#dc2626",
        "Backbone Baseline": "#4b5563",
    }
    baseline_df["color"] = baseline_df["family"].map(colors)
    baseline_df = baseline_df.sort_values("val-threshold F1", ascending=False).reset_index(drop=True)
    ranked_df = baseline_df.iloc[::-1]

    plt.close("all")
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.09])

    ax_rank = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    ax_rank.barh(
        ranked_df["experiment"].map(lambda value: shorten_label(value, max_len=24)),
        ranked_df["val-threshold F1"],
        color=ranked_df["color"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_rank.set_title("Compact Baseline Ranking by Validation-Threshold F1", pad=12)
    ax_rank.set_xlabel("Validation-Threshold F1")
    ax_rank.set_xlim(0, max(0.42, baseline_df["val-threshold F1"].max() + 0.05))
    ax_rank.grid(axis="x", alpha=0.25, linestyle="--")

    for y_pos, (_, row) in enumerate(ranked_df.iterrows()):
        ax_rank.text(
            row["val-threshold F1"] + 0.008,
            y_pos,
            f"AUROC {row['AUROC']:.3f} | PR {row['AUPRC']:.3f}",
            va="center",
            fontsize=8.4,
        )

    sizes = 60 + (baseline_df["AUROC"] - baseline_df["AUROC"].min()) / max(
        baseline_df["AUROC"].max() - baseline_df["AUROC"].min(),
        1e-9,
    ) * 220
    for family, family_df in baseline_df.groupby("family", sort=False):
        ax_scatter.scatter(
            family_df["val-threshold recall"],
            family_df["val-threshold precision"],
            s=sizes[family_df.index],
            c=family_df["color"],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.45,
            label=family,
        )

    for _, row in baseline_df.iterrows():
        ax_scatter.annotate(
            shorten_label(row["experiment"], max_len=18),
            (row["val-threshold recall"], row["val-threshold precision"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax_scatter.set_title("Compact Baseline Precision-Recall Tradeoff", pad=12)
    ax_scatter.set_xlabel("Validation-Threshold Recall")
    ax_scatter.set_ylabel("Validation-Threshold Precision")
    ax_scatter.grid(alpha=0.25, linestyle="--")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for family, color in colors.items()
        if family in baseline_df["family"].values
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(3, len(legend_handles)),
        frameon=False,
        title="Compact Baseline Family",
    )

    fig.suptitle(
        "Compact Baseline Comparison\nVAE variants, Deep SVDD, and the simple pretrained backbone baselines",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.16, hspace=0.18)

    output_path = OUTPUT_DIR / "compact_baseline_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved {output_path}")


def save_patchcore_family_plot(df: pd.DataFrame) -> None:
    patchcore_df = df[df["family"] == "PatchCore"].copy()
    if patchcore_df.empty:
        raise ValueError("No PatchCore rows found in overall comparison table")

    patchcore_df["source"] = patchcore_df.apply(infer_patchcore_source, axis=1)
    patchcore_df["score_clean"] = patchcore_df["score"].astype(str).str.replace("`", "", regex=False)
    patchcore_df = patchcore_df.sort_values("val-threshold F1", ascending=False).reset_index(drop=True)
    patchcore_df["plot_index"] = patchcore_df.index

    source_colors = {
        "AE-BN Encoder": "#b45309",
        "ResNet18": "#2563eb",
        "ResNet50": "#0f766e",
        "WideResNet50-2": "#7c2d12",
    }
    score_markers = {
        "mean": "o",
        "topk_mean": "s",
        "max": "^",
    }
    patchcore_df["color"] = patchcore_df["source"].map(source_colors).fillna("#6b7280")
    best_by_source = (
        patchcore_df.sort_values("val-threshold F1", ascending=False)
        .groupby("source", as_index=False)
        .first()
    )

    metrics = ["val-threshold F1", "AUPRC", "AUROC"]
    metric_labels = ["F1", "AUPRC", "AUROC"]
    x_positions = np.arange(len(best_by_source))
    width = 0.22

    plt.close("all")
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.11])

    ax_bar = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    for idx, metric in enumerate(metrics):
        ax_bar.bar(
            x_positions + (idx - 1) * width,
            best_by_source[metric],
            width=width,
            label=metric_labels[idx],
            color=["#111827", "#6d28d9", "#9ca3af"][idx],
            edgecolor="black",
            linewidth=0.4,
        )
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(best_by_source["source"])
    ax_bar.set_ylim(0, 0.92)
    ax_bar.set_title("Best PatchCore Variant per Backbone / Source", pad=12)
    ax_bar.set_ylabel("Metric Value")
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")
    ax_bar.legend(frameon=False, loc="upper left")

    sizes = 45 + (patchcore_df["best sweep F1"] - patchcore_df["best sweep F1"].min()) / max(
        patchcore_df["best sweep F1"].max() - patchcore_df["best sweep F1"].min(),
        1e-9,
    ) * 180
    for _, row in patchcore_df.iterrows():
        ax_scatter.scatter(
            row["AUPRC"],
            row["val-threshold F1"],
            s=sizes[row["plot_index"]],
            c=row["color"],
            marker=score_markers.get(row["score_clean"], "o"),
            alpha=0.82,
            edgecolors="black",
            linewidths=0.45,
        )

    for _, row in best_by_source.iterrows():
        ax_scatter.annotate(
            shorten_label(row["experiment"], max_len=20),
            (row["AUPRC"], row["val-threshold F1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax_scatter.set_title("All PatchCore Variants: AUPRC vs Validation-Threshold F1", pad=12)
    ax_scatter.set_xlabel("AUPRC")
    ax_scatter.set_ylabel("Validation-Threshold F1")
    ax_scatter.grid(alpha=0.25, linestyle="--")

    source_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=source,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for source, color in source_colors.items()
        if source in patchcore_df["source"].values
    ]
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            label=score_name,
            markersize=7,
        )
        for score_name, marker in score_markers.items()
        if score_name in patchcore_df["score_clean"].values
    ]
    legend_sources = ax_legend.legend(
        handles=source_handles,
        loc="center left",
        bbox_to_anchor=(0.18, 0.5),
        ncol=len(source_handles),
        frameon=False,
        title="PatchCore Backbone / Source",
    )
    ax_legend.add_artist(legend_sources)
    if marker_handles:
        ax_legend.legend(
            handles=marker_handles,
            loc="center right",
            bbox_to_anchor=(0.82, 0.5),
            ncol=len(marker_handles),
            frameon=False,
            title="Wafer-Level Reduction",
        )

    fig.suptitle(
        "PatchCore Family Comparison\nLeft: best run per backbone / source | Right: all PatchCore variants, point size scaled by best-sweep F1",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.16, hspace=0.18)

    output_path = OUTPUT_DIR / "patchcore_family_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved {output_path}")


def save_ts_family_plot(df: pd.DataFrame) -> None:
    ts_df = df[df["family"] == "Teacher-Distillation"].copy()
    if ts_df.empty:
        raise ValueError("No teacher-distillation rows found in overall comparison table")

    ts_df["backbone"] = ts_df.apply(infer_ts_backbone, axis=1)
    ts_df = ts_df.sort_values(["backbone", "val-threshold F1"], ascending=[True, False]).reset_index(drop=True)
    ts_df["plot_index"] = ts_df.index
    colors = {
        "TS-Res18-student-topk20": "#2563eb",
        "TS-Res50-mixed-topk20": "#0f766e",
        "TS-WideRes50-layer2-mixed-topk25": "#c2410c",
        "TS-WideRes50-multilayer-mixed-topk15": "#7c2d12",
    }
    ts_df["color"] = ts_df["experiment"].map(colors).fillna("#6b7280")

    metrics = ["val-threshold F1", "AUPRC", "AUROC"]
    metric_labels = ["F1", "AUPRC", "AUROC"]
    x_positions = np.arange(len(ts_df))
    width = 0.22

    plt.close("all")
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.09])

    ax_bar = fig.add_subplot(grid[0, 0])
    ax_tradeoff = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    metric_colors = ["#111827", "#7c3aed", "#9ca3af"]
    for idx, metric in enumerate(metrics):
        ax_bar.bar(
            x_positions + (idx - 1) * width,
            ts_df[metric],
            width=width,
            label=metric_labels[idx],
            color=metric_colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(ts_df["experiment"].map(lambda value: shorten_label(value, max_len=18)))
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("Metric Value")
    ax_bar.set_title("Teacher-Distillation Variants Across Core Metrics", pad=12)
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")
    ax_bar.legend(frameon=False, loc="upper left")

    sizes = 80 + (ts_df["best sweep F1"] - ts_df["best sweep F1"].min()) / max(
        ts_df["best sweep F1"].max() - ts_df["best sweep F1"].min(),
        1e-9,
    ) * 240
    ax_tradeoff.scatter(
        ts_df["val-threshold recall"],
        ts_df["val-threshold precision"],
        s=sizes,
        c=ts_df["color"],
        alpha=0.84,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in ts_df.iterrows():
        ax_tradeoff.annotate(
            shorten_label(row["experiment"], max_len=18),
            (row["val-threshold recall"], row["val-threshold precision"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax_tradeoff.set_title("Teacher-Distillation Operating-Point Tradeoff", pad=12)
    ax_tradeoff.set_xlabel("Validation-Threshold Recall")
    ax_tradeoff.set_ylabel("Validation-Threshold Precision")
    ax_tradeoff.set_xlim(0.60, 0.73)
    ax_tradeoff.set_ylim(0.36, 0.44)
    ax_tradeoff.grid(alpha=0.25, linestyle="--")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=row["experiment"],
            markerfacecolor=row["color"],
            markeredgecolor="black",
            markersize=8,
        )
        for _, row in ts_df.iterrows()
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(2, len(legend_handles)),
        frameon=False,
        title="Teacher-Distillation Variant",
    )

    fig.suptitle(
        "Teacher-Distillation Family Comparison\nMetric and operating-point view of the ResNet18, ResNet50, and WideResNet50-2 teacher variants",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.16, hspace=0.18)

    output_path = OUTPUT_DIR / "ts_family_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved {output_path}")


def save_wrn_family_plot(df: pd.DataFrame) -> None:
    representative_experiments = [
        "WideRes50-center",
        "TS-WideRes50-layer2-mixed-topk25",
        "TS-WideRes50-multilayer-mixed-topk15",
        "PatchCore-WideRes50-topk-mb50k-r010",
    ]
    wrn_df = df[df["experiment"].isin(representative_experiments)].copy()
    if wrn_df.empty:
        raise ValueError("No WideResNet50-2 rows found in overall comparison table")

    wrn_df = wrn_df.sort_values("val-threshold F1", ascending=False).reset_index(drop=True)
    wrn_df["plot_index"] = wrn_df.index
    wrn_df["variant_group"] = np.select(
        [
            wrn_df["experiment"].eq("WideRes50-center"),
            wrn_df["experiment"].str.startswith("TS-WideRes50"),
            wrn_df["experiment"].str.startswith("PatchCore-WideRes50"),
        ],
        [
            "Backbone Baseline",
            "Teacher-Distillation",
            "PatchCore",
        ],
        default="Other",
    )
    group_colors = {
        "Backbone Baseline": "#4b5563",
        "Teacher-Distillation": "#c2410c",
        "PatchCore": "#0f766e",
        "Other": "#6b7280",
    }
    wrn_df["color"] = wrn_df["variant_group"].map(group_colors).fillna(group_colors["Other"])
    wrn_df["score_clean"] = wrn_df["score"].astype(str).str.replace("`", "", regex=False)
    score_markers = {
        "center_l2": "o",
        "topk_mean": "s",
        "mean": "^",
        "max": "D",
    }

    plt.close("all")
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.09])

    ax_rank = fig.add_subplot(grid[0, 0])
    ax_tradeoff = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[1, :])
    ax_legend.axis("off")

    ranked_df = wrn_df.iloc[::-1]
    ax_rank.barh(
        ranked_df["experiment"].map(
            lambda value: {
                "WideRes50-center": "WideRes50-center",
                "TS-WideRes50-layer2-mixed-topk25": "TS-WideRes50-layer2",
                "TS-WideRes50-multilayer-mixed-topk15": "TS-WideRes50-multilayer",
                "PatchCore-WideRes50-topk-mb50k-r010": "PatchCore-WideRes50-topk",
            }.get(value, shorten_label(value, max_len=26))
        ),
        ranked_df["val-threshold F1"],
        color=ranked_df["color"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_rank.set_title("WideResNet50-2 Variants Ranked by Validation-Threshold F1", pad=12)
    ax_rank.set_xlabel("Validation-Threshold F1")
    ax_rank.set_xlim(0, max(0.62, wrn_df["val-threshold F1"].max() + 0.05))
    ax_rank.grid(axis="x", alpha=0.25, linestyle="--")

    for y_pos, (_, row) in enumerate(ranked_df.iterrows()):
        ax_rank.text(
            row["val-threshold F1"] + 0.008,
            y_pos,
            f"AUROC {row['AUROC']:.3f} | PR {row['AUPRC']:.3f}",
            va="center",
            fontsize=8.4,
        )

    sizes = 80 + (wrn_df["best sweep F1"] - wrn_df["best sweep F1"].min()) / max(
        wrn_df["best sweep F1"].max() - wrn_df["best sweep F1"].min(),
        1e-9,
    ) * 240
    for _, row in wrn_df.iterrows():
        ax_tradeoff.scatter(
            row["val-threshold recall"],
            row["val-threshold precision"],
            s=sizes[row["plot_index"]],
            c=row["color"],
            marker=score_markers.get(row["score_clean"], "o"),
            alpha=0.84,
            edgecolors="black",
            linewidths=0.5,
        )
        ax_tradeoff.annotate(
            {
                "WideRes50-center": "WideRes50-center",
                "TS-WideRes50-layer2-mixed-topk25": "TS-WideRes50-layer2",
                "TS-WideRes50-multilayer-mixed-topk15": "TS-WideRes50-multilayer",
                "PatchCore-WideRes50-topk-mb50k-r010": "PatchCore-WideRes50-topk",
            }.get(row["experiment"], shorten_label(row["experiment"], max_len=22)),
            (row["val-threshold recall"], row["val-threshold precision"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax_tradeoff.set_title("WideResNet50-2 Precision-Recall Operating Points", pad=12)
    ax_tradeoff.set_xlabel("Validation-Threshold Recall")
    ax_tradeoff.set_ylabel("Validation-Threshold Precision")
    ax_tradeoff.set_xlim(
        max(0.20, wrn_df["val-threshold recall"].min() - 0.03),
        min(0.75, wrn_df["val-threshold recall"].max() + 0.03),
    )
    ax_tradeoff.set_ylim(
        max(0.20, wrn_df["val-threshold precision"].min() - 0.01),
        min(0.44, wrn_df["val-threshold precision"].max() + 0.01),
    )
    ax_tradeoff.grid(alpha=0.25, linestyle="--")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=group,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for group, color in group_colors.items()
        if group in wrn_df["variant_group"].values
    ]
    legend_groups = ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.22, 0.5),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        title="WideResNet50-2 Family Branch",
    )
    ax_legend.add_artist(legend_groups)
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            label=score_name,
            markersize=7,
        )
        for score_name, marker in score_markers.items()
        if score_name in wrn_df["score_clean"].values
    ]
    if marker_handles:
        ax_legend.legend(
            handles=marker_handles,
            loc="center right",
            bbox_to_anchor=(0.82, 0.5),
            ncol=min(4, len(marker_handles)),
            frameon=False,
            title="Score",
        )

    fig.suptitle(
        "WideResNet50-2 Family Comparison\nRepresentative WRN experiments: baseline, single-layer TS, multilayer TS, and selected PatchCore run",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.98, wspace=0.16, hspace=0.18)

    output_path = OUTPUT_DIR / "wrn_family_comparison.png"
    fig.savefig(output_path, dpi=220)
    print(f"Saved {output_path}")


def main() -> None:
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    df = extract_overall_comparison_table(report_text)
    df["family"] = df.apply(infer_family, axis=1)
    df = df.sort_values("val-threshold F1", ascending=False).reset_index(drop=True)

    save_overall_plot(df)
    save_autoencoder_family_plot(df)
    save_baseline_family_plot(df)
    save_patchcore_family_plot(df)
    save_ts_family_plot(df)
    save_wrn_family_plot(df)


if __name__ == "__main__":
    main()
