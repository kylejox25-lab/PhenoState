"""Plots for descriptive, common-well representation diagnostics.

Projection fitting is exploratory only; quantitative scores use the original
feature space. All selections depend on metadata and a seed, never model scores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELATIONS = {
    "repeat_same_plate": "Repeat\nsame plate",
    "repeat_cross_plate": "Repeat\ncross plate",
    "nonrepeat_cross_plate": "Nonrepeat\ncross plate",
    "nonrepeat_same_plate": "Nonrepeat\nsame plate",
    "treated_control": "Treated\nvs control",
    "control_control_same_plate": "Controls\nsame plate",
    "control_control_cross_plate": "Controls\ncross plate",
    "control_control_cross_experiment": "Controls\ncross experiment",
    "repeat_cross_experiment": "Repeat\ncross experiment",
    "nonrepeat_cross_experiment": "Nonrepeat\ncross experiment",
}
KIND_COLORS = {"CRISPR": "#3974af", "COMPOUND": "#da813b"}


def _empty(ax, message="NA: no eligible observations"):
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _panels(models, *, width=4.0, height=3.5):
    count = max(1, len(models))
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def _finite(frame, field):
    if field not in frame:
        return np.empty(0)
    values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def _normalise(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite [well, feature] matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Zero vectors cannot define cosine similarity")
    return values / norms[:, None]


def plot_pair_distributions(pair_groups, models, *, zoom=False):
    """Each point is an already aggregated perturbation/control group."""
    frame = pd.DataFrame(pair_groups)
    kinds = [kind for kind in ("CRISPR", "COMPOUND") if "kind" in frame and frame.kind.eq(kind).any()]
    kinds = kinds or ["CRISPR", "COMPOUND"]
    available = [relation for relation in RELATIONS if "relation" in frame and frame.relation.eq(relation).any()]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(max(12, len(available) * 1.6), 4.6 * len(kinds)), squeeze=False)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, max(1, len(models))))
    rng = np.random.default_rng(2026)
    width = 0.8 / max(1, len(models))
    limits = (-1.02, 1.02)
    if zoom:
        all_values = _finite(frame, "similarity")
        if len(all_values):
            low, high = all_values.min(), all_values.max()
            pad = max((high - low) * 0.08, 0.005)
            limits = (max(-1.02, low - pad), min(1.02, high + pad))
    for ax, kind in zip(axes.flat, kinds):
        plotted = False
        for mi, model in enumerate(models):
            selected = frame.loc[frame.get("model", pd.Series(index=frame.index, dtype=str)).eq(model)
                                 & frame.get("kind", pd.Series(index=frame.index, dtype=str)).eq(kind)]
            for ri, relation in enumerate(available):
                group = selected.loc[selected.relation.eq(relation)]
                if relation.startswith("control_control") and "perturbation" in group:
                    group = group.loc[group.perturbation.astype(str).str.startswith("control:")]
                values = _finite(group, "similarity")
                if not len(values):
                    continue
                position = ri - 0.4 + width * (mi + 0.5)
                result = ax.boxplot([values], positions=[position], widths=width * 0.85,
                                    patch_artist=True, manage_ticks=False, showfliers=False)
                result["boxes"][0].set(facecolor=colors[mi], alpha=0.5)
                for median in result["medians"]:
                    median.set(color="#222222")
                sample = values[rng.choice(len(values), min(50, len(values)), replace=False)]
                ax.scatter(position + rng.uniform(-width * 0.3, width * 0.3, len(sample)), sample,
                           s=5, alpha=0.35, color=colors[mi])
                plotted = True
            ax.scatter([], [], color=colors[mi], label=model)
        if not plotted:
            _empty(ax)
        else:
            ax.set_xticks(range(len(available)), [RELATIONS[x] for x in available], fontsize=8)
            ax.set_ylim(*limits)
            ax.set_ylabel("Signed cosine similarity")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, ncol=min(4, max(1, len(models))))
        ax.set_title(kind + " | group-level distributions" + (" | shared observed scale (zoom)" if zoom else " | full cosine scale"))
    fig.tight_layout()
    return fig


def plot_effects(effects, models):
    frame = pd.DataFrame(effects)
    fig, axes = _panels(models)
    for ax, model in zip(axes, models):
        selected = frame.loc[frame.model.eq(model)] if "model" in frame else frame
        plotted = False
        for kind, color in KIND_COLORS.items():
            group = selected.loc[selected.kind.eq(kind)] if "kind" in selected else selected.iloc[:0]
            if not {"control_deviation", "replicate_advantage"} <= set(group):
                continue
            valid = group[["control_deviation", "replicate_advantage"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid):
                ax.scatter(valid.control_deviation, valid.replicate_advantage, s=12, alpha=0.55,
                           color=color, label=kind)
                plotted = True
        if not plotted:
            _empty(ax, "NA: both matched controls and cross-plate repeats required")
        else:
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.set(xlabel="Control-calibrated phenotype deviation", ylabel="Cross-plate replicate advantage")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.15)
        ax.set_title(model)
    # Identical quantitative axes prevent panel-specific zoom from disguising differences.
    xvalues, yvalues = _finite(frame, "control_deviation"), _finite(frame, "replicate_advantage")
    if len(xvalues) and len(yvalues):
        xvalues, yvalues = np.r_[xvalues, 0], np.r_[yvalues, 0]
        xpad = max(np.ptp(xvalues) * 0.06, 0.005)
        ypad = max(np.ptp(yvalues) * 0.06, 0.005)
        for ax in axes:
            ax.set_xlim(xvalues.min() - xpad, xvalues.max() + xpad)
            ax.set_ylim(yvalues.min() - ypad, yvalues.max() + ypad)
    fig.tight_layout()
    return fig


def plot_effect_summary(effects, models):
    frame = pd.DataFrame(effects)
    metrics = [("replicate_advantage", "Cross-plate replicate advantage (higher)"),
               ("competition", "Cross-plate repeat minus same-plate nonrepeat (higher)"),
               ("plate_advantage", "Same-plate nonrepeat advantage (zero reference)"),
               ("experiment_replicate_advantage", "Cross-experiment replicate advantage (higher)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    for ax, (field, title) in zip(axes.flat, metrics):
        plotted = False
        for mi, model in enumerate(models):
            for ki, (kind, color) in enumerate(KIND_COLORS.items()):
                group = frame.loc[frame.model.eq(model) & frame.kind.eq(kind)] if {"model", "kind"} <= set(frame) else frame.iloc[:0]
                values = _finite(group, field)
                if not len(values):
                    continue
                position = mi + (-0.17 if ki == 0 else 0.17)
                result = ax.boxplot([values], positions=[position], widths=0.28, patch_artist=True,
                                    showfliers=False, manage_ticks=False)
                result["boxes"][0].set(facecolor=color, alpha=0.6)
                ax.scatter(position, values.mean(), marker="D", color="#222222", s=16, zorder=4)
                plotted = True
        if not plotted:
            _empty(ax)
        else:
            ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.grid(axis="y", alpha=0.2)
            for kind, color in KIND_COLORS.items():
                ax.scatter([], [], color=color, label=kind)
            ax.scatter([], [], marker="D", color="#222222", label="Group mean")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Cosine difference")
    fig.tight_layout()
    return fig


def _balanced_positions(meta, max_points, seed, strata):
    """Round-robin within metadata strata; no feature-dependent selection."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(meta) <= max_points:
        return np.arange(len(meta), dtype=int)
    rng = np.random.default_rng(seed)
    fields = [field for field in strata if field in meta]
    groups = [np.asarray(indices, dtype=int) for indices in meta.groupby(fields, sort=True, dropna=False).indices.values()] if fields else [np.arange(len(meta))]
    groups = [rng.permutation(group) for group in groups]
    order = rng.permutation(len(groups))
    selected = []
    depth = 0
    while len(selected) < max_points:
        found = False
        for index in order:
            if depth < len(groups[index]):
                selected.append(int(groups[index][depth]))
                found = True
                if len(selected) == max_points:
                    break
        if not found:
            break
        depth += 1
    return np.sort(selected)


def make_projection_coordinates(embeddings: Mapping, meta, models: Sequence, *, subset,
                                method="PCA", max_points=2000, seed=2026, max_perturbations=12):
    """Fit one projection per model on identical, metadata-selected wells.

    Returned coordinates must be reused for different label colorings. PCA/UMAP
    fits here are descriptive and are not used for retrieval or score fitting.
    """
    from sklearn.decomposition import PCA

    meta = pd.DataFrame(meta).reset_index(drop=True).copy()
    if "well_id" not in meta or meta.well_id.duplicated().any():
        raise ValueError("Projection metadata requires unique well_id")
    if subset not in {"all", "controls", "shared"}:
        raise ValueError("subset must be all, controls, or shared")
    method = method.upper()
    if method not in {"PCA", "UMAP"}:
        raise ValueError("method must be PCA or UMAP")
    if method == "UMAP":
        try:
            import umap
        except ImportError as error:
            raise ImportError("UMAP is optional. Install umap-learn or explicitly select method='PCA'.") from error
    selected = meta.copy()
    if subset == "controls":
        selected = selected.loc[selected.get("is_control", pd.Series(False, index=selected.index)).astype(bool)]
    elif subset == "shared":
        selected = selected.loc[~selected.get("is_control", pd.Series(False, index=selected.index)).astype(bool)]
        if "perturbation" not in selected or not len(selected):
            selected = selected.iloc[:0]
        else:
            group_keys = [column for column in ("kind", "context", "perturbation") if column in selected]
            stats = selected.groupby(group_keys, dropna=False).agg(
                n_plates=("batch_id", "nunique"), n_experiments=("experiment_name", "nunique"))
            stats = stats.loc[stats.n_plates >= 2].reset_index()
            rng = np.random.default_rng(seed)
            stats["tie"] = rng.random(len(stats))
            stats = stats.sort_values(["n_experiments", "n_plates", "tie"], ascending=[False, False, True])
            # Keep both assay kinds visible: rank within kind, then round-robin.
            if "kind" in stats:
                queues = [group.index.tolist() for _, group in stats.groupby("kind", sort=True)]
                selected_groups = []
                for depth in range(max(map(len, queues), default=0)):
                    for queue in queues:
                        if depth < len(queue) and len(selected_groups) < max_perturbations:
                            selected_groups.append(queue[depth])
                    if len(selected_groups) >= max_perturbations:
                        break
                chosen = stats.loc[selected_groups, group_keys].assign(_chosen=True)
            else:
                chosen = stats.head(max_perturbations)[group_keys].assign(_chosen=True)
            selected = selected.assign(_original_position=selected.index).merge(chosen, on=group_keys, how="inner")
            selected = selected.set_index("_original_position").drop(columns="_chosen")
    if len(selected):
        selected = selected.iloc[_balanced_positions(selected.reset_index(drop=True), max_points, seed,
                                                      ["kind", "control_family", "cell_type", "batch_id"] if subset == "controls"
                                                      else ["kind", "perturbation", "batch_id"] if subset == "shared"
                                                      else ["kind", "batch_id"])]
    positions = selected.index.to_numpy(dtype=int)
    selected = selected.reset_index(drop=True)
    info = {"subset": subset, "method": method, "n_selected": len(selected), "seed": seed,
            "selection": "metadata-only balanced selection; shared ranks by batch coverage, never model scores",
            "fit_scope": "exploratory selected wells; not a held-out performance transform", "models": {}}
    coordinates = []
    for model in models:
        values = np.asarray(embeddings[model])
        if values.ndim != 2 or len(values) != len(meta):
            raise ValueError(f"{model}: embedding/metadata row mismatch")
        if len(selected) < 3 or values.shape[1] < 2:
            info["models"][model] = {"reason": "Need at least 3 wells and 2 feature dimensions"}
            continue
        values = _normalise(values[positions])
        if np.linalg.norm(values - values.mean(axis=0)) <= np.finfo(float).eps:
            info["models"][model] = {"reason": "Constant feature matrix has no projection structure"}
            continue
        if method == "PCA":
            estimator = PCA(n_components=2, random_state=seed)
            xy = estimator.fit_transform(values)
            info["models"][model] = {"explained_variance_ratio": estimator.explained_variance_ratio_.tolist()}
        else:
            if len(values) < 4:
                info["models"][model] = {"reason": "UMAP requires at least 4 selected wells"}
                continue
            components = min(50, values.shape[1], len(values) - 1)
            reduced = PCA(n_components=components, random_state=seed).fit_transform(values)
            estimator = umap.UMAP(n_components=2, n_neighbors=min(15, len(values) - 1),
                                  metric="euclidean", random_state=seed, init="random")
            xy = estimator.fit_transform(reduced)
            info["models"][model] = {"pca_dimensions": components, "n_neighbors": min(15, len(values) - 1)}
        result = selected.copy()
        result["model"] = model
        result["x"], result["y"] = xy[:, 0], xy[:, 1]
        result["projection_method"] = method
        coordinates.append(result)
    columns = list(dict.fromkeys([*meta.columns, "model", "x", "y", "projection_method"]))
    output = pd.concat(coordinates, ignore_index=True) if coordinates else pd.DataFrame(columns=columns)
    output.attrs["projection_info"] = info
    return output, info


def plot_projections(coords, models, color_by="experiment_name"):
    frame = pd.DataFrame(coords)
    fig, axes = _panels(models, height=3.6)
    field = "batch_id" if color_by == "plate" else color_by
    labels = sorted(frame[field].fillna("NA").astype(str).unique()) if field in frame else []
    cmap = plt.get_cmap("tab20" if len(labels) <= 20 else "turbo")
    palette = {label: cmap(i / max(1, len(labels) - 1)) for i, label in enumerate(labels)}
    for ax, model in zip(axes, models):
        selected = frame.loc[frame.model.eq(model)] if "model" in frame else frame.iloc[:0]
        if not len(selected) or field not in selected:
            _empty(ax, "NA: no eligible projection / label")
        else:
            text_labels = selected[field].fillna("NA").astype(str)
            for label in labels:
                group = selected.loc[text_labels.eq(label)]
                if len(group):
                    ax.scatter(group.x, group.y, color=palette[label], s=8, alpha=0.65,
                               rasterized=True, label=label)
            ax.set(xlabel="Projection 1", ylabel="Projection 2")
            ax.grid(alpha=0.1)
        ax.set_title(model)
    if labels and len(labels) <= 20:
        handles, texts = next((ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]), ([], []))
        fig.legend(handles, texts, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=min(4, len(labels)), fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, max(0.72, 0.94 - 0.025 * math.ceil(len(labels) / 4))))
    else:
        fig.suptitle(f"Colored by {field}; {len(labels)} categories (full legend omitted)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_similarity_heatmaps(embeddings, meta, model, max_points=160, seed=2026):
    meta = pd.DataFrame(meta).reset_index(drop=True)
    positions = _balanced_positions(meta, max_points, seed, ["kind", "perturbation", "batch_id"])
    selected = meta.iloc[positions].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), squeeze=False)
    if len(selected) < 2:
        for ax in axes.flat:
            _empty(ax, "NA: at least 2 wells required")
        return fig
    values = np.asarray(embeddings[model])
    if len(values) != len(meta):
        raise ValueError("Heatmap embedding/metadata row mismatch")
    vectors = _normalise(values[positions])
    similarity = np.clip(vectors @ vectors.T, -1, 1)
    np.fill_diagonal(similarity, np.nan)
    image = None
    for ax, fields, title in zip(axes.flat,
                                (["kind", "experiment_name", "batch_id", "perturbation", "well_id"],
                                 ["kind", "perturbation", "batch_id", "well_id"]),
                                ("Ordered by experiment / plate", "Ordered by treatment")):
        fields = [field for field in fields if field in selected]
        order = selected.sort_values(fields, kind="stable").index.to_numpy() if fields else np.arange(len(selected))
        image = ax.imshow(similarity[np.ix_(order, order)], vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set(title=title, xlabel="Selected wells", ylabel="Selected wells")
        ax.set_xticks([])
        ax.set_yticks([])
        boundary_field = "batch_id" if "plate" in title else "perturbation"
        if boundary_field in selected:
            labels = selected.iloc[order][boundary_field].astype(str).to_numpy()
            breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 0.5
            if len(breaks) <= 35:
                for boundary in breaks:
                    ax.axhline(boundary, lw=0.25, color="#333333", alpha=0.5)
                    ax.axvline(boundary, lw=0.25, color="#333333", alpha=0.5)
    fig.suptitle(f"{model} | identical {len(selected)} wells in both orderings", fontsize=11)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.25, right=0.85)
    colorbar_axis = fig.add_axes([0.90, 0.18, 0.016, 0.66])
    fig.colorbar(image, cax=colorbar_axis, label="Signed cosine; self-pairs hidden")
    return fig


def plot_neighbor_summary(neighbors, models):
    frame = pd.DataFrame(neighbors)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), squeeze=False)
    for ax, level in zip(axes.flat, ("plate", "experiment")):
        plotted = False
        for mi, model in enumerate(models):
            for ki, (kind, color) in enumerate(KIND_COLORS.items()):
                group = frame.loc[frame.model.eq(model) & frame.kind.eq(kind) & frame.batch_level.eq(level)] if {"model", "kind", "batch_level"} <= set(frame) else frame.iloc[:0]
                values = _finite(group, "enrichment")
                if len(values):
                    position = mi + (-0.17 if ki == 0 else 0.17)
                    ax.scatter(np.repeat(position, len(values)), values, color=color, s=22, alpha=0.6)
                    ax.scatter(position, np.mean(values), marker="D", color=color, s=45, edgecolor="black", linewidth=0.5)
                    plotted = True
        if not plotted:
            _empty(ax, "NA: eligible matched controls / multiple batches required")
        else:
            ax.axhline(1, color="gray", ls="--", label="Size-adjusted random expectation")
            ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
            for kind, color in KIND_COLORS.items():
                ax.scatter([], [], color=color, label=kind)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
        ax.set_title(level + " neighbor enrichment | controls only")
        ax.set_ylabel("Observed same-batch fraction / random expectation")
    fig.tight_layout()
    return fig
