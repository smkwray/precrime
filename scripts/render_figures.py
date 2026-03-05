from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "reports" / "plots"
FIG_DIR = REPO_ROOT / "docs" / "figures"


def _load_json(path: Path):
    import json

    return json.loads(path.read_text())


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_calibration_by_group(in_path: Path, out_path: Path, title: str) -> None:
    obj = _load_json(in_path)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#777", label="Ideal")

    # Two formats are supported:
    # (A) calibration_by_group output:
    #     {"GROUP": {"pred_mean":[...], "true_rate":[...], ...}, ...}
    # (B) plot_calibration spec:
    #     {"kind":"calibration", "traces":[{"name":"Group", "x":[...], "y":[...]} ...], ...}
    if isinstance(obj, dict) and "traces" in obj:
        traces = obj.get("traces", [])
        for trace in traces:
            name = str(trace.get("name", ""))
            x = np.array(trace.get("x", []), dtype=float)
            y = np.array(trace.get("y", []), dtype=float)
            if len(x) == 0 or len(y) == 0:
                continue
            ax.plot(x, y, marker="o", linewidth=1.6, label=name)
    else:
        for group in sorted(obj.keys()):
            series = obj[group]
            x = np.array(series["pred_mean"], dtype=float)
            y = np.array(series["true_rate"], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.6, label=str(group))

    ax.set_title(title)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed outcome rate (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out_path)


def render_calibration_by_race(in_path: Path, out_path: Path, title: str) -> None:
    # Backwards-compatible alias.
    render_calibration_by_group(in_path, out_path, title)


def render_bar_top_n(in_path: Path, out_path: Path, title: str, top_n: int = 10) -> None:
    obj = _load_json(in_path)
    # Two common formats:
    # (A) plot spec dict: {"x":[features], "y":[values], "title":..., ...}
    # (B) list of dicts: [{"feature":..., "mean_abs_shap":...}, ...]
    if isinstance(obj, dict) and "x" in obj and "y" in obj:
        feats = [str(x) for x in obj["x"]]
        vals = [float(y) for y in obj["y"]]
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and "feature" in obj[0]:
        value_key = "mean_abs_shap" if "mean_abs_shap" in obj[0] else "importance"
        feats = [str(r["feature"]) for r in obj]
        vals = [float(r.get(value_key, 0.0)) for r in obj]
    else:
        raise ValueError(f"Unknown bar format: {in_path}")

    pairs = list(zip(feats, vals))
    pairs = sorted(pairs, key=lambda t: t[1], reverse=True)[:top_n]
    feats_top = [p[0] for p in pairs][::-1]
    vals_top = [p[1] for p in pairs][::-1]

    def _humanize_count_bucket(text: str) -> str:
        text = text.replace("5 or more", "5+").replace("6 or more", "6+").replace("10 or more", "10+")
        return text

    def _humanize_feature(text: str) -> str:
        # NIJ one-hot patterns + a light general fallback.
        name = str(text)
        if name.endswith("__missing"):
            base = name[: -len("__missing")]
            return f"{_humanize_feature(base)} (missing)"

        if name.startswith("_v1_"):
            return f"Prior PP-violation arrests: {_humanize_count_bucket(name[len('_v1_'):])}"

        if name.startswith("Age_at_Release_"):
            bucket = name[len("Age_at_Release_") :]
            bucket = bucket.replace("48 or older", "48+")
            return f"Age at release: {bucket}"

        if name.startswith("Gang_Affiliated_"):
            return f"Gang affiliated: {name[len('Gang_Affiliated_'):]}"

        if name == "Supervision_Risk_Score_First":
            return "Supervision risk score (first)"

        if name.startswith("Prior_Arrest_Episodes_Property_"):
            return f"Prior property arrests: {_humanize_count_bucket(name[len('Prior_Arrest_Episodes_Property_'):])}"

        if name.startswith("Prior_Arrest_Episodes_Felony_"):
            return f"Prior felony arrests: {_humanize_count_bucket(name[len('Prior_Arrest_Episodes_Felony_'):])}"

        if name.startswith("Prior_Arrest_Episodes_Misd_"):
            return f"Prior misdemeanor arrests: {_humanize_count_bucket(name[len('Prior_Arrest_Episodes_Misd_'):])}"

        if name.startswith("Prison_Years_"):
            bucket = name[len("Prison_Years_") :]
            bucket = bucket.replace("Less than 1 year", "<1 year")
            bucket = bucket.replace("More than 3 years", "3+ years")
            bucket = bucket.replace("Greater than 2 to 3 years", "2–3 years")
            bucket = bucket.replace("1-2 years", "1–2 years")
            return f"Prison time: {bucket}"

        if name.startswith("Condition_MH_SA_"):
            return f"MH/SA condition: {name[len('Condition_MH_SA_'):]}"

        # Fallback: de-underscore common patterns.
        return _humanize_count_bucket(name.replace("_", " "))

    feats_top = [_humanize_feature(f) for f in feats_top]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh(feats_top, vals_top, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP| (top features)" if "shap" in in_path.name else "Importance (top features)")
    _save(fig, out_path)


def render_threshold_gap(in_path: Path, out_path: Path, title: str) -> None:
    obj = _load_json(in_path)
    thr = np.array(obj["thresholds"], dtype=float)
    fpr = np.array(obj["fpr_gap_per_threshold"], dtype=float)
    fnr = np.array(obj["fnr_gap_per_threshold"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(thr, fpr, label="FPR gap (max-min)", linewidth=1.8)
    ax.plot(thr, fnr, label="FNR gap (max-min)", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Gap across groups")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(np.nanmax(fpr), np.nanmax(fnr)) * 1.05 if len(thr) else 1.0)
    ax.legend(frameon=False)
    _save(fig, out_path)


def render_policy_tradeoffs(in_path: Path, out_path: Path, title: str) -> None:
    obj = _load_json(in_path)
    x = np.array(obj.get("x", []), dtype=float)
    metrics = obj.get("metrics", {})
    if len(x) == 0 or not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Missing x/metrics in policy-curve spec: {in_path}")

    # Prefer percent selection rates for readability.
    x_pct = x * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), sharex=True)
    panels = [
        ("FPR", "fpr"),
        ("FNR", "fnr"),
        ("PPV", "ppv"),
    ]
    colors = {
        "overall": "#1f77b4",
        "black": "#d62728",
        "white": "#2ca02c",
    }

    for ax, (label, key) in zip(axes, panels):
        series = metrics.get(key, {})
        if not isinstance(series, dict):
            continue

        for group in ("overall", "black", "white"):
            y = np.array(series.get(group, []), dtype=float)
            if len(y) != len(x):
                continue
            ax.plot(
                x_pct,
                y,
                label=group.title(),
                linewidth=2.2 if group == "overall" else 1.8,
                color=colors.get(group),
            )

        ax.set_title(label)
        ax.set_xlabel("Selection rate (top-k%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xlim(float(np.nanmin(x_pct)), float(np.nanmax(x_pct)))

        # Avoid a flat line when rates are very small; clamp at 0 with a small headroom.
        y_all = []
        for group_vals in series.values():
            try:
                y_all.extend([float(v) for v in group_vals])
            except TypeError:
                continue
        if y_all:
            y_max = float(np.nanmax(np.array(y_all, dtype=float)))
            if not np.isnan(y_max) and y_max > 0:
                ax.set_ylim(0.0, min(1.0, y_max * 1.15))

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(title, y=1.04, fontsize=12)
    _save(fig, out_path)


def render_model_sweep_tradeoff(in_path: Path, out_path: Path) -> None:
    obj = _load_json(in_path)
    panels = obj.get("panels", [])
    if not isinstance(panels, list) or not panels:
        raise ValueError(f"Missing panels in tradeoff spec: {in_path}")

    title = str(obj.get("title", "Model sweep tradeoff"))
    subtitle = str(obj.get("subtitle", "")).strip()
    x_label = str(obj.get("x_label", "Brier (sex-avg)"))
    y_label = str(obj.get("y_label", "FairAcc (sex-avg)"))

    horizons = sorted({str(p.get("horizon", "")) for p in panels if p.get("horizon")}, key=lambda s: (len(s), s))
    variants = sorted({str(p.get("variant", "")) for p in panels if p.get("variant")})
    if not horizons or not variants:
        raise ValueError(f"Unable to infer horizons/variants from panels: {in_path}")

    # Build a consistent color map across all panels.
    all_models = sorted(
        {str(pt.get("model", "")) for p in panels for pt in (p.get("points", []) or []) if pt.get("model")}
    )
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_for = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    nrows = len(variants)
    ncols = len(horizons)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), squeeze=False)

    def _panel_lookup(h: str, v: str) -> dict | None:
        for p in panels:
            if str(p.get("horizon", "")) == h and str(p.get("variant", "")) == v:
                return p
        return None

    for r, variant in enumerate(variants):
        for c, horizon in enumerate(horizons):
            ax = axes[r][c]
            panel = _panel_lookup(horizon, variant)
            if not panel:
                ax.axis("off")
                continue

            points = panel.get("points", []) or []
            for pt in points:
                model = str(pt.get("model", ""))
                cal = str(pt.get("calibration", ""))
                x = float(pt.get("x", np.nan))
                y = float(pt.get("y", np.nan))
                if np.isnan(x) or np.isnan(y):
                    continue

                is_best_pred = bool(pt.get("best_prediction", False))
                is_best_fair = bool(pt.get("best_fairacc", False))

                marker = "o"
                size = 42
                edge = "none"
                lw = 0.0
                if is_best_pred:
                    marker = "*"
                    size = 140
                    edge = "#111"
                    lw = 0.6
                elif is_best_fair:
                    marker = "D"
                    size = 80
                    edge = "#111"
                    lw = 0.5

                ax.scatter(
                    [x],
                    [y],
                    s=size,
                    marker=marker,
                    color=color_for.get(model, "#1f77b4"),
                    edgecolors=edge,
                    linewidths=lw,
                    alpha=0.92,
                )

                # Light annotation for "best" points only.
                if is_best_pred or is_best_fair:
                    label = f"{model} ({cal})"
                    ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)

            ax.set_title(f"{horizon.upper()} — {variant}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)

    # Legend (models) in the upper-right of the first panel.
    handles = []
    labels = []
    for m in all_models:
        h = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_for[m], markersize=7, linewidth=0)
        handles.append(h)
        labels.append(m)
    if handles:
        axes[0][0].legend(handles, labels, frameon=False, loc="lower right", fontsize=8, title="Model")

    fig.suptitle(title, y=1.02, fontsize=12)
    if subtitle:
        fig.text(0.5, 0.99, subtitle, ha="center", va="top", fontsize=9, color="#444")
    _save(fig, out_path)


def _parse_stability_md(path: Path):
    text = path.read_text()
    # Blocks like: "## STATIC Y1 ..." then a markdown table with Metric|Mean|Std
    horizon = None
    rows = []
    for line in text.splitlines():
        m = re.match(r"^##\s+([A-Z]+)\s+(Y[123])", line.strip())
        if m:
            horizon = f"{m.group(1)} {m.group(2)}"
            continue
        if horizon and line.startswith("| Brier |"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            rows.append((horizon, "Brier", float(parts[1]), float(parts[2])))
        if horizon and line.startswith("| AUROC |"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            rows.append((horizon, "AUROC", float(parts[1]), float(parts[2])))
    return rows


def render_stability(in_path: Path, out_path: Path, title: str) -> None:
    rows = _parse_stability_md(in_path)
    if not rows:
        raise ValueError(f"Unable to parse stability markdown: {in_path}")

    horizons = sorted({r[0] for r in rows}, key=lambda s: (s.split()[-1], s.split()[0]))
    metrics = ["Brier", "AUROC"]

    data = {(h, m): (mean, std) for (h, m, mean, std) in rows for mean, std in [(mean, std)]}
    x = np.arange(len(horizons))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, metric in enumerate(metrics):
        means = [data[(h, metric)][0] for h in horizons]
        stds = [data[(h, metric)][1] for h in horizons]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=metric)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, rotation=20, ha="right")
    ax.legend(frameon=False)
    _save(fig, out_path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    render_model_sweep_tradeoff(
        PLOTS_DIR / "model_sweep_tradeoff.json",
        FIG_DIR / "model_sweep_tradeoff.png",
    )
    render_policy_tradeoffs(
        PLOTS_DIR / "nij_y1_policy_curves.json",
        FIG_DIR / "nij_y1_policy_tradeoffs.png",
        "NIJ Y1 (Static) — Policy Tradeoffs by Selection Budget",
    )
    render_calibration_by_group(
        PLOTS_DIR / "fairness_y1_static_race_calibration.json",
        FIG_DIR / "nij_y1_static_calibration_by_race.png",
        "NIJ Y1 (Static) — Calibration by Race (XGBoost best model)",
    )
    render_calibration_by_group(
        PLOTS_DIR / "fairness_y1_static_gender_calibration.json",
        FIG_DIR / "nij_y1_static_calibration_by_gender.png",
        "NIJ Y1 (Static) — Calibration by Gender (XGBoost best model)",
    )
    render_bar_top_n(
        PLOTS_DIR / "xgb_shap_static_y1.json",
        FIG_DIR / "nij_y1_static_shap_top10.png",
        "NIJ Y1 (Static) — Top Factors (SHAP, top 10)",
        top_n=10,
    )
    render_threshold_gap(
        PLOTS_DIR / "fairness_y1_static_race_threshold_gap.json",
        FIG_DIR / "nij_y1_static_race_gap_vs_threshold.png",
        "NIJ Y1 (Static) — Race Gap vs Threshold (FPR/FNR)",
    )
    render_threshold_gap(
        PLOTS_DIR / "fairness_y1_static_gender_threshold_gap.json",
        FIG_DIR / "nij_y1_static_gender_gap_vs_threshold.png",
        "NIJ Y1 (Static) — Gender Gap vs Threshold (FPR/FNR)",
    )
    render_stability(
        REPO_ROOT / "reports" / "stability_eval.md",
        FIG_DIR / "nij_stability_brier_auroc.png",
        "NIJ Stability (Seeds 42/43/44) — Brier & AUROC",
    )

    # COMPAS (ProPublica) benchmark visuals.
    render_calibration_by_group(
        PLOTS_DIR / "calibration_compas_xgboost_platt_race.json",
        FIG_DIR / "compas_xgb_calibration_by_race.png",
        "COMPAS (XGBoost, Platt) — Calibration by Race",
    )
    render_calibration_by_group(
        PLOTS_DIR / "calibration_compas_xgboost_platt_sex.json",
        FIG_DIR / "compas_xgb_calibration_by_sex.png",
        "COMPAS (XGBoost, Platt) — Calibration by Sex",
    )
    render_bar_top_n(
        PLOTS_DIR / "xgb_shap_compas.json",
        FIG_DIR / "compas_xgb_shap_top10.png",
        "COMPAS — Top Factors (SHAP, top 10)",
        top_n=10,
    )

    # NCRP (ICPSR 37973) term-record benchmark visuals (optional; only if present).
    ncrp_cal = PLOTS_DIR / "calibration_ncrp37973_y1_xgboost_platt_race.json"
    if ncrp_cal.exists():
        render_calibration_by_group(
            ncrp_cal,
            FIG_DIR / "ncrp37973_y1_xgb_calibration_by_race.png",
            "NCRP 37973 (Return-to-prison, Y1) — Calibration by Race (XGBoost, Platt)",
        )

    ncrp_shap = PLOTS_DIR / "xgb_shap_ncrp37973_y1.json"
    if ncrp_shap.exists():
        render_bar_top_n(
            ncrp_shap,
            FIG_DIR / "ncrp37973_y1_xgb_shap_top10.png",
            "NCRP 37973 (Return-to-prison, Y1) — Top Factors (SHAP, top 10)",
            top_n=10,
        )

    print(f"wrote figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
