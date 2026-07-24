"""Participant-template C↔E corrected transitions and edge-wise inference."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from analyse_ce_group_condition import add_semiology, default_manifest


def aggregate_edges(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    edge = ["state_from_label", "state_to_label"]
    sums = (data.groupby([*keys, *edge], observed=True)[
        ["transition_count", "outgoing_transitions", "state_from_samples", "state_to_samples", "valid_samples"]
    ].sum().reset_index())
    sums["empirical_probability"] = sums["transition_count"] / sums["outgoing_transitions"].replace(0, np.nan)
    denominator = sums["valid_samples"] - sums["state_from_samples"]
    sums["expected_probability"] = sums["state_to_samples"] / denominator.replace(0, np.nan)
    sums["excess_probability"] = sums["empirical_probability"] - sums["expected_probability"]
    sums["corrected_transition_probability"] = sums["excess_probability"]
    return sums


def load_units(results_dir: Path, manifest: Path, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(results_dir / "participant_condition_transition_metrics.csv")
    data = add_semiology(data, manifest)
    data["group"] = data["group"].astype(str).str.strip().str.upper()
    data["condition"] = data["condition"].astype(str).str.strip().replace({"preict": "preictal", "postict": "postictal"})
    data["hypo_hyper"] = data["hypo_hyper"].astype("string").str.strip().str.lower()
    numeric = ["state_from_index", "state_to_index", "transition_count", "outgoing_transitions",
               "state_from_samples", "state_to_samples", "valid_samples"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data[data["backfit_method"].eq(method) & data["backfit_available"].eq(1)].copy()
    for column in ("state_from_label", "state_to_label"):
        data[column] = data[column].astype(str).str.strip().str.upper()
    data = data[data["state_from_label"].isin(("C", "E")) & data["state_to_label"].isin(("C", "E"))].copy()
    if data.empty:
        raise ValueError("No participant-template C→E or E→C transition rows were found")
    main = aggregate_edges(data, ["participant", "group", "condition"])
    preictal = aggregate_edges(
        data[data["condition"].eq("preictal") & data["hypo_hyper"].isin(("hypo", "hyper"))],
        ["participant", "group", "hypo_hyper"],
    )
    return main, preictal


def summaries(unit: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    edge = ["state_from_label", "state_to_label"]
    long = unit.melt(id_vars=["participant", *factors, *edge],
                     value_vars=["empirical_probability", "expected_probability", "corrected_transition_probability"],
                     var_name="matrix", value_name="value").dropna(subset=["value"])
    return (long.groupby([*factors, *edge, "matrix"], observed=True)["value"]
            .agg(n_participants="count", mean="mean", sd="std").reset_index())


def edge_models(unit: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge = ["state_from_label", "state_to_label"]
    preferred = {"group": "DD", "condition": "II", "hypo_hyper": "hypo"}
    references = {factor: (preferred.get(factor) if preferred.get(factor) in set(unit[factor]) else sorted(unit[factor].unique())[0])
                  for factor in factors}
    rhs = " * ".join(f"C({factor}, Treatment(reference={references[factor]!r}))" for factor in factors)
    terms, coefficients = [], []
    for key, frame in unit.dropna(subset=["corrected_transition_probability"]).groupby(edge, observed=True):
        prefix = dict(zip(edge, key)) | {"n_participants": frame["participant"].nunique(),
                                        "reference_levels": "; ".join(f"{k}={v}" for k, v in references.items())}
        if any(frame[factor].nunique() < 2 for factor in factors):
            continue
        try:
            fit = smf.ols(f"corrected_transition_probability ~ {rhs}", data=frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["participant"]})
            for term, result in fit.wald_test_terms(scalar=True).table.iterrows():
                terms.append({**prefix, "term": term, "chi2": np.asarray(result["statistic"]).item(),
                              "df": int(result["df_constraint"]), "p_value": float(result["pvalue"])})
            ci = fit.conf_int()
            for name in fit.params.index:
                coefficients.append({**prefix, "coefficient": name, "estimate": float(fit.params[name]),
                                     "se": float(fit.bse[name]), "ci_low": float(ci.loc[name, 0]),
                                     "ci_high": float(ci.loc[name, 1]), "p_value": float(fit.pvalues[name])})
        except Exception as exc:
            terms.append({**prefix, "term": "", "note": str(exc), "p_value": np.nan})
    terms, coefficients = pd.DataFrame(terms), pd.DataFrame(coefficients)
    for frame in (terms, coefficients):
        frame["p_fdr"] = np.nan
        if "p_value" not in frame:
            continue
        label = frame.get("term", frame.get("coefficient", pd.Series(index=frame.index, dtype=str)))
        valid = frame["p_value"].notna() & label.ne("Intercept")
        if valid.any():
            frame.loc[valid, "p_fdr"] = multipletests(frame.loc[valid, "p_value"], method="fdr_bh")[1]
    return terms, coefficients


def empirical_vs_expected(unit: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    edge = ["state_from_label", "state_to_label"]
    rows = []
    for key, frame in unit.dropna(subset=["corrected_transition_probability"]).groupby([*factors, *edge], observed=True):
        values = frame["corrected_transition_probability"].to_numpy(float)
        n, mean, sd = len(values), values.mean(), values.std(ddof=1)
        se = sd / np.sqrt(n) if n > 1 else np.nan
        statistic = mean / se if se > 0 else np.nan
        p_value = 2 * t.sf(abs(statistic), n - 1) if n > 2 and np.isfinite(statistic) else np.nan
        critical = t.ppf(.975, n - 1) if n > 1 else np.nan
        rows.append(dict(zip([*factors, *edge], key)) | {"n_participants": n, "mean_excess": mean,
                    "se": se, "ci_low": mean - critical * se, "ci_high": mean + critical * se, "p_value": p_value})
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    valid = out["p_value"].notna()
    if valid.any():
        out.loc[valid, "p_fdr"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
    return out


def plot_matrices(summary: pd.DataFrame, row_factor: str, column_factor: str, out_dir: Path, prefix: str) -> None:
    labels = [label for label in ("C", "E") if label in set(summary["state_from_label"])]
    rows, columns = sorted(summary[row_factor].astype(str).unique()), sorted(summary[column_factor].astype(str).unique())
    for matrix in ("empirical_probability", "expected_probability", "corrected_transition_probability"):
        panel = summary[summary["matrix"].eq(matrix)]
        limit = np.nanmax(np.abs(panel["mean"]))
        vmin, vmax, cmap = ((-limit, limit, "RdBu_r") if matrix == "corrected_transition_probability" else (0, limit, "viridis"))
        fig, axes = plt.subplots(len(rows), len(columns), squeeze=False,
                                 figsize=(3 * len(columns), 2.8 * len(rows)), facecolor="white")
        image = None
        for r, row_value in enumerate(rows):
            for c, column_value in enumerate(columns):
                values = panel[(panel[row_factor].astype(str) == row_value) &
                               (panel[column_factor].astype(str) == column_value)]
                pivot = values.pivot(index="state_from_label", columns="state_to_label", values="mean").reindex(index=labels, columns=labels)
                image = axes[r, c].imshow(pivot, vmin=vmin, vmax=vmax, cmap=cmap)
                axes[r, c].set_title(f"{row_value} | {column_value}")
                axes[r, c].set_xticks(range(len(labels)), labels)
                axes[r, c].set_yticks(range(len(labels)), labels)
                axes[r, c].set_xlabel("To")
                axes[r, c].set_ylabel("From")
        fig.colorbar(image, ax=axes, shrink=.75)
        fig.suptitle(matrix.replace("_", " ").title(), fontweight="bold")
        fig.subplots_adjust(top=.90, right=.90, wspace=.35, hspace=.4)
        fig.savefig(out_dir / f"{prefix}_{matrix}.png", dpi=300, facecolor="white")
        plt.close(fig)


def analyse(results_dir: Path, out_dir: Path, manifest: Path | None = None, method: str = "hard") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main, preictal = load_units(results_dir, manifest or default_manifest(results_dir), method)
    analyses = (("group_condition", main, ["group", "condition"]),
                ("preictal_group_semiology", preictal, ["group", "hypo_hyper"]))
    for name, unit, factors in analyses:
        summary = summaries(unit, factors)
        terms, coefficients = edge_models(unit, factors)
        deviations = empirical_vs_expected(unit, factors)
        unit.to_csv(out_dir / f"{name}_participant_edges.csv", index=False)
        summary.to_csv(out_dir / f"{name}_matrix_summary.csv", index=False)
        terms.to_csv(out_dir / f"{name}_edge_model_terms.csv", index=False)
        coefficients.to_csv(out_dir / f"{name}_edge_model_coefficients.csv", index=False)
        deviations.to_csv(out_dir / f"{name}_empirical_vs_expected.csv", index=False)
        terms[terms["p_fdr"].lt(.05)].to_csv(out_dir / f"{name}_significant_edge_terms.csv", index=False)
        coefficients[coefficients["p_fdr"].lt(.05)].to_csv(out_dir / f"{name}_significant_edge_coefficients.csv", index=False)
        plot_matrices(summary, factors[0], factors[1], out_dir, name)


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, rows, manifest_rows, rng = Path(tmp), [], [], np.random.default_rng(3)
        for i in range(16):
            participant, group = f"P{i:02d}", ("FS" if i < 8 else "ES")
            semiology = "hypo" if i % 2 == 0 else "hyper"
            for condition in ("II", "preict"):
                path = str(root / f"{participant}_{condition}.mat")
                manifest_rows.append({"file_path": path, "hypo_hyper": semiology})
                for origin in range(1, 4):
                    for destination in range(1, 4):
                        if origin == destination:
                            continue
                        count = (20 + i % 3 + (origin * destination) % 2 + rng.integers(-2, 3) +
                                 4 * (group == "ES") * (destination == 2) + 2 * (condition == "preict"))
                        rows.append({"participant": participant, "group": group, "condition": condition, "file_path": path,
                                     "backfit_method": "hard", "backfit_available": 1, "state_from_index": origin,
                                     "state_from_label": ("C", "E", "A")[origin-1], "state_to_index": destination,
                                     "state_to_label": ("C", "E", "A")[destination-1], "transition_count": count,
                                     "outgoing_transitions": 50, "state_from_samples": 300,
                                     "state_to_samples": 300, "valid_samples": 900})
        pd.DataFrame(rows).to_csv(root / "participant_condition_transition_metrics.csv", index=False)
        manifest = root / "vg_microstate_manifest_mat.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        analyse(root, root / "out", manifest)
        assert (root / "out" / "group_condition_empirical_probability.png").exists()
        assert (root / "out" / "preictal_group_semiology_edge_model_terms.csv").exists()
        edges = pd.read_csv(root / "out" / "group_condition_participant_edges.csv")
        assert set(zip(edges["state_from_label"], edges["state_to_label"])) == {("C", "E"), ("E", "C")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--method", default="hard")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    out_dir = args.out_dir or args.results_dir / "participant_ce_transition_analysis"
    analyse(args.results_dir, out_dir, args.manifest, args.method)
    print(f"Transition analysis: {out_dir}")


if __name__ == "__main__":
    main()
