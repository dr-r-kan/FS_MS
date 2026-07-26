"""Hierarchical Bayesian FS-versus-ES statewise transition analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_ORDER = ("A", "F", "C", "G", "B", "E")
PERIODS = {"final30s": (0, 30), "final5min": (0, 300)}


def inv_gamma(rng: np.random.Generator, shape, scale):
    return 1 / rng.gamma(shape, 1 / np.asarray(scale))


def prepare_events(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data.template_label.isin(STATE_ORDER) & data.semiology.isin(("hyper", "hypo"))].copy()
    data["is_functional"] = data.stratum.isin(("FS functional", "DD functional")).astype(int)
    data["is_hypo"] = data.semiology.eq("hypo").astype(int)
    data["is_dd"] = data.stratum.str.startswith("DD").astype(int)
    rows = []
    for period, (start, stop) in PERIODS.items():
        frame = data[data.window_end_to_ictus_s.between(start, stop, inclusive="left")]
        summary = (frame.groupby(["participant", "file_path", "stratum", "semiology", "template_label",
                                  "is_functional", "is_hypo", "is_dd"], observed=True)
                   .agg(js_from_II=("js_from_II", "mean"), log_kappa=("log_kappa", "mean"),
                        outgoing_transitions=("outgoing_transitions", "mean"), n_windows=("js_from_II", "size")).reset_index())
        summary["period"] = period
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    out["log_js"] = np.nan
    positive_js = np.isfinite(out.js_from_II) & out.js_from_II.gt(0)
    out.loc[positive_js, "log_js"] = np.log(out.loc[positive_js, "js_from_II"])
    return out


def model_arrays(frame: pd.DataFrame, outcome: str, group_column: str = "is_functional") -> dict:
    state = pd.Categorical(frame.template_label, categories=STATE_ORDER).codes
    if np.any(state < 0):
        raise ValueError("unknown state label")
    participant, participant_names = pd.factorize(frame.participant, sort=True)
    event, event_names = pd.factorize(frame.file_path, sort=True)
    n, s = len(frame), len(STATE_ORDER)
    nuisance = ("is_hypo", "is_dd") if group_column == "is_functional" else ("is_dd",)
    x = np.zeros((n, 2 * s + len(nuisance) + 1))
    x[np.arange(n), state] = 1
    x[np.arange(n), s + state] = frame[group_column].to_numpy(float)
    for column, name in enumerate(nuisance, 2 * s):
        x[:, column] = frame[name] - frame[name].mean()
    log_n = np.log(frame.outgoing_transitions.clip(lower=1))
    x[:, -1] = log_n - log_n.mean()
    y = frame[outcome].to_numpy(float)
    y_mean = y.mean(); y = y - y_mean
    return {"x": x, "y": y, "y_mean": y_mean, "state": state, "participant": participant,
            "event": event, "participant_names": participant_names, "event_names": event_names}


def one_chain(arrays: dict, iterations: int, burn: int, thin: int, seed: int,
              effect_name: str = "fs") -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x, y = arrays["x"], arrays["y"]
    state, participant, event = arrays["state"], arrays["participant"], arrays["event"]
    n_states, p = len(STATE_ORDER), x.shape[1]
    n_participants, n_events = participant.max() + 1, event.max() + 1
    coef = np.linalg.solve(x.T @ x + np.eye(p), x.T @ y)
    u, v = np.zeros(n_participants), np.zeros(n_events)
    residual = y - x @ coef
    sigma2 = np.array([np.var(residual[state == s]) + .05 for s in range(n_states)])
    participant_var = event_var = .1
    beta_mean, beta_var = float(coef[n_states:2 * n_states].mean()), .1
    names = ([f"global_{effect_name}_effect"] + [f"{effect_name}_effect_{label}" for label in STATE_ORDER] + [f"{effect_name}_effect_sd"] +
             ["participant_sd", "event_sd"] + [f"residual_sd_{label}" for label in STATE_ORDER])
    saved = []
    for iteration in range(iterations):
        weights = 1 / sigma2[state]
        nuisance_count = p - 2 * n_states
        prior_precision = np.r_[np.repeat(1 / 4, n_states), np.repeat(1 / beta_var, n_states), np.repeat(1 / 4, nuisance_count)]
        prior_mean = np.r_[np.zeros(n_states), np.repeat(beta_mean, n_states), np.zeros(nuisance_count)]
        precision = x.T @ (weights[:, None] * x) + np.diag(prior_precision)
        rhs = x.T @ (weights * (y - u[participant] - v[event])) + prior_precision * prior_mean
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(precision, rhs)
        coef = mean + np.linalg.solve(chol.T, rng.normal(size=p))

        without_u = y - x @ coef - v[event]
        group_precision = np.bincount(participant, weights=weights, minlength=n_participants) + 1 / participant_var
        group_mean = np.bincount(participant, weights=weights * without_u, minlength=n_participants) / group_precision
        u = group_mean + rng.normal(size=n_participants) / np.sqrt(group_precision)

        without_v = y - x @ coef - u[participant]
        group_precision = np.bincount(event, weights=weights, minlength=n_events) + 1 / event_var
        group_mean = np.bincount(event, weights=weights * without_v, minlength=n_events) / group_precision
        v = group_mean + rng.normal(size=n_events) / np.sqrt(group_precision)

        residual = y - x @ coef - u[participant] - v[event]
        counts = np.bincount(state, minlength=n_states)
        sums = np.bincount(state, weights=residual ** 2, minlength=n_states)
        sigma2 = inv_gamma(rng, 2 + counts / 2, .1 + sums / 2)
        participant_var = float(inv_gamma(rng, 2 + n_participants / 2, .1 + np.sum(u ** 2) / 2))
        event_var = float(inv_gamma(rng, 2 + n_events / 2, .1 + np.sum(v ** 2) / 2))

        beta = coef[n_states:2 * n_states]
        beta_mean_var = 1 / (n_states / beta_var + 1)
        beta_mean = float(rng.normal(beta_mean_var * beta.sum() / beta_var, np.sqrt(beta_mean_var)))
        beta_var = float(inv_gamma(rng, 2 + n_states / 2, .05 + np.sum((beta - beta_mean) ** 2) / 2))
        if iteration >= burn and (iteration - burn) % thin == 0:
            saved.append(np.r_[beta_mean, beta, np.sqrt(beta_var), np.sqrt(participant_var),
                               np.sqrt(event_var), np.sqrt(sigma2)])
    return np.asarray(saved), names


def split_rhat(chains: np.ndarray) -> np.ndarray:
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    n = split.shape[1]
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = n * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    m, n, p = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    variance = np.sum(centred ** 2, axis=1)
    out = np.empty(p)
    for column in range(p):
        rho_sum = 0.0
        previous = np.inf
        for lag in range(1, min(n - 1, 500), 2):
            pair = 0.0
            for this_lag in (lag, lag + 1):
                numerator = np.sum(centred[:, :-this_lag, column] * centred[:, this_lag:, column], axis=1)
                pair += np.mean(numerator / variance[:, column])
            if pair < 0 or pair > previous:
                break
            rho_sum += pair; previous = pair
        out[column] = m * n / max(1 + 2 * rho_sum, 1)
    return out


def fit_model(frame: pd.DataFrame, outcome: str, period: str, iterations: int, burn: int, thin: int,
              chains: int, seed: int, out_dir: Path, group_column: str = "is_functional",
              comparison: str = "FS versus ES", effect_name: str = "fs") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    eligible = (np.isfinite(frame[outcome]) & np.isfinite(frame.outgoing_transitions) &
                frame.outgoing_transitions.gt(0))
    frame = frame.loc[eligible].copy()
    if frame.empty or set(frame.template_label) != set(STATE_ORDER) or frame[group_column].nunique() != 2:
        raise ValueError(f"{comparison} {period} {outcome}: insufficient eligible data across all six states")
    arrays = model_arrays(frame, outcome, group_column)
    results, names = [], None
    for chain in range(chains):
        draws, names = one_chain(arrays, iterations, burn, thin, seed + 1009 * chain, effect_name)
        results.append(draws)
    draws = np.stack(results)
    rhat, ess = split_rhat(draws), effective_sample_size(draws)
    diagnostics = pd.DataFrame({"comparison": comparison, "period": period, "outcome": outcome,
                                "parameter": names, "rhat": rhat, "ess": ess,
                                "n_observations": len(frame), "n_participants": frame.participant.nunique(),
                                "n_events": frame.file_path.nunique()})
    flat = draws.reshape(-1, draws.shape[-1])
    rows = []
    for effect, column in [("global", 0)] + [(label, i + 1) for i, label in enumerate(STATE_ORDER)]:
        values = flat[:, column]
        low, median, high = np.quantile(values, [.025, .5, .975])
        rows.append({"comparison": comparison, "period": period, "outcome": outcome,
                     "effect": effect, "state": effect if effect != "global" else "all",
                     "posterior_median": median, "ci95_low": low, "ci95_high": high,
                     "ratio_median": np.exp(median), "ratio_ci95_low": np.exp(low), "ratio_ci95_high": np.exp(high),
                     "probability_gt_zero": np.mean(values > 0), "probability_lt_zero": np.mean(values < 0),
                     "rhat": rhat[column], "ess": ess[column], "n_observations": len(frame),
                     "n_participants": frame.participant.nunique(), "n_events": frame.file_path.nunique()})
    beta = flat[:, 1:1 + len(STATE_ORDER)]
    summary = pd.DataFrame(rows)
    summary["probability_all_states_gt_zero"] = np.nan
    summary.loc[summary.effect.eq("global"), "probability_all_states_gt_zero"] = np.mean(np.all(beta > 0, axis=1))
    np.savez_compressed(out_dir / f"posterior_{effect_name}_{period}_{outcome}.npz", draws=draws,
                        parameter_names=np.asarray(names), state_order=np.asarray(STATE_ORDER))
    return summary, diagnostics, draws


def forest_plot(summary: pd.DataFrame, output: Path, title: str, xlabel: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    order = ("global",) + STATE_ORDER
    for row, period in enumerate(PERIODS):
        for col, outcome in enumerate(("log_js", "log_kappa")):
            ax = axes[row, col]
            part = summary[(summary.period.eq(period)) & summary.outcome.eq(outcome)].set_index("effect").reindex(order)
            y = np.arange(len(part)); x = part.ratio_median.to_numpy(float)
            colors = ["black" if label == "global" else ("#D55E00" if label == "C" else "#0072B2") for label in order]
            for yi, xi, low, high, color in zip(y, x, part.ratio_ci95_low, part.ratio_ci95_high, colors):
                ax.errorbar(xi, yi, xerr=[[xi - low], [high - xi]], fmt="o", color=color, capsize=3)
            ax.axvline(1, color=".5", lw=1)
            ax.set(yticks=y, yticklabels=[f"{label}  P(>0)={prob:.3f}" for label, prob in zip(order, part.probability_gt_zero)],
                   xlabel=xlabel, title=f"{period}: {'JS divergence' if outcome == 'log_js' else 'transition kappa'}")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def c_posterior_plot(draws_by_model: dict, output: Path, comparison: str) -> None:
    c_column = 1 + STATE_ORDER.index("C")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, outcome, title in zip(axes, ("log_js", "log_kappa"), ("C-origin JS divergence", "C-origin transition kappa")):
        draws = draws_by_model[("final30s", outcome)][:, :, c_column].ravel()
        ax.hist(draws, bins=50, density=True, color="#D55E00", alpha=.75)
        ax.axvline(0, color="black", lw=1)
        low, median, high = np.quantile(draws, [.025, .5, .975])
        ax.set(xlabel=f"{comparison} posterior log-ratio", ylabel="Posterior density",
               title=f"{title}\nmedian={median:.3f}; 95% CrI {low:.3f} to {high:.3f}; P(>0)={np.mean(draws>0):.3f}")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for participant in range(36):
        functional = participant >= 18
        random_intercept = rng.normal(0, .15)
        for state in STATE_ORDER:
            rows.append({"participant": f"p{participant}", "file_path": f"e{participant}", "template_label": state,
                         "is_functional": int(functional), "is_hypo": participant % 2, "is_dd": participant % 3 == 0,
                         "outgoing_transitions": 150,
                         "test": random_intercept + .35 * functional + .25 * (participant % 2) + rng.normal(0, .2)})
    frame = pd.DataFrame(rows)
    arrays = model_arrays(frame, "test")
    draws = np.stack([one_chain(arrays, 900, 300, 2, 20 + chain)[0] for chain in range(2)])
    assert np.mean(draws[:, :, 0] > 0) > .9 and np.isfinite(split_rhat(draws)).all()
    fs_arrays = model_arrays(frame[frame.is_functional.eq(1)], "test", "is_hypo")
    fs_draws = np.stack([one_chain(fs_arrays, 900, 300, 2, 40 + chain, "hypo")[0] for chain in range(2)])
    assert np.mean(fs_draws[:, :, 0] > 0) > .8 and np.isfinite(split_rhat(fs_draws)).all()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None))


def run(args) -> dict:
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    events = prepare_events(args.statewise_windows)
    events.to_csv(data_dir / "bayesian_statewise_event_input.csv", index=False)
    summaries, diagnostics, draws_by_model, semio_draws = [], [], {}, {}
    for period in PERIODS:
        frame = events[events.period.eq(period)]
        for outcome in ("log_js", "log_kappa"):
            print(f"Bayesian model: {period} {outcome}", flush=True)
            summary, diagnostic, draws = fit_model(frame, outcome, period, args.iterations, args.burn,
                                                    args.thin, args.chains, args.seed, data_dir)
            summaries.append(summary); diagnostics.append(diagnostic); draws_by_model[(period, outcome)] = draws
            fs = frame[frame.is_functional.eq(1)]
            print(f"Bayesian FS semiology model: {period} {outcome}", flush=True)
            summary, diagnostic, draws = fit_model(
                fs, outcome, period, args.iterations, args.burn, args.thin, args.chains, args.seed + 7919,
                data_dir, group_column="is_hypo", comparison="Hypomotor versus hypermotor FS", effect_name="hypo")
            summaries.append(summary); diagnostics.append(diagnostic); semio_draws[(period, outcome)] = draws
    summary, diagnostic = pd.concat(summaries, ignore_index=True), pd.concat(diagnostics, ignore_index=True)
    if (not np.isfinite(diagnostic[["rhat", "ess"]].to_numpy(float)).all() or
            diagnostic.rhat.max() > 1.05 or diagnostic.ess.min() < 100):
        raise RuntimeError(f"MCMC diagnostic failure: max R-hat={diagnostic.rhat.max():.3f}, min ESS={diagnostic.ess.min():.1f}")
    summary.to_csv(data_dir / "bayesian_statewise_posterior_summary.csv", index=False)
    diagnostic.to_csv(data_dir / "bayesian_statewise_diagnostics.csv", index=False)
    fs_es = summary[summary.comparison.eq("FS versus ES")]
    semio = summary[summary.comparison.eq("Hypomotor versus hypermotor FS")]
    forest_plot(fs_es, figure_dir / "bayesian_statewise_fs_es_forest.png",
                "Hierarchical Bayesian statewise FS-versus-ES effects", "FS / ES ratio")
    forest_plot(semio, figure_dir / "bayesian_statewise_fs_semiology_forest.png",
                "Hierarchical Bayesian statewise hypomotor-versus-hypermotor FS effects", "Hypomotor / hypermotor ratio")
    c_posterior_plot(draws_by_model, figure_dir / "bayesian_c_origin_posterior.png", "FS minus ES")
    c_posterior_plot(semio_draws, figure_dir / "bayesian_c_origin_semiology_posterior.png", "Hypomotor minus hypermotor FS")
    primary = fs_es[fs_es.period.eq("final30s")]
    c_js = primary[(primary.outcome.eq("log_js")) & primary.effect.eq("C")].iloc[0]
    global_js = primary[(primary.outcome.eq("log_js")) & primary.effect.eq("global")].iloc[0]
    c_kappa = primary[(primary.outcome.eq("log_kappa")) & primary.effect.eq("C")].iloc[0]
    semio_primary = semio[semio.period.eq("final30s")]
    c_semio_js = semio_primary[(semio_primary.outcome.eq("log_js")) & semio_primary.effect.eq("C")].iloc[0]
    c_semio_kappa = semio_primary[(semio_primary.outcome.eq("log_kappa")) & semio_primary.effect.eq("C")].iloc[0]
    semio_five = semio[semio.period.eq("final5min")]
    a_five_js = semio_five[(semio_five.outcome.eq("log_js")) & semio_five.effect.eq("A")].iloc[0]
    b_five_js = semio_five[(semio_five.outcome.eq("log_js")) & semio_five.effect.eq("B")].iloc[0]
    c_five_js = semio_five[(semio_five.outcome.eq("log_js")) & semio_five.effect.eq("C")].iloc[0]
    global_five_kappa = semio_five[(semio_five.outcome.eq("log_kappa")) & semio_five.effect.eq("global")].iloc[0]
    report = ["# Hierarchical Bayesian statewise transition analysis", "",
              "The primary model uses one final-30-second summary per event and origin state. State-specific group effects are partially "
              "pooled around a common effect. The FS/ES model adjusts for semiology, DD cohort and log outgoing-transition count; the "
              "within-FS model adjusts for DD cohort and transition count. Both include participant and event random intercepts. The "
              "final-five-minute model is a sensitivity analysis.", "",
              "The likelihood is Gaussian on log(JS) or log(kappa). Weakly informative priors are normal for fixed effects and "
              "inverse-gamma for variance components; the six state-specific FS effects share an estimated global mean and variance. "
              "Rows with a non-finite outcome or no outgoing transition are excluded before fitting.", "",
              "## Primary findings", "",
              f"- Common all-state JS effect: FS/ES ratio {global_js.ratio_median:.3f} (95% CrI {global_js.ratio_ci95_low:.3f} to "
              f"{global_js.ratio_ci95_high:.3f}); P(FS>ES)={global_js.probability_gt_zero:.4f}.",
              f"- C-origin JS effect: FS/ES ratio {c_js.ratio_median:.3f} (95% CrI {c_js.ratio_ci95_low:.3f} to "
              f"{c_js.ratio_ci95_high:.3f}); P(FS>ES)={c_js.probability_gt_zero:.4f}.",
              f"- C-origin kappa effect: FS/ES ratio {c_kappa.ratio_median:.3f} (95% CrI {c_kappa.ratio_ci95_low:.3f} to "
              f"{c_kappa.ratio_ci95_high:.3f}); P(FS>ES)={c_kappa.probability_gt_zero:.4f}.", "",
              "## Within-FS semiology", "",
              f"- C-origin JS hypomotor/hypermotor ratio: {c_semio_js.ratio_median:.3f} (95% CrI "
              f"{c_semio_js.ratio_ci95_low:.3f} to {c_semio_js.ratio_ci95_high:.3f}); P(hypomotor>hypermotor)="
              f"{c_semio_js.probability_gt_zero:.4f}.",
              f"- C-origin kappa hypomotor/hypermotor ratio: {c_semio_kappa.ratio_median:.3f} (95% CrI "
              f"{c_semio_kappa.ratio_ci95_low:.3f} to {c_semio_kappa.ratio_ci95_high:.3f}); P(hypomotor>hypermotor)="
              f"{c_semio_kappa.probability_gt_zero:.4f}.",
              "- No final-30-second semiology contrast met the strong-evidence criterion.",
              f"- Final-five-minute A-origin JS was lower in hypomotor FS: ratio {a_five_js.ratio_median:.3f} "
              f"(95% CrI {a_five_js.ratio_ci95_low:.3f} to {a_five_js.ratio_ci95_high:.3f}); "
              f"P(hypomotor<hypermotor)={1-a_five_js.probability_gt_zero:.4f}.",
              f"- Final-five-minute B-origin JS was lower in hypomotor FS: ratio {b_five_js.ratio_median:.3f} "
              f"(95% CrI {b_five_js.ratio_ci95_low:.3f} to {b_five_js.ratio_ci95_high:.3f}); "
              f"P(hypomotor<hypermotor)={1-b_five_js.probability_gt_zero:.4f}.",
              f"- Final-five-minute C-origin JS showed the same direction but remained uncertain: ratio {c_five_js.ratio_median:.3f} "
              f"(95% CrI {c_five_js.ratio_ci95_low:.3f} to {c_five_js.ratio_ci95_high:.3f}); "
              f"P(hypomotor<hypermotor)={1-c_five_js.probability_gt_zero:.4f}.",
              f"- Final-five-minute global kappa was suggestively higher in hypomotor FS but not strong: ratio "
              f"{global_five_kappa.ratio_median:.3f} (95% CrI {global_five_kappa.ratio_ci95_low:.3f} to "
              f"{global_five_kappa.ratio_ci95_high:.3f}); P(hypomotor>hypermotor)={global_five_kappa.probability_gt_zero:.4f}.", "",
              "A 95% credible interval excluding a ratio of one and posterior direction probability >=0.975 is treated as strong evidence, "
              "not as a frequentist p-value.", "",
              markdown_table(summary, ["comparison", "period", "outcome", "effect", "ratio_median", "ratio_ci95_low", "ratio_ci95_high",
                                       "probability_gt_zero", "rhat", "ess"]), "",
              "![Bayesian forest](figures/bayesian_statewise_fs_es_forest.png)", "",
              "![C-origin posterior](figures/bayesian_c_origin_posterior.png)", "",
              "![FS semiology forest](figures/bayesian_statewise_fs_semiology_forest.png)", "",
              "![C-origin FS semiology posterior](figures/bayesian_c_origin_semiology_posterior.png)", "",
              "## Limitation", "",
              "The transition-count covariate reduces finite-sample JS confounding but is not a full parametric-bootstrap debiasing of JS. "
              "Accordingly, the model tests whether observed FS JS is higher than ES JS at comparable transition counts; it does not test "
              "whether either group has zero divergence from its estimated interictal operator."]
    (args.out_dir / "bayesian_statewise_findings.md").write_text("\n".join(report), encoding="utf-8")
    audit = ["Hierarchical Bayesian statewise transition audit", f"Completed UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Input: {args.statewise_windows}", f"States: {','.join(STATE_ORDER)}", "Primary period: final30s",
              "Sensitivity period: final5min", "Comparisons: FS versus ES; hypomotor versus hypermotor within FS",
              f"Chains: {args.chains}", f"Iterations per chain: {args.iterations}",
              f"Burn-in: {args.burn}", f"Thin: {args.thin}", f"Seed: {args.seed}",
              "Likelihood: Gaussian on log(JS) or log(kappa)",
              "Priors: state intercept SD=2; global group SD=1; nuisance SD=2; state group effects share estimated normal mean/variance",
              "Variance priors: inverse-gamma(2,0.1); state-group variance inverse-gamma(2,0.05)",
              f"Max split R-hat: {diagnostic.rhat.max():.6f}", f"Minimum ESS: {diagnostic.ess.min():.1f}",
             f"Event-state rows: {len(events)}", f"Participants: {events.participant.nunique()}", f"Events: {events.file_path.nunique()}"]
    (args.out_dir / "bayesian_statewise_audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    (data_dir / "analyse_bayesian_statewise.py").write_bytes(Path(__file__).read_bytes())
    result = {"participants": int(events.participant.nunique()), "events": int(events.file_path.nunique()),
              "event_state_rows": int(len(events)), "max_rhat": float(diagnostic.rhat.max()),
              "min_ess": float(diagnostic.ess.min()), "c_js_probability_gt_zero": float(c_js.probability_gt_zero),
              "c_js_ratio_median": float(c_js.ratio_median),
              "c_hypo_hyper_js_probability_gt_zero": float(c_semio_js.probability_gt_zero),
              "c_hypo_hyper_js_ratio_median": float(c_semio_js.ratio_median),
              "c_hypo_hyper_kappa_probability_gt_zero": float(c_semio_kappa.probability_gt_zero),
              "c_hypo_hyper_kappa_ratio_median": float(c_semio_kappa.ratio_median)}
    (data_dir / "bayesian_statewise_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statewise-windows", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--burn", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=26072026)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not args.statewise_windows or not args.out_dir:
        parser.error("--statewise-windows and --out-dir are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
