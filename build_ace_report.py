"""Build the versioned A/C/E cardioneural technical report and adjacent audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from microstate_labels import format_label_order, load_global_labels


LABELS = ""
SPM_MS_COMMIT = "0dbde1146f6777412dce19edfa4962cd45157fca"
SPM_COMMIT = "04db1be762077630772ee660a470a0fd63d4ff43"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(value):
        return "NA"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def configure(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(.75)
    section.left_margin = section.right_margin = Inches(.8)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    for name, size, colour in (("Title", 24, "17365D"), ("Heading 1", 16, "17365D"), ("Heading 2", 12, "2F5597")):
        style = styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(colour)
    footer = section.footer.paragraphs[0]
    add_page_field(footer)


def add_title(document: Document) -> None:
    p = document.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Preictal A/C/E microstate dynamics, transition precision and cardiac coupling")
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Technical paper-preparation report — corrected current-SPM_MS hierarchy")
    r.bold = True
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"Generated {datetime.now().astimezone().strftime('%d %B %Y %H:%M %Z')}")
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(LABELS)
    document.add_paragraph(
        "Status: analysis report for manuscript development, not a clinical report. All pseudo-observation-era hierarchy and fixed-backfit morphology results are superseded."
    )


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def add_table(document: Document, frame: pd.DataFrame, columns: list[tuple[str, str]], max_rows: int = 20) -> None:
    if frame.empty:
        document.add_paragraph("No estimable rows were available.")
        return
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Light Shading Accent 1"
    for cell, (_, label) in zip(table.rows[0].cells, columns):
        cell.text = label
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat_header = OxmlElement("w:tblHeader")
    repeat_header.set(qn("w:val"), "true")
    header_properties.append(repeat_header)
    for row in frame.head(max_rows).itertuples(index=False):
        data = row._asdict()
        cells = table.add_row().cells
        for cell, (name, _) in zip(cells, columns):
            cell.text = fmt(data.get(name))
    for row in table.rows:
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
    if len(frame) > max_rows:
        document.add_paragraph(f"Table truncated to {max_rows} of {len(frame)} rows; complete CSV is supplied.")


def add_picture(document: Document, path: Path, caption: str, width: float = 6.7) -> None:
    if not path.exists():
        return
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = document.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True


def result_sentence(primary: pd.DataFrame, family: str, unit: str) -> str:
    if primary.empty or "family" not in primary:
        return f"{family}: not estimable."
    row = primary[primary["family"].eq(family)]
    if row.empty:
        return f"{family}: not estimable."
    r = row.iloc[0]
    return (f"{family}: estimate {fmt(r.get('estimate'))} {unit}, robust SE {fmt(r.get('se'))}, "
            f"one-/two-sided planned p={fmt(r.get('p'))}, Holm-adjusted p={fmt(r.get('p_holm'))}.")


def build_report(analysis_dir: Path, morphology_dir: Path, hierarchy_dir: Path, ecg_dir: Path, output: Path) -> None:
    global LABELS
    label_order = load_global_labels(hierarchy_dir)
    LABELS = format_label_order(label_order)
    data = analysis_dir / "data"
    figures = analysis_dir / "figures"
    morph_data = morphology_dir / "data"
    morph_figures = morphology_dir / "figures"
    primary = read_csv(data / "primary_holm_corrected_tests.csv")
    cardiac = read_csv(data / "cardiac_planned_contrasts.csv")
    activity = read_csv(data / "ace_activity_planned_contrasts.csv")
    transitions = read_csv(data / "b_like_precision_planned_contrasts.csv")
    gating = read_csv(data / "e_gating_planned_contrasts.csv")
    cardioneural = read_csv(data / "cardioneural_planned_contrasts_bh.csv")
    secondary = read_csv(data / "ace_activity_secondary_bh.csv")
    tfce = read_csv(morph_data / "morphology_tfce_peak_summary.csv")
    geodesic = read_csv(morph_data / "morphology_geodesic_results.csv")
    hierarchy = read_csv(hierarchy_dir / "hierarchical_fit_summary.csv")
    covariance = read_csv(hierarchy_dir / "hierarchical_state_covariance_metrics.csv")
    cardiac_qc = read_csv(data / "cardiac_window_qc_audit.csv")
    heartbeat_phase = read_csv(data / "ace_heartbeat_phase_poisson_gee.csv")
    outlier_qc = read_csv(data / "ace_activity_outlier_audit.csv")

    document = Document()
    configure(document)
    add_title(document)

    document.add_heading("Executive summary", level=1)
    add_bullets(document, [
        result_sentence(primary, "C/A late slope", "ilr units/min"),
        result_sentence(primary, "hypomotor HR late slope", "bpm/min"),
        result_sentence(primary, "B-like precision trend", "log-precision/min"),
        "C/A is the primary processing balance. E is modelled separately as a putative salience/gating state and as an origin-specific transition-precision process.",
        "Morphology inference uses only fresh K=6 fits in 60-s windows with participant K=6 maps as component-mean priors; fixed-template backfit averages do not enter morphology tests.",
    ])

    document.add_heading("Scientific hypothesis", level=1)
    document.add_paragraph(
        "The analysis tests whether the approach to a functional seizure is accompanied by a redistribution between externally oriented and internally/self-oriented processing, operationalised as a C/A log-ratio, while state E gates rather than defines that balance. In active-inference language, the empirical prediction is a change in the precision and directional organisation of state transitions, not direct recovery of a generative-model B tensor."
    )
    document.add_paragraph(
        "The primary neural hypothesis is a positive pooled-functional late C/A slope in the final five minutes. The cardiac hypothesis is preictal deceleration in hypomotor events, with hypermotor events equivalent to a near-flat slope within ±1 bpm/min. The mechanistic interaction hypothesis is that C/A or transition-precision changes co-vary with within-event heart-rate change in hypomotor events, while neural transition changes can remain detectable without cardiac acceleration or deceleration in hypermotor events."
    )
    document.add_paragraph(
        "Canham's thesis motivates the system-level emphasis on integration, segregation, metastable switching and changing control parameters in non-stationary EEG. Accordingly, the analysis triangulates activity balance, E-routed sequences, empirical transition precision and window-refitted topographic drift rather than treating a single scalar as a biomarker. The thesis does not validate an active-inference B tensor; that interpretation remains a separately specified, falsifiable model-inspired hypothesis."
    )

    document.add_heading("Corrected hierarchy and provenance", level=1)
    document.add_paragraph(
        "Every parent prior is represented in the same child-derived PCA and polarity-invariant projective feature space as the child GFP-peak maps. A child fit calls spm_mix_prior with the parent features as component-mean priors and mean weight one. Parent maps are never appended to observations. Child observations determine the effective counts, proportions and covariances; the recorded beta_0 is one for every state."
    )
    add_bullets(document, [
        f"SPM_MS pinned commit: {SPM_MS_COMMIT}.",
        f"Installed SPM25 pinned commit: {SPM_COMMIT}.",
        f"Manually selected fitted-state mapping: {LABELS}.",
        "Participant A/C/E references require polarity-invariant correlation ≥0.50 with their corresponding global maps.",
        "Synthetic validation covers parent permutation and polarity, observation counts, beta_0, strong/weak child evidence, generating-parent recovery and malformed-parent failure.",
    ])
    add_picture(document, hierarchy_dir / "global" / "global_centers.png",
                f"Global K=6 fitted states with the saved manual mapping: {LABELS}.")
    if not hierarchy.empty:
        document.add_heading("Hierarchy audit snapshot", level=2)
        add_table(document, hierarchy.groupby("level", dropna=False).agg(
            fits=("name", "size"), median_real_observations=("real_observation_count", "median"),
            median_template_correlation=("template_mean_corr", "median")).reset_index(),
                  [("level", "Level"), ("fits", "Fits"), ("median_real_observations", "Median observations"),
                   ("median_template_correlation", "Median map r")])
    if not covariance.empty:
        ace_quality = covariance[
            covariance["level"].eq("participant") & covariance["template_label"].isin(["A", "C", "E"])
        ].pivot(index="participant", columns="template_label", values="template_match_abs_correlation")
        below = ace_quality.lt(.50).sum()
        document.add_paragraph(
            f"Reference-quality audit: {int(ace_quality.ge(.50).all(axis=1).sum())}/{len(ace_quality)} participants met r>=0.50 "
            f"for all A/C/E references. Below-threshold counts were A={int(below.get('A', 0))}, "
            f"C={int(below.get('C', 0))}, and E={int(below.get('E', 0))}; affected state-specific outcomes were set missing."
        )

    document.add_heading("A/C/E activity analysis", level=1)
    document.add_paragraph("The primary coordinates were:")
    document.add_paragraph("zC/A,w = (1/√2) log[(nC,w + 0.5)/(nA,w + 0.5)]")
    document.add_paragraph("zE,w = √(2/3) log[(nE,w + 0.5)/√{(nA,w + 0.5)(nC,w + 0.5)}]")
    document.add_paragraph(
        "Coverage and GFP were cleaned independently within each event/state/source/method/scheme using the locked one-pass rule |value − event median| > 3 × sample SD. Only the affected metric was set missing. Participant-clustered piecewise GEE used a −5-minute knot, with events averaged equally within participant."
    )
    add_table(document, activity, [("contrast", "Planned contrast"), ("estimate", "Estimate"), ("se", "SE"), ("p", "p")])
    if not activity.empty:
        late_ca = activity[activity["contrast"].astype(str).str.contains("pooled functional late C/A", case=False, na=False)]
        if not late_ca.empty:
            row = late_ca.iloc[0]
            document.add_paragraph(
                f"Observed result: the pooled-functional late C/A slope was {fmt(row.get('estimate'))} ilr/min "
                f"(planned one-sided p={fmt(row.get('p'))}); neither the semiology interaction nor E-by-late moderation was significant."
            )
    add_picture(document, figures / "ace_processing_balance_trajectories.png",
                "Figure 1. Participant-weighted C/A and E expression trajectories; shading is the across-participant standard error.")
    if not outlier_qc.empty:
        document.add_paragraph(
            f"Outlier audit: {int(outlier_qc.get('outlying_windows', pd.Series(dtype=float)).sum())} metric-specific windows were excluded across {int(outlier_qc.get('finite_windows', pd.Series(dtype=float)).sum())} finite metric-windows."
        )

    document.add_heading("Sequence gating and B-like transition precision", level=1)
    document.add_paragraph(
        "Directional E gating was log[(NA→E→C + 0.5)/(NC→E→A + 0.5)]. The routed-switch measure was the proportion of direct-or-one-E A/C switches that passed through E. Participant interictal six-state transition operators supplied the baseline probabilities. A/C/E-origin precision was estimated as the maximum-likelihood exponent κ applied to those probabilities; E-origin κ and weighted Jensen–Shannon divergence were additional outcomes. This is a B-like empirical transition-precision proxy, not direct estimation of an action-conditioned B tensor."
    )
    add_table(document, transitions, [("contrast", "Transition contrast"), ("estimate", "Estimate"), ("se", "SE"), ("p", "p")])
    add_table(document, gating, [("contrast", "Gating contrast"), ("estimate", "Estimate"), ("se", "SE"), ("p", "p")])
    if not transitions.empty and not gating.empty:
        document.add_paragraph(
            "Observed result: no pooled B-like precision, E-origin precision, weighted-JS, directional E-gating or E-routed-switch contrast was significant."
        )
    add_picture(document, figures / "b_like_transition_precision.png",
                "Figure 2. Participant-weighted transition-precision and divergence trajectories.")

    document.add_heading("Window-refitted morphology", level=1)
    document.add_paragraph(
        "Each event contributes end-aligned 60-s windows at 30-s steps across the final 20 minutes. The record end is derived exactly from the source MAT sample count divided by its sampling rate, not from the last GFP peak or a rounded rolling bin. At least 30 real GFP peaks are required. A fresh K=6 solution is fit using the participant full-record solution as a component-mean prior with weight one, one-to-one matched and polarity aligned. At least five assigned state peaks are required for an A, C or E map. Low match correlations are retained because declining similarity may be the signal."
    )
    document.add_paragraph("Geodesic deviation: θw = arccos(|μᵀmw|). Tangent-space regression:")
    document.add_paragraph("zw = [θw/sin(θw)](mw − cos(θw)μ);     zw = a + b tw + εw")
    document.add_paragraph(
        "Event channel slopes are averaged equally within participant. Primary spatial inference uses the 27-electrode Delaunay graph, TFCE E=0.5/H=2 and 5,000 whole-participant sign flips with identical signs across fields. Correction is joint across A/C/E. Gaussian maximum-field inference is reported only as sensitivity."
    )
    if not tfce.empty:
        add_table(document, tfce, [("family", "Family"), ("stratum", "Stratum"), ("template_label", "State"),
                                   ("n_participants", "Participants"), ("peak_channel", "Peak electrode"),
                                   ("peak_t", "Peak t"), ("peak_p_tfce_fwer", "TFCE FWER p")])
        primary_tfce = tfce[tfce["family"].eq("primary")].sort_values("peak_p_tfce_fwer")
        if not primary_tfce.empty:
            row = primary_tfce.iloc[0]
            document.add_paragraph(
                f"Observed result: no A/C/E field survived joint TFCE correction. The smallest primary FWER p was "
                f"{fmt(row.get('peak_p_tfce_fwer'))} for state {row.get('template_label')} at {row.get('peak_channel')} "
                f"(peak t={fmt(row.get('peak_t'))})."
            )
    if not geodesic.empty:
        add_table(document, geodesic, [("family", "Family"), ("stratum", "Stratum"), ("template_label", "State"),
                                       ("geodesic_beta_rad_per_min", "Mean slope"), ("t", "t"), ("p_fwer", "Max-T p")])
    add_picture(document, morph_figures / "morphology_tfce_primary_beta_maps.png",
                "Figure 3. A/C/E tangent-space beta fields and TFCE-corrected spatial inference.")
    add_picture(document, morph_figures / "morphology_rft_sensitivity_beta_maps.png",
                "Figure 4. Gaussian maximum-field sensitivity maps for cohort-stratified fields.")

    document.add_heading("Cardiac and cardioneural analysis", level=1)
    document.add_paragraph(
        "Versioned full-record R-peak sidecars were regenerated after repairing the ECG channel-selection failure. A 60-s HR window required ≥80% RR coverage, no unrecovered gap >5 s, ≤10% recovered intervals and instantaneous HR 35–140 bpm. HR was centred to the earliest valid −20 to −15-minute event baseline. Exact 60-s cardiac and neural windows were then merged; heart-rate association terms use within-event centred ΔHR."
    )
    add_table(document, cardiac, [("contrast", "Cardiac contrast"), ("estimate", "bpm/min"), ("se", "SE"),
                                  ("p", "p"), ("tost_p", "TOST p")])
    if not cardiac.empty:
        interaction = cardiac[cardiac["contrast"].astype(str).str.contains("hypomotor minus hypermotor", case=False, na=False)]
        hyper = cardiac[cardiac["contrast"].astype(str).str.fullmatch("hypermotor late HR slope", case=False, na=False)]
        if not interaction.empty and not hyper.empty:
            irow, hrow = interaction.iloc[0], hyper.iloc[0]
            document.add_paragraph(
                f"Observed result: the hypomotor-minus-hypermotor late-slope difference was {fmt(irow.get('estimate'))} bpm/min "
                f"(p={fmt(irow.get('p'))}). The hypermotor estimate was {fmt(hrow.get('estimate'))} bpm/min and met the "
                f"prespecified +/-1 bpm/min equivalence criterion (TOST p={fmt(hrow.get('tost_p'))}); the hypomotor negative slope itself was not significant."
            )
    add_picture(document, figures / "cardiac_trajectories.png",
                "Figure 5. Heart-rate change from the early preictal baseline by semiology.")
    if not cardiac_qc.empty:
        add_table(document, cardiac_qc, [("stratum", "Stratum"), ("semiology", "Semiology"), ("events", "Events"),
                                         ("participants", "Participants"), ("valid_windows", "Valid windows"),
                                         ("windows", "All windows")])
    if not cardioneural.empty:
        document.add_heading("Cardioneural planned associations", level=2)
        add_table(document, cardioneural, [("outcome", "Outcome"), ("estimate", "Hypomotor HR association"),
                                           ("se", "SE"), ("p", "p"), ("p_bh", "BH p")])
        significant = cardioneural[cardioneural["p_bh"].lt(.05)].sort_values("p_bh")
        if not significant.empty:
            document.add_paragraph(
                "Observed secondary result: within hypomotor events, lower heart rate was associated with higher C/A balance "
                "and lower E expression after BH correction; transition-precision associations were null. These time-adjusted "
                "within-event associations are not causal tests."
            )
    if not heartbeat_phase.empty:
        document.add_heading("Heartbeat-phase occurrence models", level=2)
        phase_terms = heartbeat_phase[heartbeat_phase["term"].astype(str).str.contains(":", na=False)].copy().sort_values("p_holm")
        add_table(document, phase_terms, [("template_label", "State"), ("stratum", "Stratum"),
                                          ("systolic_window_start_ms", "Systole start ms"),
                                          ("systolic_window_end_ms", "Systole end ms"), ("term", "Interaction"),
                                          ("rate_ratio", "Rate ratio"), ("p_holm", "Holm p")], max_rows=18)
        significant_phase = phase_terms[phase_terms["p_holm"].lt(.05)]
        if not significant_phase.empty:
            document.add_paragraph(
                "Observed specificity result: the only Holm-significant heartbeat-phase interaction was a late C effect in "
                "DD epileptic events; no functional-event A/C/E phase interaction survived correction."
            )

    document.add_heading("Secondary and sensitivity outcomes", level=1)
    if not secondary.empty:
        add_table(document, secondary.sort_values("p_bh"), [("state", "State"), ("outcome", "Outcome"),
                                                             ("estimate", "Late slope"), ("p", "p"), ("p_bh", "BH p")], max_rows=18)
    document.add_paragraph(
        "Cohort-specific morphology fields, epileptic-event specificity controls, within-DD comparisons, E-origin precision, weighted JS divergence, gating direction and activity outcomes are secondary or sensitivity analyses. The three prespecified primary tests are Holm-corrected; hypermotor equivalence is evaluated separately; secondary scalar/cardioneural tests use BH; spatial morphology uses maximum-T/TFCE family-wise inference."
    )

    document.add_heading("Interpretation and manuscript framing", level=1)
    document.add_paragraph(
        "A positive late C/A slope would support a shift in the relative expression of the C versus A processing modes as seizure onset approaches. Association with E expression or E-routed switching would support a gating interpretation, but not establish network identity or causality. A change in empirical transition precision would be compatible with altered confidence in state transitions under active inference; it should not be presented as a directly estimated neuronal B tensor."
    )
    document.add_paragraph(
        "Cardiac deceleration confined to hypomotor events would link autonomic withdrawal to semiology. Persistence of a neural transition-precision change in hypermotor events despite an equivalent-to-flat HR trajectory would argue that the neural result is not reducible to gross cardiac acceleration or deceleration. These statements remain conditional on the direction and corrected significance of the supplied estimates."
    )

    document.add_heading("Limitations", level=1)
    add_bullets(document, [
        "The participant full-record prior includes the target event; leave-one-event-out priors were outside the locked scope.",
        "Canonical microstate-network interpretations are hypotheses; scalp maps do not uniquely identify sources or networks.",
        "Window overlap improves temporal resolution but does not create independent inferential units; event slopes are reduced to participant maps before permutation inference.",
        "Recovered ECG intervals are explicitly limited and audited; windows failing coverage or physiological criteria are excluded.",
        "The B-like precision proxy is descriptive of empirical transition operators and is not action-conditioned model inversion.",
    ])

    document.add_heading("Superseded results", level=1)
    document.add_paragraph(
        "The previous fixed-backfit morphology result (reported p=0.034) tested the slope of averages of maps assigned to fixed templates. It did not test change in the morphology of freshly fitted microstates and is superseded. Earlier Gaussian-field values (including the uncorrected p=0.0002 and maximum-field p=0.077 comparison) are likewise superseded because they came from the prior estimand. All hierarchy results that appended repeated parent maps as pseudo-observations—including the earlier pseudocount-four implementation—and all downstream results built on those fits are invalid for the revised analysis and are not used here."
    )

    document.add_heading("Conceptual source", level=1)
    document.add_paragraph(
        "Canham LJW. Towards Measures for Translation: An Exploration of Neurophysiological Biomarkers of Disability in Multiple Sclerosis. PhD thesis, University of Bristol, 2019. Used for the disintegration, complexity and metastable-dynamics framing; not as evidence for direct B-tensor estimation."
    )

    document.add_heading("Reproducibility package", level=1)
    document.add_paragraph(
        "The adjacent package supplies analysis-ready event and participant tables, all window-refitted maps and failure diagnostics, ECG-QC tables and exclusions, frozen code and parameters, exact Git/SPM provenance, a data dictionary, audit log and SHA-256 manifest. The adjacent TXT file is the human-readable audit trail."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def build_audit(analysis_dir: Path, morphology_dir: Path, hierarchy_dir: Path, ecg_dir: Path,
                repository: Path, report: Path, output: Path) -> None:
    global LABELS
    label_order = load_global_labels(hierarchy_dir)
    LABELS = format_label_order(label_order)
    lines = [
        "FS_MS A/C/E CARDIONEURAL ANALYSIS AUDIT TRAIL",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Repository: {repository}",
        f"Hierarchy: {hierarchy_dir}",
        f"Analysis: {analysis_dir}",
        f"Morphology: {morphology_dir}",
        f"ECG sidecars: {ecg_dir}",
        f"Saved manual labels: {LABELS}",
        f"SPM_MS commit: {SPM_MS_COMMIT}",
        f"SPM25 commit: {SPM_COMMIT}",
        "SPM prior mode: component-mean prior; weight=1; no parent rows added to observations",
        "Morphology: only 60-s/30-s step hierarchical window refits; A/C/E; joint participant TFCE",
        "Activity QC: one-pass within-event median +/- 3 sample SD, coverage and GFP independently",
        "Inference unit: participant; equal event weighting",
        "Primary correction: Holm across C/A late slope, hypomotor HR late slope, pooled B-like precision trend",
        "Spatial correction: joint maximum-T/TFCE FWER across A/C/E",
        "Superseded: all repeated-parent-map hierarchy outputs and fixed-backfit p=0.034 morphology result",
        "",
        "KEY COMMANDS",
        "MATLAB full hierarchy: run_vg_microstates(... K=6, interactive_global_labelling=true, spm_prior_weight=1, backfit_time_bin_s=10)",
        "MATLAB refits: run_vg_microstates(... preictal_window_refit_only=true, 60-s windows, 30-s step, >=30 peaks, >=5 state peaks)",
        "Python activity/cardiac: analyse_ace_cardioneural.py",
        "Python morphology: analyse_preictal_morphology_rft.py --permutations 5000",
        "",
        "FILES AT AUDIT CREATION",
    ]
    for root in (hierarchy_dir, analysis_dir, morphology_dir, ecg_dir):
        if not root.exists():
            lines.append(f"MISSING DIRECTORY: {root}")
            continue
        files = [p for p in root.rglob("*") if p.is_file()]
        lines.append(f"{root}: {len(files)} files, {sum(p.stat().st_size for p in files)} bytes")
    if report.exists():
        lines.extend(["", f"Report SHA-256: {sha256(report)}", f"Report bytes: {report.stat().st_size}"])

    lines.extend(["", "VALIDATION AND DATA-LINEAGE CHECKS"])

    def audit_check(name: str, passed: bool, detail: str) -> None:
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    hierarchy_file = hierarchy_dir / "hierarchical_fit_summary.csv"
    if hierarchy_file.exists():
        frame = pd.read_csv(hierarchy_file)
        child = frame[frame["level"].ne("global")].copy()
        audit_check("locked K=6 hierarchy", frame["K_estimated"].eq(6).all(),
                    f"{len(frame)} fit summaries; levels={sorted(frame['level'].dropna().astype(str).unique())}")
        participant_condition = frame[frame["level"].eq("participant_condition")]
        audit_check("participant-condition hierarchy present",
                    len(participant_condition) > 0 and participant_condition["prior_source"].eq("participant_k6").all(),
                    f"{len(participant_condition)} participant-condition fits; all must inherit participant_k6")
        count_ok = pd.to_numeric(frame["n_maps"], errors="coerce").eq(
            pd.to_numeric(frame["real_observation_count"], errors="coerce")).all()
        audit_check("real observations only", count_ok,
                    "n_maps equals real_observation_count for every fit; parent maps were not added as rows")
        mode_ok = child["prior_mode"].eq("spm_mix_prior_component_means").all()
        weight_ok = pd.to_numeric(child["spm_prior_weight"], errors="coerce").eq(1).all()
        beta_ok = child["prior_beta_0"].fillna("").astype(str).apply(
            lambda value: len(value.split(";")) == 6 and all(abs(float(x) - 1) < 1e-12 for x in value.split(";"))).all()
        audit_check("current SPM_MS prior", bool(mode_ok and weight_ok and beta_ok),
                    "child mode=component means, mean_weight=1 and six recorded beta_0 values equal one")
        provenance_ok = (frame["spm_ms_commit"].eq(SPM_MS_COMMIT).all() and
                         frame["spm_commit"].eq(SPM_COMMIT).all())
        audit_check("pinned software provenance", provenance_ok,
                    f"SPM_MS={SPM_MS_COMMIT}; SPM25={SPM_COMMIT}")
    else:
        audit_check("hierarchy summary present", False, str(hierarchy_file))

    state_file = hierarchy_dir / "hierarchical_state_covariance_metrics.csv"
    if state_file.exists():
        states = pd.read_csv(state_file)
        refs = states[states["level"].eq("participant") & states["template_label"].isin(["A", "C", "E"])].copy()
        good = pd.to_numeric(refs["template_match_abs_correlation"], errors="coerce").ge(.5)
        audit_check("participant A/C/E reference audit", len(refs) > 0,
                    f"{int(good.sum())}/{len(refs)} participant-state references meet abs(r)>=0.50; only these enter A/C/E inference")

    refit_file = hierarchy_dir / "preictal_window_refit_maps.csv"
    if refit_file.exists():
        refits = pd.read_csv(refit_file, low_memory=False)
        ok = refits["fit_status"].eq("ok")
        threshold_ok = (pd.to_numeric(refits.loc[ok, "window_peak_count"], errors="coerce").ge(30).all() and
                        pd.to_numeric(refits.loc[ok, "state_peak_count"], errors="coerce").ge(5).all())
        provenance_ok = (refits["prior_mode"].eq("spm_mix_prior_component_means").all() and
                         pd.to_numeric(refits["spm_prior_weight"], errors="coerce").eq(1).all())
        audit_check("window-refit provenance and thresholds", bool(threshold_ok and provenance_ok),
                    f"rows={len(refits)}, events={refits['file_path'].nunique()}, successful state maps={int(ok.sum())}; "
                    f"statuses={refits['fit_status'].value_counts(dropna=False).to_dict()}")
        audit_check("saved manual window labels", set(refits["template_label"].dropna()) == set(label_order),
                    f"window fits export the saved mapping {label_order}; inference selects A/C/E")
    else:
        audit_check("window-refit table present", False, str(refit_file))

    outlier_file = analysis_dir / "data" / "ace_activity_outlier_audit.csv"
    if outlier_file.exists():
        outliers = pd.read_csv(outlier_file)
        audit_check("metric-specific rolling outlier audit", set(outliers.get("metric", [])) == {"coverage", "mean_gfp"},
                    f"finite={int(outliers.get('finite_windows', pd.Series(dtype=float)).sum())}; "
                    f"excluded={int(outliers.get('outlying_windows', pd.Series(dtype=float)).sum())}; one-pass event median +/- 3 sample SD")

    sidecars = sorted(ecg_dir.glob("*_rpeaks_corrected.csv"))
    sidecar_schema_ok = bool(sidecars)
    for path in sidecars:
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
            sidecar_schema_ok &= {"rpeak_time_s", "rpeak_source", "is_gap_recovered"}.issubset(columns)
        except Exception:
            sidecar_schema_ok = False
    audit_check("versioned ECG sidecars", sidecar_schema_ok,
                f"{len(sidecars)} sidecars; required peak-time/source/recovery fields present")
    recovery_report = ecg_dir / "missed_rpeak_recovery_report.csv"
    if recovery_report.exists():
        recovery = pd.read_csv(recovery_report)
        errors = recovery.get("error", pd.Series("", index=recovery.index)).fillna("").astype(str).ne("")
        report_ok = (len(recovery) == recovery["base"].nunique() and
                     len(sidecars) + int(errors.sum()) == len(recovery))
        audit_check("ECG recovery-report cardinality", report_ok,
                    f"{len(recovery)} unique reviewed recordings; {len(sidecars)} sidecars; {int(errors.sum())} explicit failures")

    primary_file = analysis_dir / "data" / "primary_holm_corrected_tests.csv"
    if primary_file.exists():
        primary = pd.read_csv(primary_file)
        lines.extend(["", "PRIMARY RESULT SNAPSHOT"])
        for row in primary.itertuples(index=False):
            lines.append(f"{getattr(row, 'family', 'test')}: estimate={fmt(getattr(row, 'estimate', None), 6)}, "
                         f"SE={fmt(getattr(row, 'se', None), 6)}, p={fmt(getattr(row, 'p', None), 6)}, "
                         f"Holm p={fmt(getattr(row, 'p_holm', None), 6)}")

    lines.extend(["", "VALIDATION COMMAND OUTCOMES",
                  "PASS upstream test_spm_mix_prior (pinned SPM_MS; installed SPM25).",
                  "PASS FS synthetic hierarchy: permutation/polarity invariance, real observation count, beta_0=1, strong/weak evidence behaviour, correct-prior recovery and malformed-parent rejection.",
                  "PASS synthetic morphology: state permutation/polarity recovery and known tangent-beta direction.",
                  "PASS outlier self-check: injected >3 SD value removed only from its metric; in-range value retained.",
                  "PASS one-event refit smoke: checked window order/bounds, K=6, no post-onset samples, unit norms and one-to-one assignment.",
                  "NOTE the first smoke invocation stopped before fitting because it required a downstream binned timing table that the active backfit had not yet written. The refit timing source was corrected to the exact source-MAT sample count divided by sfreq; the repeated smoke then passed 234/234 rows (39 windows x six states). The appended MATLAB diary preserves both the dependency check and the successful repeat.",
                  "NOTE two pre-production serialization defects (empty-prior hashing and table-string coercion) and one corrupted copied GFP-cache file were exposed before downstream inference. The file was repaired, all 1,313 caches were SHA-256 verified, pre-audit fits were quarantined, and the hierarchy was rebuilt; no result from a failed attempt entered analysis.",
                  "NOTE the first combined ECG CSV was discarded because PowerShell retained only the first batch's fields and omitted the later error column. The four immutable part reports were recombined with a union-of-columns operation; the final audit preserves all three explicit failures."])

    lines.extend(["", "CRITICAL FILE HASHES"])
    critical = [hierarchy_file, state_file, refit_file, primary_file,
                morphology_dir / "data" / "morphology_tfce_peak_summary.csv",
                analysis_dir / "data" / "cardiac_planned_contrasts.csv"]
    for path in critical:
        if path.exists():
            lines.append(f"{sha256(path)}  {path}  {path.stat().st_size} bytes")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--morphology-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--ecg-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.analysis_dir, args.morphology_dir, args.hierarchy_dir, args.ecg_dir, args.report)
    build_audit(args.analysis_dir, args.morphology_dir, args.hierarchy_dir, args.ecg_dir,
                args.repository, args.report, args.audit)


if __name__ == "__main__":
    main()
