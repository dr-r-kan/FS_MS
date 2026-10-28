"""Read and validate the fitted-state labels of the global K=6 fit.

Since the pre-registration the labels are not hand-chosen: they are derived by the pre-specified rule
``template_prespecified_v2`` (``fsms/matlab/template_prespecified_v2.m``) -- the exact optimal one-to-one
assignment of the six global fitted states to MetaMaps classes A, B, C, E, F, G (class D excluded a priori)
maximising summed polarity-invariant spatial correlation.  State names are NOMINAL: primary and secondary
outcomes do not depend on them.  An assigned |r| < 0.5 does not stop the analysis; it sets
``labelling_quality = 'below_threshold'`` (descriptive per-state names unreliable).  Only technical failures
(no usable channel match, non-finite maps, wrong number of states, tied assignment) stop the stage.
The fitted-state order depends on the data and RNG, so it must always be READ from the run
(``global/global_manual_labels.csv`` or ``recovery_audit.json``), never hard-coded.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PRESPECIFIED_RULE_ID = "template_prespecified_v2"
LABELLING_QUALITIES = ("ok", "below_threshold")
PRESPECIFIED_CLASSES = ("A", "B", "C", "E", "F", "G")
STRONG_THRESHOLD = 0.5
ALLOWED_CHANNEL_MODES = ("labels", "coordinates_idw_to_template")
GATE_KEYS = ("compact_export_validated_against_full_backfit",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_label_order(labels) -> tuple[str, ...]:
    """Six unique A-G labels including A, C and E (also accepts the legacy hand-chosen sets)."""
    order = tuple(str(label).strip().upper() for label in labels)
    if len(order) != 6 or len(set(order)) != 6 or not set(order) <= set("ABCDEFG") or not set("ACE") <= set(order):
        raise ValueError("Global labels must be six unique A-G labels including A, C and E")
    return order


def validate_prespecified_labels(labels) -> tuple[str, ...]:
    """The labels the pre-specified rule can produce: a permutation of A, B, C, E, F, G (D never appears)."""
    order = tuple(str(label).strip().upper() for label in labels)
    if sorted(order) != sorted(PRESPECIFIED_CLASSES):
        raise ValueError(f"Pre-specified labels must be a permutation of {PRESPECIFIED_CLASSES}; got {order}")
    return order


def load_global_labels(results_dir: Path) -> tuple[str, ...]:
    manual = Path(results_dir) / "global" / "global_manual_labels.csv"
    if manual.exists():
        with manual.open(newline="", encoding="utf-8-sig") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["state_index"]))
        return validate_label_order(row["label"] for row in rows)

    covariance = Path(results_dir) / "hierarchical_state_covariance_metrics.csv"
    with covariance.open(newline="", encoding="utf-8-sig") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row["level"] == "global"),
            key=lambda row: int(float(row["state_index"])),
        )
    return validate_label_order(row["template_label"] for row in rows)


def load_label_source(results_dir: Path) -> str:
    """The ``source`` column of global_manual_labels.csv (``template_prespecified_v2`` for the study)."""
    manual = Path(results_dir) / "global" / "global_manual_labels.csv"
    with manual.open(newline="", encoding="utf-8-sig") as handle:
        sources = {row["source"] for row in csv.DictReader(handle)}
    if len(sources) != 1:
        raise ValueError(f"global_manual_labels.csv mixes label sources: {sorted(sources)}")
    return sources.pop()


def format_label_order(labels) -> str:
    return ", ".join(f"State {index}={label}" for index, label in enumerate(validate_label_order(labels), 1))


def _read(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text(encoding="utf-8"))


def validate_labelling_audit(audit_or_path) -> dict:
    """Schema and acceptance check of ``global/labelling_audit.json``; returns the audit dict."""
    audit = _read(audit_or_path)
    if audit.get("rule_id") != PRESPECIFIED_RULE_ID:
        raise ValueError(f"Labelling rule is {audit.get('rule_id')!r}, expected {PRESPECIFIED_RULE_ID!r}")
    if audit.get("accepted") is not True:
        raise ValueError("labelling_audit.json does not record a technically sound labelling")
    labels = validate_prespecified_labels(audit["labels_by_fitted_state"])
    absolute = audit["assigned_abs_r"]
    if len(absolute) != 6 or audit.get("strong_threshold") != STRONG_THRESHOLD:
        raise ValueError("assigned |r| must list six values and strong_threshold must be 0.5")
    _check_quality(audit.get("labelling_quality"), audit.get("states_below_threshold"), absolute,
                   audit.get("nominal_label_semantics"))
    matrix = audit["correlation_matrix_abs"]
    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise ValueError("correlation_matrix_abs must be 6 x 6 (states x classes A,B,C,E,F,G)")
    if tuple(audit["correlation_matrix_abs_columns"]) != PRESPECIFIED_CLASSES:
        raise ValueError("correlation_matrix_abs columns must be A,B,C,E,F,G")
    for state, label in enumerate(labels):
        if abs(matrix[state][PRESPECIFIED_CLASSES.index(label)] - absolute[state]) > 1e-9:
            raise ValueError(f"assigned |r| for state {state + 1} disagrees with the correlation matrix")
    if audit.get("channel_match_mode") not in ALLOWED_CHANNEL_MODES:
        raise ValueError(f"channel_match_mode {audit.get('channel_match_mode')!r} is not allowed")
    if not _SHA256.match(str(audit.get("template_sha256", ""))):
        raise ValueError("template_sha256 missing or malformed")
    if "D" not in audit.get("classes_excluded_a_priori", []):
        raise ValueError("class D must be recorded as excluded a priori")
    return audit


def _check_quality(quality, states_below, absolute, semantics) -> None:
    """labelling_quality must agree with the per-state |r| list (a flag, never a stop)."""
    if quality not in LABELLING_QUALITIES:
        raise ValueError(f"labelling_quality must be one of {LABELLING_QUALITIES}, got {quality!r}")
    weak = [index + 1 for index, value in enumerate(absolute) if not value >= STRONG_THRESHOLD]
    if (quality == "ok") != (not weak):
        raise ValueError(f"labelling_quality {quality!r} contradicts the assigned |r| values {absolute}")
    listed = sorted(int(item["state_index"] if isinstance(item, dict) else item) for item in (states_below or []))
    if listed != weak:
        raise ValueError(f"states below threshold {listed} do not match the assigned |r| values (weak: {weak})")
    if not semantics or "nominal" not in str(semantics):
        raise ValueError("the nominal-label semantics statement is missing")


def labelling_quality_from_recovery_audit(audit_or_path) -> tuple[str, tuple[int, ...]]:
    """(labelling_quality, 1-based states below the 0.5 threshold) recorded by the MATLAB stage."""
    audit = _read(audit_or_path)
    quality = audit["labelling_quality"]
    below = tuple(sorted(int(item["state_index"] if isinstance(item, dict) else item)
                         for item in audit.get("labelling_states_below_threshold", [])))
    return quality, below


def fit_labels_from_recovery_audit(audit_or_path) -> tuple[str, ...]:
    """Fitted-state labels (state 1..6 order) recorded by the MATLAB stage in recovery_audit.json."""
    audit = _read(audit_or_path)
    labels = validate_prespecified_labels(audit["fit_labels"])
    legacy = audit.get("manual_labels_by_fitted_state")
    if legacy is not None and tuple(str(x) for x in legacy) != labels:
        raise ValueError("fit_labels and manual_labels_by_fitted_state disagree")
    return labels


def validate_recovery_audit_schema(audit_or_path) -> dict:
    """Schema check of ``timestamped_backfits/recovery_audit.json`` (stage-C output, schema_version 2)."""
    audit = _read(audit_or_path)
    required = ("schema_version", "manifest_sha256", "fit_labels", "labelling_rule", "labelling_correlations",
                "labelling_quality", "labelling_states_below_threshold", "labelling_semantics",
                "matlab_version", "script_sha256", "script_fingerprint", "n_timestamped_records", "n_transitions",
                "compact_export_validated_against_full_backfit", "focused_window_fit_recovery_validation")
    missing = [key for key in required if key not in audit]
    if missing:
        raise ValueError(f"recovery_audit.json lacks: {missing}")
    if audit["schema_version"] < 2:
        raise ValueError("recovery_audit.json is a pre-labelling-rule (legacy) audit")
    fit_labels_from_recovery_audit(audit)
    if audit["labelling_rule"] != PRESPECIFIED_RULE_ID:
        raise ValueError(f"labelling_rule is {audit['labelling_rule']!r}")
    if not _SHA256.match(str(audit["manifest_sha256"])) or not _SHA256.match(str(audit["script_fingerprint"])):
        raise ValueError("manifest_sha256 / script_fingerprint must be lower-case SHA-256 hex")
    correlations = audit["labelling_correlations"]
    _check_quality(audit["labelling_quality"], audit["labelling_states_below_threshold"],
                   correlations["assigned_abs_r"], audit["labelling_semantics"])
    if audit["compact_export_validated_against_full_backfit"] is not True:
        raise ValueError("compact export was not validated against the full backfit")
    gate = audit["focused_window_fit_recovery_validation"]
    if gate.get("performed") is not True or gate.get("transition_matrix_exact_match") is not True:
        raise ValueError("focused-window validation gate is not satisfied")
    if gate.get("n_windows_checked", 0) < 1 or gate.get("n_windows_mismatched", 1) != 0:
        raise ValueError("focused-window validation checked no windows or found mismatches")
    for entry in audit["script_sha256"]:
        if not _SHA256.match(entry["sha256"]):
            raise ValueError(f"malformed script hash for {entry.get('path')}")
    return audit


if __name__ == "__main__":
    assert validate_label_order("FEGBAC") == ("F", "E", "G", "B", "A", "C")
    try:
        validate_label_order("FFGBAC")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate-label check failed")
    try:
        validate_prespecified_labels("FEGBAD")
    except ValueError:
        pass
    else:
        raise AssertionError("class D must be rejected by the pre-specified label set")
    print("self-check passed")
