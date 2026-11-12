# flake8: noqa: E501
"""Build self-contained profiler-style reports from cross-config artifacts.

The reporter is deliberately downstream of execution.  It validates a sealed
attempt directory, then renders the recorded comparison, token deltas, actual
operator provenance, and materialized topology without changing either the
training or rollout path.
"""

from __future__ import annotations

import html
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from rl_engine.validation.cross_config.artifacts import ArtifactStore

REPORT_SCHEMA_VERSION = 2
_STATUS_LABELS = {"pass": "PASS", "warning": "WARN", "failure": "FAIL", "info": "INFO"}
_STATUS_COLORS = {
    "pass": "#39d98a",
    "warning": "#f4b942",
    "failure": "#ff5c77",
    "info": "#70a7ff",
}
_REPORT_IMAGE_COLORS = {
    "background": "#eef0f2",
    "panel": "#ffffff",
    "panel_alt": "#f7f8f9",
    "track": "#c8cdd2",
    "line": "#cbd1d7",
    "grid": "#e5e8eb",
    "text": "#20252b",
    "muted": "#66707a",
    "green": "#43a64b",
    "yellow": "#d79700",
    "red": "#d64242",
    "blue": "#3677c8",
    "purple": "#7659bb",
    "pass": "#43a64b",
    "warning": "#d79700",
    "failure": "#d64242",
    "info": "#3677c8",
}


def build_cross_config_attempt_report(
    attempt_dir: str | Path,
    *,
    title: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Render one immutable cross-configuration attempt into a drift report.

    ``attempt_dir`` must be a directory emitted by :class:`PairedRunner`.  The
    report never derives a numerical claim from an incomplete attempt.  A
    caller can set ``validate=False`` only when it has already validated the
    same sealed directory in the current process.
    """

    directory = Path(attempt_dir)
    store = ArtifactStore(directory.parent)
    if validate:
        store.validate_completed_attempt(directory)

    comparison = _read_attempt_json(directory / "comparison.json")
    actual = _read_attempt_json(directory / "actual.json")
    materialized = _read_attempt_json(directory / "materialized.json")
    identity = _read_attempt_json(directory / "identity.json")
    token_payload = store.load_tensor_bundle(directory / "token_diffs.pt")
    tensors = token_payload["tensors"]
    active_mask = tensors["active_mask"].to(dtype=torch.bool)
    absolute_diff = tensors["absolute_diff"].to(dtype=torch.float32)
    active_count = int(active_mask.sum().item())
    worst_token = _worst_token(absolute_diff, active_mask)

    materialized_case = _plain_mapping(materialized.get("materialized_case"))
    case = _plain_mapping(materialized_case.get("case"))
    requested = _plain_mapping(case.get("requested"))
    rollout_requested = _plain_mapping(requested.get("rollout"))
    training_requested = _plain_mapping(requested.get("training"))
    rollout_actual = _plain_mapping(actual.get("rollout"))
    training_actual = _plain_mapping(actual.get("training"))
    diagnostics = _plain_mapping(comparison.get("diagnostics"))

    failures: list[dict[str, Any]] = []
    if not bool(comparison.get("comparable")):
        failures.append(
            {
                "code": "not_comparable",
                "message": "The sealed attempt failed an identity or artifact comparison gate.",
            }
        )
    for field in ("identity_errors", "artifact_errors"):
        for value in comparison.get(field) or ():
            failures.append({"code": field, "message": str(value)})
    if not bool(comparison.get("passed")) and not failures:
        failures.append(
            {
                "code": str(comparison.get("status", "comparison_failed")),
                "message": "The selected-token comparison exceeded its fixed contract.",
            }
        )

    metrics = {
        "active_token_count": active_count,
        "mismatch_count": int(comparison.get("mismatch_count", 0)),
        "max_abs_dlogp": 0.0 if worst_token is None else worst_token["abs_dlogp"],
        "fixed_threshold": comparison.get("fixed_threshold"),
        "comparison_status": comparison.get("status"),
        "comparable": bool(comparison.get("comparable")),
        "passed": bool(comparison.get("passed")),
    }
    axes = {
        "case_id": comparison.get("case_id"),
        "attempt_id": comparison.get("attempt_id"),
        "rollout_tp": rollout_requested.get("tensor_parallel_size"),
        "rollout_cp": rollout_requested.get("context_parallel_size"),
        "rollout_dtype": rollout_requested.get("dtype"),
        "training_dtype": training_requested.get("compute_dtype"),
        "training_sharding": training_requested.get("sharding"),
        "contract_fingerprint": comparison.get("contract_fingerprint"),
    }
    rollout_backend = _first_present(
        rollout_actual.get("backend_id"), rollout_actual.get("actual_backend")
    )
    training_backend = _first_present(
        training_actual.get("backend_id"), training_actual.get("actual_backend")
    )
    provenance = {
        "operator_source": actual.get("operator_source"),
        "execution_fingerprint": actual.get("execution_fingerprint"),
        "environment_fingerprint": actual.get("environment_fingerprint"),
        "actual_backend": _backend_label(rollout_backend, training_backend),
        "rollout_backend": rollout_backend,
        "training_backend": training_backend,
        "rollout": rollout_actual,
        "training": training_actual,
    }
    samples = _samples_from_identity(identity, active_mask)
    replay_manifest = {
        "mode": "cross_config_fixed_replay",
        "samples": samples,
        "batch_invariance_cases": (),
        "validation": {"warnings": (), "failures": failures},
        "runtime_provenance": provenance,
    }
    result_cube = {
        "mode": "cross_config_fixed_replay",
        "axes": axes,
        "metrics": metrics,
        "worst_token": worst_token or {},
        "metadata_validation": replay_manifest["validation"],
        "runtime_provenance": provenance,
        "diagnostics": diagnostics,
    }
    return build_drift_report(
        replay_manifest=replay_manifest,
        result_cube=result_cube,
        runtime_provenance=provenance,
        title=title
        or f"Cross-config drift: {comparison.get('case_id', directory.parent.name)} / {directory.name}",
    )


def build_drift_report(
    *,
    replay_manifest: Mapping[str, Any] | None = None,
    result_cube: Mapping[str, Any] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Normalize existing audit artifacts into a visual diagnostic report."""

    manifest = _plain_mapping(replay_manifest)
    cube = _plain_mapping(result_cube)
    metrics = _plain_mapping(cube.get("metrics"))
    axes = _plain_mapping(cube.get("axes"))
    validation = _plain_mapping(cube.get("metadata_validation") or manifest.get("validation"))
    provenance = _plain_mapping(
        runtime_provenance or cube.get("runtime_provenance") or manifest.get("runtime_provenance")
    )
    samples = [
        _plain_mapping(item) for item in manifest.get("samples", []) if isinstance(item, Mapping)
    ]
    warnings = [
        _plain_mapping(item) for item in validation.get("warnings", []) if isinstance(item, Mapping)
    ]
    failures = [
        _plain_mapping(item) for item in validation.get("failures", []) if isinstance(item, Mapping)
    ]

    max_abs_dlogp = _number(metrics.get("max_abs_dlogp"))
    warning_count = _number(metrics.get("warning_count"), default=0.0) or 0.0
    metadata_warning_count = (
        _number(metrics.get("metadata_warning_count"), default=float(len(warnings))) or 0.0
    )
    metadata_failure_count = (
        _number(metrics.get("metadata_failure_count"), default=float(len(failures))) or 0.0
    )
    runtime_fallback = bool(metrics.get("runtime_fallback") or provenance.get("fallback"))
    strict_failure = bool(metrics.get("runtime_strict_failure") or provenance.get("strict_failure"))
    comparison_passed = metrics.get("passed") is True

    if strict_failure or metadata_failure_count > 0:
        status = "failure"
    elif warning_count > 0 or metadata_warning_count > 0 or runtime_fallback:
        status = "warning"
    elif comparison_passed:
        status = "pass"
    elif (max_abs_dlogp or 0.0) > 0.0:
        status = "warning"
    else:
        status = "pass"

    span = max(1.0, float(len(samples)))
    has_timestamps = any(_number(sample.get("start_ts")) is not None for sample in samples)
    timeline_mode = "timestamp" if has_timestamps else "ordinal_diagnostic"
    events: list[dict[str, Any]] = []

    train_status = (
        "failure" if status == "failure" else "warning" if status == "warning" else "pass"
    )
    events.append(
        {
            "id": "train-audit",
            "kind": "bar",
            "lane": "Training audit",
            "start": 0.0,
            "end": span,
            "label": "training-side audit",
            "status": train_status,
            "details": {
                "mode": manifest.get("mode", cube.get("mode", "unknown")),
                "rank": cube.get("rank"),
            },
        }
    )
    for position, sample in enumerate(samples):
        sample_label = sample.get("sample_index")
        if sample_label is None:
            sample_label = sample.get("rollout_id")
        if sample_label is None:
            sample_label = position
        start = _number(sample.get("start_ts"), default=float(position))
        end = _number(sample.get("end_ts"), default=float(start or position) + 0.82)
        if end is None or start is None or end <= start:
            start, end = float(position), float(position) + 0.82
        events.append(
            {
                "id": f"rollout-{position}",
                "kind": "bar",
                "lane": "Rollout samples",
                "start": start,
                "end": end,
                "label": f"sample {sample_label}",
                "status": "info",
                "details": sample,
            }
        )

    actual_backend = _first_present(
        provenance.get("actual_backend"),
        provenance.get("backend_id"),
        (
            cube.get("axes", {}).get("logp_backend")
            if isinstance(cube.get("axes"), Mapping)
            else None
        ),
        provenance.get("requested_backend"),
        "unknown backend",
    )
    operator_status = "failure" if strict_failure else "warning" if runtime_fallback else "pass"
    events.append(
        {
            "id": "operator-backend",
            "kind": "bar",
            "lane": "Operator / backend",
            "start": 0.12,
            "end": max(0.94, span - 0.12),
            "label": str(actual_backend),
            "status": operator_status,
            "details": provenance,
        }
    )

    worst_token = _plain_mapping(cube.get("worst_token"))
    if worst_token:
        marker_position = (
            _number(worst_token.get("sample_position"), default=max(0.0, span - 0.5)) or 0.0
        )
        events.append(
            {
                "id": "worst-drift",
                "kind": "marker",
                "lane": "Drift markers",
                "start": marker_position + 0.41,
                "end": marker_position + 0.41,
                "label": f"|dlogp| {_format_number(worst_token.get('abs_dlogp'))}",
                "status": (
                    "failure"
                    if status == "failure"
                    else "warning" if status == "warning" else "pass"
                ),
                "details": worst_token,
            }
        )
    if warnings or failures or warning_count:
        marker_status = "failure" if failures else "warning"
        events.append(
            {
                "id": "validation-marker",
                "kind": "marker",
                "lane": "Drift markers",
                "start": max(0.2, span - 0.22),
                "end": max(0.2, span - 0.22),
                "label": f"{len(failures)} failures / {len(warnings)} warnings",
                "status": marker_status,
                "details": {
                    "warnings": warnings,
                    "failures": failures,
                    "dlogp_warning_count": warning_count,
                },
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "title": title or "RL-Kernel cross-config drift report",
        "status": status,
        "status_label": _STATUS_LABELS[status],
        "timeline_mode": timeline_mode,
        "timeline_note": (
            "Real artifact timestamps are shown."
            if timeline_mode == "timestamp"
            else (
                "No artifact timestamps were available; positions are stable sample "
                "ordinals, not elapsed time."
            )
        ),
        "lanes": [
            "Training audit",
            "Rollout samples",
            "Operator / backend",
            "Drift markers",
        ],
        "events": events,
        "axes": axes,
        "metrics": metrics,
        "worst_token": worst_token,
        "validation": validation,
        "runtime_provenance": provenance,
        "sample_count": len(samples),
        "replay_case_count": len(manifest.get("batch_invariance_cases", []) or []),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "cube_fingerprint": cube.get("fingerprint"),
    }


def build_drift_trace(report: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a consistency report into Chrome Trace Event JSON.

    The resulting file can be opened directly in Perfetto and expanded by
    process/thread track.  It deliberately uses the report's timestamp mode;
    ordinal reports remain diagnostic sample positions, never fabricated time.
    """

    normalized = _plain_mapping(report)
    events = [_plain_mapping(event) for event in normalized.get("events", [])]
    status = str(normalized.get("status", "info"))
    timeline_mode = str(normalized.get("timeline_mode", "ordinal_diagnostic"))
    lanes = [
        ("Audit", None),
        ("Training audit", "Training audit"),
        ("Rollout samples", "Rollout samples"),
        ("Execution", None),
        ("Operator / backend", "Operator / backend"),
        ("Token comparison", "Token comparison"),
        ("Validation", None),
        ("Drift markers", "Drift markers"),
    ]
    lane_ids = {name: 100 + index for index, (name, _) in enumerate(lanes)}
    process_id = 1
    trace_events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": process_id,
            "args": {"name": "RL-Kernel cross-config drift"},
        },
    ]
    for lane, event_lane in lanes:
        if event_lane is None:
            continue
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": process_id,
                "tid": lane_ids[lane],
                "args": {"name": lane},
            }
        )

    def trace_time(value: Any) -> float:
        number = _number(value, default=0.0) or 0.0
        # Chrome Trace timestamps are microseconds.  In ordinal mode the same
        # scale is retained only to make adjacent sample positions visible.
        return number * 1_000_000.0

    event_colors = {
        "pass": "good",
        "warning": "terrible",
        "failure": "bad",
        "info": "thread_state_running",
    }
    for event in events:
        event_lane = str(event.get("lane", "Drift markers"))
        tid = lane_ids.get(event_lane, lane_ids["Drift markers"])
        start = _number(event.get("start"), default=0.0) or 0.0
        end = _number(event.get("end"), default=start) or start
        details = event.get("details") if isinstance(event.get("details"), Mapping) else {}
        args = {
            "event_id": str(event.get("id", "")),
            "status": str(event.get("status", status)),
            "timeline_mode": timeline_mode,
            "timeline_note": normalized.get("timeline_note", ""),
            "details": details,
        }
        color = event_colors.get(str(event.get("status", "info")), "thread_state_running")
        if event.get("kind") == "marker":
            trace_events.append(
                {
                    "name": str(event.get("label", "marker")),
                    "cat": "consistency.drift",
                    "ph": "I",
                    "s": "t",
                    "pid": process_id,
                    "tid": tid,
                    "ts": trace_time(start),
                    "cname": color,
                    "args": args,
                }
            )
            continue
        trace_events.append(
            {
                "name": str(event.get("label", event.get("id", "event"))),
                "cat": "consistency.audit",
                "ph": "X",
                "pid": process_id,
                "tid": tid,
                "ts": trace_time(start),
                "dur": max(1.0, trace_time(end) - trace_time(start)),
                "cname": color,
                "args": args,
            }
        )

    metrics = _plain_mapping(normalized.get("metrics"))
    max_abs_dlogp = _number(metrics.get("max_abs_dlogp"))
    if max_abs_dlogp is not None:
        trace_events.append(
            {
                "name": "max |dlogp|",
                "cat": "consistency.metric",
                "ph": "C",
                "pid": process_id,
                "tid": lane_ids["Token comparison"],
                "ts": 0.0,
                "args": {"max_abs_dlogp": max_abs_dlogp},
            }
        )

    return {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "report_title": normalized.get("title", "RL-Kernel cross-config drift report"),
            "status": status,
            "timeline_mode": timeline_mode,
            "timeline_note": normalized.get("timeline_note", ""),
            "schema_version": normalized.get("schema_version", REPORT_SCHEMA_VERSION),
        },
    }


def write_drift_trace(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a Chrome Trace Event JSON file for Perfetto or trace viewers."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_drift_trace(report), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return output


def write_drift_bundle(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    include_preview: bool = True,
) -> Path:
    """Write a self-contained ``.rlk-drift`` desktop-viewer bundle.

    The bundle contains sanitized report JSON and a portable trace.  The PNG
    preview is included for PR/issue sharing but is not required by the viewer.
    Raw train dumps are intentionally not copied into the bundle.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = _plain_mapping(report)
    trace = build_drift_trace(normalized)
    manifest = {
        "format": "rl_kernel.cross_config_drift",
        "bundle_version": 1,
        "report_schema_version": normalized.get("schema_version", REPORT_SCHEMA_VERSION),
        "files": [
            "manifest.json",
            "report.json",
            "trace.json",
            *(["preview.png"] if include_preview else []),
        ],
        "preview_included": bool(include_preview),
        "title": normalized.get("title", "RL-Kernel cross-config drift report"),
        "status": normalized.get("status", "info"),
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))
        archive.writestr("report.json", json.dumps(normalized, ensure_ascii=True, indent=2))
        archive.writestr("trace.json", json.dumps(trace, ensure_ascii=True, indent=2))
        if include_preview:
            preview = render_drift_report_image(normalized)
            buffer = io.BytesIO()
            preview.save(buffer, format="PNG", optimize=True)
            archive.writestr("preview.png", buffer.getvalue())
    return output


def load_drift_bundle(path: str | Path) -> dict[str, Any]:
    """Load a ``.rlk-drift`` bundle without importing the optional GUI."""

    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if "report.json" not in names or "trace.json" not in names:
            raise ValueError(
                "invalid RL-Kernel drift bundle: report.json and trace.json are required"
            )
        report = json.loads(archive.read("report.json"))
        trace = json.loads(archive.read("trace.json"))
        manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in names else {}
    return {"manifest": manifest, "report": report, "trace": trace}


def render_drift_report(report: Mapping[str, Any]) -> str:
    """Render a report as a self-contained HTML document with an SVG timeline."""

    normalized = _plain_mapping(report)
    events = [_plain_mapping(event) for event in normalized.get("events", [])]
    lanes = [str(lane) for lane in normalized.get("lanes", [])]
    width = 1180
    left = 190
    right = 28
    top = 46
    row_height = 54
    timeline_width = width - left - right
    timeline_span = max(
        1.0,
        max(
            (_number(event.get("end"), default=1.0) or 1.0 for event in events),
            default=1.0,
        ),
    )
    svg_height = top + row_height * len(lanes) + 42
    event_map = {str(event.get("id")): event for event in events}

    def x(value: float) -> float:
        return left + max(0.0, min(timeline_span, value)) / timeline_span * timeline_width

    svg_parts = [
        f'<svg class="timeline" viewBox="0 0 {width} {svg_height}" role="img" aria-label="Consistency drift timeline">',
        f'<rect x="0" y="0" width="{width}" height="{svg_height}" rx="8" fill="#111827"/>',
    ]
    grid_steps = min(12, max(2, int(timeline_span) + 1))
    for index in range(grid_steps + 1):
        value = timeline_span * index / grid_steps
        xpos = x(value)
        svg_parts.append(
            f'<line x1="{xpos:.2f}" y1="{top - 18}" x2="{xpos:.2f}" y2="{svg_height - 38}" class="grid"/>'
        )
        svg_parts.append(
            f'<text x="{xpos:.2f}" y="{svg_height - 16}" class="axis-label">{html.escape(_format_number(value))}</text>'
        )
    for lane_index, lane in enumerate(lanes):
        ypos = top + lane_index * row_height
        svg_parts.append(
            f'<text x="18" y="{ypos + 13}" class="lane-label">{html.escape(lane)}</text>'
        )
        svg_parts.append(
            f'<line x1="{left}" y1="{ypos + 27}" x2="{width - right}" y2="{ypos + 27}" class="lane-line"/>'
        )

    for event in events:
        lane_index = lanes.index(str(event.get("lane"))) if str(event.get("lane")) in lanes else 0
        ypos = top + lane_index * row_height
        status = str(event.get("status", "info"))
        color = _STATUS_COLORS.get(status, _STATUS_COLORS["info"])
        event_id = html.escape(str(event.get("id")), quote=True)
        label = html.escape(_truncate(str(event.get("label", "event")), 28))
        title = html.escape(
            f"{event.get('label', 'event')} [{_STATUS_LABELS.get(status, status.upper())}]",
            quote=True,
        )
        if event.get("kind") == "marker":
            xpos = x(_number(event.get("start"), default=0.0) or 0.0)
            points = f"{xpos:.2f},{ypos + 7} {xpos + 9:.2f},{ypos + 16} {xpos:.2f},{ypos + 25} {xpos - 9:.2f},{ypos + 16}"
            svg_parts.append(
                f'<polygon points="{points}" fill="{color}" class="event" data-event-id="{event_id}" tabindex="0"><title>{title}</title></polygon>'
            )
            svg_parts.append(
                f'<text x="{xpos + 14:.2f}" y="{ypos + 20}" class="event-label">{label}</text>'
            )
        else:
            start = _number(event.get("start"), default=0.0) or 0.0
            end = _number(event.get("end"), default=start + 0.5) or start + 0.5
            xpos = x(start)
            event_width = max(8.0, x(end) - xpos)
            svg_parts.append(
                f'<rect x="{xpos:.2f}" y="{ypos + 5}" width="{event_width:.2f}" height="24" rx="4" fill="{color}" fill-opacity="0.78" class="event" data-event-id="{event_id}" tabindex="0"><title>{title}</title></rect>'
            )
            if event_width > 60:
                svg_parts.append(
                    f'<text x="{xpos + 8:.2f}" y="{ypos + 21}" class="event-label event-label-on-bar">{label}</text>'
                )
            else:
                svg_parts.append(
                    f'<text x="{xpos + event_width + 8:.2f}" y="{ypos + 21}" class="event-label">{label}</text>'
                )
    svg_parts.append("</svg>")
    svg = "".join(svg_parts)

    metrics = _plain_mapping(normalized.get("metrics"))
    metric_cards = [
        ("Max |dlogp|", _format_number(metrics.get("max_abs_dlogp"))),
        ("Active tokens", _format_number(metrics.get("active_token_count"))),
        ("Warnings", _format_number(metrics.get("warning_count"), default="0")),
        ("Replay cases", str(normalized.get("replay_case_count", 0))),
    ]
    cards_html = "".join(
        f'<div class="metric"><div class="metric-name">{html.escape(name)}</div><div class="metric-value">{html.escape(value)}</div></div>'
        for name, value in metric_cards
    )
    axes_html = _render_key_value_table(
        normalized.get("axes"), empty="No normalized axes were recorded."
    )
    provenance_html = _render_key_value_table(
        normalized.get("runtime_provenance"),
        empty="No runtime provenance was recorded.",
    )
    validation_html = _render_validation(normalized.get("validation"))
    event_json = json.dumps(event_map, ensure_ascii=True, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    status = str(normalized.get("status", "info"))
    status_color = _STATUS_COLORS.get(status, _STATUS_COLORS["info"])
    title = html.escape(str(normalized.get("title", "RL-Kernel cross-config drift report")))
    timeline_note = html.escape(str(normalized.get("timeline_note", "")))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#151d31; --line:#2a3854; --muted:#96a4bd; --text:#e9eef8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:13px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.shell {{ max-width:1220px; margin:0 auto; padding:28px 20px 42px; }} .top {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; }}
h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }} .subtitle {{ color:var(--muted); max-width:760px; }}
.status {{ border:1px solid {status_color}; color:{status_color}; border-radius:999px; padding:6px 12px; font-weight:700; letter-spacing:.06em; white-space:nowrap; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:24px 0 16px; }} .metric,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
.metric {{ padding:13px 15px; }} .metric-name {{ color:var(--muted); font-size:12px; }} .metric-value {{ font-size:20px; font-weight:700; margin-top:5px; }}
.panel {{ padding:16px; margin-top:16px; }} .panel h2 {{ margin:0 0 12px; font-size:15px; }} .timeline-wrap {{ overflow:auto; }} .timeline {{ min-width:900px; width:100%; height:auto; display:block; }}
.grid {{ stroke:#2c3c5d; stroke-width:1; stroke-dasharray:2 5; }} .lane-line {{ stroke:#25334d; stroke-width:1; }} .lane-label {{ fill:#cbd5e1; font-weight:600; }} .axis-label {{ fill:#8292ad; font-size:11px; text-anchor:middle; }}
.event {{ cursor:pointer; outline:none; }} .event:focus,.event:hover {{ filter:brightness(1.25); stroke:#fff; stroke-width:1.5; }} .event-label {{ fill:#dbe6f7; font-size:11px; pointer-events:none; }} .event-label-on-bar {{ fill:#07111f; font-weight:700; }}
.detail-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; vertical-align:top; padding:7px 8px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:500; width:32%; }} td {{ word-break:break-word; }} .empty {{ color:var(--muted); }} .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; }}
.warning {{ color:#f4b942; }} .failure {{ color:#ff5c77; }} .pass {{ color:#39d98a; }} .info {{ color:#70a7ff; }} pre {{ margin:0; white-space:pre-wrap; color:#cbd5e1; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
@media (max-width:760px) {{ .shell {{ padding:20px 12px 32px; }} .top {{ display:block; }} .status {{ display:inline-block; margin-top:12px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .detail-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main class="shell">
<div class="top"><div><h1>{title}</h1><div class="subtitle">{timeline_note}</div></div><div class="status">{html.escape(_STATUS_LABELS.get(status, status.upper()))}</div></div>
<section class="metrics">{cards_html}</section>
<section class="panel"><h2>Operator drift timeline <span class="pill {status}">{html.escape(str(normalized.get("timeline_mode", "diagnostic")))}</span></h2><div class="timeline-wrap">{svg}</div></section>
<section class="panel"><h2>Selected event</h2><div id="event-detail" class="empty">Select a bar or marker in the timeline.</div></section>
<div class="detail-grid"><section class="panel"><h2>Normalized axes</h2>{axes_html}</section><section class="panel"><h2>Runtime provenance</h2>{provenance_html}</section></div>
<section class="panel"><h2>Validation</h2>{validation_html}</section>
</main><script>
const EVENTS = {event_json};
const detail = document.getElementById('event-detail');
function renderValue(value) {{ if (value === null || value === undefined) return 'null'; if (typeof value === 'object') return JSON.stringify(value, null, 2); return String(value); }}
function selectEvent(id) {{
  const event = EVENTS[id];
  if (!event) return;
  const status = String(event.status || 'info');
  detail.replaceChildren();
  const pill = document.createElement('div');
  pill.className = 'pill ' + status;
  pill.textContent = status.toUpperCase();
  const heading = document.createElement('h3');
  heading.textContent = String(event.label || event.id);
  const pre = document.createElement('pre');
  pre.textContent = renderValue(event.details || {{}});
  detail.append(pill, heading, pre);
}}
document.querySelectorAll('[data-event-id]').forEach((node) => {{ node.addEventListener('click', () => selectEvent(node.dataset.eventId)); node.addEventListener('keydown', (e) => {{ if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); selectEvent(node.dataset.eventId); }} }}); }});
</script></body></html>"""


def write_drift_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a self-contained HTML report and return its resolved path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_drift_report(report), encoding="utf-8")
    return output


def render_drift_report_image(report: Mapping[str, Any], *, width: int = 2400) -> Any:
    """Render a static profiler-style report image.

    The image is intentionally self-contained and suitable for attaching to a
    PR, issue, or debug artifact.  It uses a light, high-density profiler
    layout: a compact tool chrome, a hierarchical track tree, a fine-grained
    ruler/grid, thin event bars, and tabular details below the timeline.
    """

    from PIL import Image, ImageDraw

    normalized = _plain_mapping(report)
    metrics = _plain_mapping(normalized.get("metrics"))
    events = [_plain_mapping(event) for event in normalized.get("events", [])]
    status = str(normalized.get("status", "info"))
    status_color = _REPORT_IMAGE_COLORS.get(status, _REPORT_IMAGE_COLORS["info"])
    bg = _REPORT_IMAGE_COLORS["background"]
    panel_bg = _REPORT_IMAGE_COLORS["panel"]
    panel_alt = _REPORT_IMAGE_COLORS["panel_alt"]
    line = _REPORT_IMAGE_COLORS["line"]
    text_color = _REPORT_IMAGE_COLORS["text"]
    muted = _REPORT_IMAGE_COLORS["muted"]

    height = 1680
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    regular = _load_report_font(22)
    small = _load_report_font(18)
    tiny = _load_report_font(15)
    label_font = _load_report_font(18, bold=True)
    section_font = _load_report_font(20, bold=True)
    title_font = _load_report_font(30, bold=True)
    metric_font = _load_report_font(25, bold=True)
    mono = _load_report_font(16, mono=True)

    def rect(
        box: tuple[int, int, int, int],
        fill: str,
        radius: int = 2,
        outline: str | None = None,
    ) -> None:
        draw.rectangle(box, fill=fill, outline=outline, width=1 if outline else 1)

    def write(
        x: int,
        y: int,
        value: Any,
        font: Any = regular,
        fill: str = text_color,
        anchor: str | None = None,
    ) -> None:
        draw.text((x, y), str(value), font=font, fill=fill, anchor=anchor)

    def fit(value: Any, limit: int) -> str:
        return _truncate(str(value), limit)

    def key_value_rows(values: Any, limit: int = 33) -> list[tuple[str, str]]:
        if not isinstance(values, Mapping):
            return []
        return [
            (fit(key, limit), fit(_format_value(values[key]), 47))
            for key in sorted(values, key=str)
        ]

    # Nsight-like tool chrome: compact title, run metadata, and controls.
    draw.rectangle((0, 0, width, 42), fill="#252a30")
    write(30, 12, "RL-KERNEL CROSS-CONFIG ANALYSIS", tiny, "#e9edf1")
    write(
        width - 30,
        12,
        "static report  |  schema v" + str(normalized.get("schema_version", REPORT_SCHEMA_VERSION)),
        tiny,
        "#b7c0c8",
        anchor="ra",
    )
    write(
        34,
        60,
        normalized.get("title", "RL-Kernel cross-config drift report"),
        title_font,
        text_color,
    )
    write(34, 101, "run / audit replay", tiny, muted)
    write(180, 101, normalized.get("timeline_note", ""), tiny, muted)
    badge_text = _STATUS_LABELS.get(status, status.upper())
    draw.rectangle((width - 188, 61, width - 34, 101), fill=panel_bg, outline=status_color, width=2)
    write(width - 111, 81, badge_text, label_font, status_color, anchor="mm")

    # Compact metric strip. These are values, not dashboard cards.
    strip_y = 127
    draw.rectangle((34, strip_y, width - 34, strip_y + 64), fill=panel_bg, outline=line, width=1)
    strip_items = [
        ("MAX |DLOGP|", _format_number(metrics.get("max_abs_dlogp")), status_color),
        (
            "ACTIVE TOKENS",
            _format_number(metrics.get("active_token_count")),
            _REPORT_IMAGE_COLORS["blue"],
        ),
        (
            "WARNINGS",
            _format_number(metrics.get("warning_count"), default="0"),
            _REPORT_IMAGE_COLORS["yellow"],
        ),
        (
            "REPLAY CASES",
            normalized.get("replay_case_count", 0),
            _REPORT_IMAGE_COLORS["purple"],
        ),
    ]
    strip_width = (width - 68) // len(strip_items)
    for index, (name, value, accent) in enumerate(strip_items):
        x0 = 34 + index * strip_width
        if index:
            draw.line((x0, strip_y + 10, x0, strip_y + 54), fill=line, width=1)
        draw.rectangle((x0, strip_y, x0 + 4, strip_y + 64), fill=accent)
        write(x0 + 18, strip_y + 12, name, tiny, muted)
        write(x0 + 18, strip_y + 34, value, metric_font, text_color)

    # Timeline panel: strict left track tree + dense aligned lanes.
    timeline_y = 216
    timeline_h = 620
    rect((34, timeline_y, width - 34, timeline_y + timeline_h), panel_bg, 2, line)
    write(52, timeline_y + 14, "CAPTURED EXECUTION", section_font, text_color)
    mode = str(normalized.get("timeline_mode", "diagnostic"))
    mode_color = (
        _REPORT_IMAGE_COLORS["yellow"] if mode != "timestamp" else _REPORT_IMAGE_COLORS["blue"]
    )
    write(width - 52, timeline_y + 18, mode, tiny, mode_color, anchor="ra")

    lanes = [
        ("Audit", None),
        ("Training audit", "Training audit"),
        ("Rollout samples", "Rollout samples"),
        ("Execution", None),
        ("Operator / backend", "Operator / backend"),
        ("Token comparison", "Token comparison"),
        ("Validation", None),
        ("Drift markers", "Drift markers"),
    ]
    left = 370
    right = width - 54
    ruler_y = timeline_y + 59
    chart_top = timeline_y + 91
    row_height = 62
    chart_bottom = chart_top + row_height * len(lanes)
    max_end = max((_number(event.get("end"), default=1.0) or 1.0 for event in events), default=1.0)
    span = max(1.0, max_end)

    def x_for(value: float) -> int:
        return int(left + max(0.0, min(span, value)) / span * (right - left))

    # Group header, ruler, minor grid, and hierarchical track labels.
    draw.rectangle((34, timeline_y + 42, width - 34, timeline_y + 80), fill="#f1f3f5")
    write(52, timeline_y + 54, "TRACKS", tiny, muted)
    write(left + 8, timeline_y + 54, "TIME / SAMPLE ORDINAL", tiny, muted)
    draw.line((left, ruler_y + 16, right, ruler_y + 16), fill=line, width=1)
    for tick in range(0, 33):
        value = span * tick / 32
        xpos = x_for(value)
        major = tick % 4 == 0
        draw.line(
            (xpos, ruler_y + (7 if major else 13), xpos, chart_bottom),
            fill=line if major else _REPORT_IMAGE_COLORS["grid"],
            width=1,
        )
        if major:
            write(xpos, ruler_y - 3, _format_number(value), tiny, muted, anchor="ma")

    for lane_index, (lane, event_lane) in enumerate(lanes):
        y0 = chart_top + lane_index * row_height
        draw.rectangle(
            (34, y0, right, y0 + row_height),
            fill=panel_alt if lane_index % 2 else panel_bg,
        )
        draw.line((34, y0 + row_height, right, y0 + row_height), fill=line, width=1)
        if event_lane is None:
            draw.rectangle((34, y0, left, y0 + row_height), fill="#e8ebee")
            write(52, y0 + 20, lane.upper(), label_font, text_color)
        else:
            write(52, y0 + 22, "|--", tiny, muted)
            write(92, y0 + 21, lane, tiny, text_color)

    event_colors = {
        "pass": _REPORT_IMAGE_COLORS["green"],
        "warning": _REPORT_IMAGE_COLORS["yellow"],
        "failure": _REPORT_IMAGE_COLORS["red"],
        "info": _REPORT_IMAGE_COLORS["blue"],
    }
    for event in events:
        event_lane = str(event.get("lane", "Drift markers"))
        lane_index = next(
            (idx for idx, (_, name) in enumerate(lanes) if name == event_lane),
            len(lanes) - 1,
        )
        y0 = chart_top + lane_index * row_height
        color = event_colors.get(str(event.get("status", "info")), event_colors["info"])
        start = _number(event.get("start"), default=0.0) or 0.0
        end = _number(event.get("end"), default=start + 0.5) or start + 0.5
        x0 = x_for(start)
        x1 = max(x0 + 10, x_for(end))
        if event.get("kind") == "marker":
            mid = x_for(start)
            draw.line((mid, y0 + 6, mid, y0 + row_height - 6), fill=color, width=3)
            draw.polygon([(mid - 7, y0 + 7), (mid + 7, y0 + 7), (mid, y0 + 17)], fill=color)
            marker_label = fit(event.get("label", "marker"), 34)
            label_x = mid + 14 if mid <= right - 250 else max(left + 10, mid - 240)
            write(label_x, y0 + 22, marker_label, tiny, color)
        else:
            bar_y = y0 + 22
            draw.rectangle((x0, bar_y, x1, bar_y + 18), fill=color)
            if x1 - x0 >= 130:
                write(
                    x0 + 10,
                    bar_y + 2,
                    fit(event.get("label", "event"), 42),
                    tiny,
                    "#ffffff",
                )
            else:
                write(
                    min(x1 + 10, right - 260),
                    bar_y + 2,
                    fit(event.get("label", "event"), 34),
                    tiny,
                    text_color,
                )

    # Keep the comparison track meaningful even when the dump only contains a
    # scalar worst-token summary instead of per-token samples.
    worst = _plain_mapping(normalized.get("worst_token"))
    comparison_y = chart_top + 5 * row_height + 31
    draw.line(
        (left + 18, comparison_y, right - 18, comparison_y),
        fill=_REPORT_IMAGE_COLORS["track"],
        width=3,
    )
    write(left + 18, comparison_y - 22, "train vs rollout", tiny, muted)
    if worst:
        sample_position = _number(worst.get("sample_position"), default=0.0) or 0.0
        comparison_x = x_for(sample_position + 0.5)
        draw.line(
            (comparison_x, comparison_y - 15, comparison_x, comparison_y + 15),
            fill=status_color,
            width=3,
        )
        write(
            comparison_x + 10,
            comparison_y - 10,
            f"delta={_format_number(worst.get('abs_dlogp'))}",
            tiny,
            status_color,
        )

    # A compact legend makes the static image readable without hover state.
    legend_y = timeline_y + timeline_h - 30
    for x0, label, color in (
        (52, "PASS", _REPORT_IMAGE_COLORS["green"]),
        (130, "WARN", _REPORT_IMAGE_COLORS["yellow"]),
        (214, "FAIL", _REPORT_IMAGE_COLORS["red"]),
        (298, "INFO", _REPORT_IMAGE_COLORS["blue"]),
    ):
        draw.rectangle((x0, legend_y + 3, x0 + 14, legend_y + 15), fill=color)
        write(x0 + 22, legend_y, label, tiny, muted)
    write(
        right,
        legend_y,
        "positions are sample ordinals when timestamps are absent",
        tiny,
        muted,
        anchor="ra",
    )

    # Diagnostic summary panels.
    summary_y = 850
    summary_h = 220
    half_gap = 18
    half_width = (width - 68 - half_gap) // 2
    rect((34, summary_y, 34 + half_width, summary_y + summary_h), panel_bg, 2, line)
    rect(
        (34 + half_width + half_gap, summary_y, width - 34, summary_y + summary_h),
        panel_bg,
        2,
        line,
    )
    write(54, summary_y + 18, "DRIFT SUMMARY", section_font)
    write(54, summary_y + 57, "observed maximum", tiny, muted)
    max_value = _number(metrics.get("max_abs_dlogp"), default=0.0) or 0.0
    bar_x = 54
    bar_y = summary_y + 91
    bar_width = half_width - 110
    rect((bar_x, bar_y, bar_x + bar_width, bar_y + 24), _REPORT_IMAGE_COLORS["track"], 4)
    fill_width = int(min(1.0, max_value / max(max_value, 1.0)) * bar_width) if max_value else 0
    if fill_width:
        rect((bar_x, bar_y, bar_x + max(8, fill_width), bar_y + 24), status_color, 4)
    write(bar_x, bar_y + 30, "0", tiny, muted)
    write(
        bar_x + bar_width,
        bar_y + 30,
        _format_number(max(max_value, 1.0)),
        tiny,
        muted,
        anchor="ra",
    )
    worst = _plain_mapping(normalized.get("worst_token"))
    worst_text = (
        f"worst token: sample={worst.get('sample_position', '-')}  token={worst.get('token_position', '-')}  "
        f"|dlogp|={_format_number(worst.get('abs_dlogp'))}"
    )
    write(54, summary_y + 157, fit(worst_text, 92), mono, text_color)
    right_x = 34 + half_width + half_gap + 20
    write(right_x, summary_y + 18, "SELECTED EVENT", section_font)
    warning_items = _plain_mapping(normalized.get("validation"))
    warnings = warning_items.get("warnings") or []
    failures = warning_items.get("failures") or []
    issue_text = "No anomaly recorded."
    if failures or warnings:
        first = (failures or warnings)[0]
        first = first if isinstance(first, Mapping) else {"message": first}
        issue_text = f"{str(first.get('code', 'validation'))}: {str(first.get('message', ''))}"
    elif worst:
        issue_text = (
            f"worst token: sample={worst.get('sample_position', '-')}  "
            f"token={worst.get('token_position', '-')}  "
            f"|dlogp|={_format_number(worst.get('abs_dlogp'))}"
        )
    write(right_x, summary_y + 57, fit(issue_text, 95), small, status_color)
    write(right_x, summary_y + 101, "worst token / validation marker", tiny, muted)
    write(
        right_x,
        summary_y + 140,
        f"status={badge_text}  warnings={len(warnings)}  failures={len(failures)}",
        mono,
        text_color,
    )

    # Bottom tables: axis capture and runtime provenance are the actionable part
    # of the image for a post-training user.
    table_y = 1095
    table_h = 400
    rect((34, table_y, 34 + half_width, table_y + table_h), panel_bg, 2, line)
    rect(
        (34 + half_width + half_gap, table_y, width - 34, table_y + table_h),
        panel_bg,
        2,
        line,
    )
    write(54, table_y + 18, "CAPTURED EXECUTION AXES", section_font)
    write(
        34 + half_width + half_gap + 20,
        table_y + 18,
        "RUNTIME PROVENANCE",
        section_font,
    )
    axes_rows = key_value_rows(normalized.get("axes"))
    provenance_rows = key_value_rows(normalized.get("runtime_provenance"))

    def table(rows: list[tuple[str, str]], x0: int, y0: int, w: int, max_rows: int = 8) -> None:
        row_y = y0
        for index, (key, value) in enumerate(rows[:max_rows]):
            if index % 2 == 0:
                draw.rectangle((x0, row_y - 3, x0 + w, row_y + 35), fill=panel_alt)
            write(x0 + 14, row_y + 8, key, tiny, muted)
            write(x0 + 270, row_y + 8, value, mono, text_color)
            draw.line((x0, row_y + 38, x0 + w, row_y + 38), fill=line, width=1)
            row_y += 39
        if not rows:
            write(x0 + 14, row_y + 8, "not recorded", small, muted)

    table(axes_rows, 54, table_y + 66, half_width - 40)
    table(provenance_rows, 34 + half_width + half_gap + 20, table_y + 66, half_width - 40)
    validation_label = "VALIDATION: PASS" if status == "pass" else f"VALIDATION: {badge_text}"
    if failures or warnings:
        validation_label = f"VALIDATION: {badge_text} | {fit(issue_text, 86)}"
    write(70, height - 58, validation_label, tiny, status_color)
    footer = "static diagnostic image | schema v" + str(
        normalized.get("schema_version", REPORT_SCHEMA_VERSION)
    )
    write(width - 54, height - 28, footer, tiny, muted, anchor="ra")
    return image


def write_drift_report_image(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a static PNG/JPEG consistency drift report image."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_drift_report_image(report)
    suffix = output.suffix.lower()
    image_format = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    if image_format == "JPEG":
        image.save(output, format=image_format, quality=95, optimize=True)
    else:
        image.save(output, format=image_format, optimize=True)
    return output


def _read_attempt_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cross-config report input: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"cross-config report input must be a JSON object: {path}")
    return _plain_mapping(value)


def _worst_token(
    absolute_diff: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, Any] | None:
    if absolute_diff.shape != active_mask.shape:
        raise ValueError("token_diffs absolute_diff and active_mask must have matching shapes")
    if not bool(active_mask.any()):
        return None
    values = absolute_diff.masked_fill(~active_mask, -1.0)
    flat_index = int(values.reshape(-1).argmax().item())
    coordinates = list(torch.unravel_index(torch.tensor(flat_index), values.shape))
    sample_position = int(coordinates[0].item()) if coordinates else 0
    token_position = int(coordinates[-1].item()) if coordinates else flat_index
    return {
        "sample_position": sample_position,
        "token_position": token_position,
        "abs_dlogp": float(values.reshape(-1)[flat_index].item()),
    }


def _samples_from_identity(
    identity: Mapping[str, Any],
    active_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    logical_identity = _plain_mapping(identity.get("identity"))
    token_ids = logical_identity.get("token_ids")
    samples: list[dict[str, Any]] = []
    for index in range(active_mask.shape[0] if active_mask.ndim else 0):
        row = token_ids[index] if isinstance(token_ids, list) and index < len(token_ids) else None
        samples.append(
            {
                "sample_position": index,
                "sample_index": index,
                "token_count": (len(row) if isinstance(row, list) else int(active_mask.shape[-1])),
                "active_token_count": int(active_mask[index].sum().item()),
            }
        )
    return samples


def _render_key_value_table(values: Any, *, empty: str) -> str:
    if not isinstance(values, Mapping) or not values:
        return f'<div class="empty">{html.escape(empty)}</div>'
    rows = []
    for key in sorted(values, key=str):
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format_value(values[key]))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _render_validation(validation: Any) -> str:
    if not isinstance(validation, Mapping):
        return '<div class="empty">No validation record was captured.</div>'
    warnings = validation.get("warnings") or []
    failures = validation.get("failures") or []
    if not warnings and not failures:
        return '<div class="pass">PASS: no metadata validation warnings or failures.</div>'
    rows = []
    for kind, items in (("failure", failures), ("warning", warnings)):
        for item in items:
            item = item if isinstance(item, Mapping) else {"message": item}
            rows.append(
                f'<tr><td class="{kind}">{kind.upper()}</td><td>{html.escape(str(item.get("code", "")))}</td><td>{html.escape(str(item.get("message", "")))}</td></tr>'
            )
    return (
        "<table><thead><tr><th>Status</th><th>Code</th><th>Message</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _plain_mapping(value: Any) -> dict[str, Any]:
    return (
        {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Mapping)
        else {}
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _backend_label(rollout: Any, training: Any) -> str | None:
    """Show one actual backend only when both execution sides agree."""

    if rollout in (None, "") and training in (None, ""):
        return None
    if rollout in (None, ""):
        return f"training={training}"
    if training in (None, ""):
        return f"rollout={rollout}"
    if rollout == training:
        return str(rollout)
    return f"rollout={rollout}; training={training}"


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: Any, default: str = "-") -> str:
    number = _number(value)
    if number is None:
        return default
    if abs(number) >= 1000 or (abs(number) < 0.001 and number != 0):
        return f"{number:.3e}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(1, limit - 3)] + "..."


def _load_report_font(size: int, *, bold: bool = False, mono: bool = False) -> Any:
    """Load a platform font with deterministic fallbacks for report images."""

    from PIL import ImageFont

    candidates = []
    if mono:
        candidates.extend(
            [
                "C:/Windows/Fonts/consola.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
