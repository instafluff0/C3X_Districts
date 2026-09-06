#!/usr/bin/env python3
"""Validate the renderer roadmap handoff contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = RENDERER_ROOT / "project_status.json"
VALID_STATUSES = {
    "complete",
    "in_progress",
    "ready",
    "blocked_by_previous",
    "blocked_by_lab",
    "planned",
}
PATCH_REQUEST_FIELDS = {
    "milestone_step",
    "symbol",
    "reason",
    "capability",
    "signature",
    "supported_build_addresses",
    "fallback",
    "verification",
}


def validate_project_state(status_path: Path = DEFAULT_STATUS) -> list[str]:
    errors: list[str] = []
    try:
        status: dict[str, Any] = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Could not read project status: {exc}"]

    if status.get("schema") != "c3x.renderer_project_status.v0":
        errors.append("Unsupported or missing project status schema.")

    for doc in status.get("required_docs", []):
        if not (RENDERER_ROOT / doc).is_file():
            errors.append(f"Required project document is missing: {doc}")

    patch_dependencies = status.get("civ3_patch_dependencies")
    if not isinstance(patch_dependencies, dict):
        errors.append("civ3_patch_dependencies must be a machine-readable object.")
    else:
        ledger = patch_dependencies.get("ledger")
        if not isinstance(ledger, str) or not ledger:
            errors.append("civ3_patch_dependencies must name its ledger document.")
        else:
            if ledger not in status.get("required_docs", []):
                errors.append("The Civ III patch dependency ledger must be a required document.")
            if not (RENDERER_ROOT / ledger).is_file():
                errors.append(f"Civ III patch dependency ledger is missing: {ledger}")
        actions = patch_dependencies.get("required_user_action")
        if not isinstance(actions, list):
            errors.append("civ3_patch_dependencies.required_user_action must be a list.")
        else:
            for index, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"Civ III patch request {index} must be an object.")
                    continue
                missing = sorted(PATCH_REQUEST_FIELDS - set(action))
                if missing:
                    errors.append(
                        f"Civ III patch request {index} is missing fields: {', '.join(missing)}"
                    )
                addresses = action.get("supported_build_addresses")
                if not isinstance(addresses, dict) or not addresses:
                    errors.append(
                        f"Civ III patch request {index} must include supported-build addresses."
                    )
        if not patch_dependencies.get("reporting_rule"):
            errors.append("civ3_patch_dependencies must include its user reporting rule.")

    steps: dict[str, dict[str, Any]] = {}
    ready_ids: list[str] = []
    milestone_ids: set[str] = set()
    for milestone in status.get("milestones", []):
        milestone_id = milestone.get("id")
        if not milestone_id or milestone_id in milestone_ids:
            errors.append(f"Missing or duplicate milestone ID: {milestone_id!r}")
        else:
            milestone_ids.add(milestone_id)
        milestone_status = milestone.get("status")
        if milestone_status not in VALID_STATUSES:
            errors.append(f"Invalid status for {milestone_id}: {milestone_status!r}")
        if milestone_status == "complete":
            if not milestone.get("evidence"):
                errors.append(f"Completed milestone {milestone_id} has no evidence.")
            if not milestone.get("verification"):
                errors.append(f"Completed milestone {milestone_id} has no portable verification gates.")
            for evidence in milestone.get("evidence", []):
                if not (RENDERER_ROOT / evidence).exists():
                    errors.append(f"Missing completion evidence for {milestone_id}: {evidence}")

        for step in milestone.get("steps", []):
            step_id = step.get("id")
            if not step_id or step_id in steps:
                errors.append(f"Missing or duplicate step ID: {step_id!r}")
                continue
            steps[step_id] = step
            step_status = step.get("status")
            if step_status not in VALID_STATUSES:
                errors.append(f"Invalid status for {step_id}: {step_status!r}")
            if step_status == "ready":
                ready_ids.append(step_id)
            if step_status == "complete":
                if not step.get("evidence"):
                    errors.append(f"Completed step {step_id} has no evidence.")
                if not step.get("verification"):
                    errors.append(f"Completed step {step_id} has no portable verification gates.")
                for evidence in step.get("evidence", []):
                    if not (RENDERER_ROOT / evidence).exists():
                        errors.append(f"Missing completion evidence for {step_id}: {evidence}")

    current_milestone = status.get("current_milestone")
    if current_milestone not in milestone_ids:
        errors.append(f"current_milestone does not exist: {current_milestone!r}")

    next_step = status.get("next_step", {})
    next_step_id = next_step.get("id")
    if next_step.get("status") != "ready":
        errors.append("next_step must be marked ready.")
    if next_step_id not in steps:
        errors.append(f"next_step does not exist in milestone steps: {next_step_id!r}")
    elif steps[next_step_id].get("status") != "ready":
        errors.append(f"Milestone step {next_step_id} is not marked ready.")
    elif steps[next_step_id].get("title") != next_step.get("title"):
        errors.append(f"next_step title does not match milestone step {next_step_id}.")
    if ready_ids != [next_step_id]:
        errors.append(f"Exactly the declared next_step must be ready; found {ready_ids!r}.")
    for field in ("scope", "acceptance", "must_not"):
        if not next_step.get(field):
            errors.append(f"next_step must include nonempty {field}.")

    if next_step.get("workstream") == "game_integration":
        handoffs = next_step.get("lab_handoffs")
        if not isinstance(handoffs, list) or not handoffs:
            errors.append("A game-integration next step must name at least one lab handoff record.")
        else:
            for index, handoff in enumerate(handoffs):
                if not isinstance(handoff, dict) or not handoff.get("gate") or not handoff.get("record"):
                    errors.append(f"Lab handoff {index} must name a gate and record.")
                    continue
                handoff_path = RENDERER_ROOT / handoff["record"]
                try:
                    record = json.loads(handoff_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"Could not read lab handoff {handoff['record']}: {exc}")
                    continue
                if record.get("schema") != "c3x.renderer_lab_handoff.v0":
                    errors.append(f"Lab handoff {handoff['record']} has an unsupported schema.")
                if record.get("lab_gate") != handoff["gate"] or record.get("status") != "approved":
                    errors.append(
                        f"Lab handoff {handoff['record']} must be approved for {handoff['gate']}."
                    )
                for field in ("source_contract", "coverage", "reference", "ownership_intent", "cache_inputs", "fallback"):
                    if not record.get(field):
                        errors.append(f"Lab handoff {handoff['record']} is missing {field}.")

    if isinstance(patch_dependencies, dict):
        current_symbols = patch_dependencies.get("current_step_existing_symbols")
        audit_candidates = patch_dependencies.get("current_step_audit_candidates")
        if not isinstance(current_symbols, list):
            errors.append("civ3_patch_dependencies must list current-step existing symbols.")
        if not isinstance(audit_candidates, list):
            errors.append("civ3_patch_dependencies must list current-step audit candidates.")

    workstreams = status.get("workstreams")
    if not isinstance(workstreams, dict):
        errors.append("workstreams must define Renderer Lab and Game Integration.")
    else:
        for stream_id in ("renderer_lab", "game_integration"):
            stream = workstreams.get(stream_id)
            if not isinstance(stream, dict):
                errors.append(f"workstreams.{stream_id} must be an object.")
                continue
            current_step_id = stream.get("current_step")
            if current_step_id not in steps:
                errors.append(
                    f"workstreams.{stream_id}.current_step does not exist: {current_step_id!r}"
                )
            command = stream.get("iteration_command")
            if not isinstance(command, str) or "renderer_dev.py" not in command:
                errors.append(
                    f"workstreams.{stream_id} must name its renderer_dev.py iteration command."
                )
        vm = workstreams.get("windows_vm")
        if not isinstance(vm, dict) or not vm.get("name") or not vm.get("shared_repository"):
            errors.append("workstreams.windows_vm must name the native execution host and shared repository.")
        handoff = workstreams.get("handoff_contract")
        if handoff not in status.get("required_docs", []):
            errors.append("The renderer workstream handoff contract must be a required document.")

    milestone_by_id = {
        milestone.get("id"): milestone for milestone in status.get("milestones", [])
    }
    lab = milestone_by_id.get("LAB", {})
    integration = milestone_by_id.get("INTEGRATION", {})
    integration_steps = {
        step.get("lab_gate"): step for step in integration.get("steps", [])
    }
    for lab_step in lab.get("steps", []):
        lab_id = lab_step.get("id", "")
        if not lab_id.startswith("L"):
            continue
        expected_id = "I" + lab_id[1:]
        paired = integration_steps.get(lab_id)
        if not paired or paired.get("id") != expected_id:
            errors.append(f"Lab gate {lab_id} must have same-numbered integration gate {expected_id}.")
            continue
        if paired.get("status") == "complete" and lab_step.get("status") != "complete":
            errors.append(f"Integration gate {expected_id} cannot complete before {lab_id} approval.")

    roadmap_path = RENDERER_ROOT / "ROADMAP.md"
    if roadmap_path.is_file() and next_step_id:
        roadmap_text = roadmap_path.read_text(encoding="utf-8")
        if f"## Next Step: {next_step_id}" not in roadmap_text:
            errors.append(f"ROADMAP.md does not declare the JSON next step {next_step_id}.")

    return errors


def main() -> int:
    status_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STATUS
    errors = validate_project_state(status_path)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    status = json.loads(status_path.read_text(encoding="utf-8"))
    next_step = status["next_step"]
    print(f"Renderer project state is valid. Next step: {next_step['id']} - {next_step['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
