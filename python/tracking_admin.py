from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from run_tracking import (
	MILESTONES_FIELDNAMES,
	MODEL_PACKAGE_FILENAME,
	MODEL_PACKAGE_MANIFEST_FILENAME,
	MODEL_PACKAGE_SUMMARY_FILENAME,
	POLICIES_FIELDNAMES,
	RUNS_FIELDNAMES,
	merge_policy_registry_row,
)


def rebuild_tracking_registries(tracking_dir: str | Path) -> dict[str, int]:
	tracking_root = Path(tracking_dir)
	runs_root = tracking_root / "runs"
	manifests = sorted(runs_root.glob("*/manifest.json")) if runs_root.exists() else []
	run_rows: list[dict[str, Any]] = []
	policy_rows_by_id: dict[str, dict[str, Any]] = {}
	milestone_rows: list[dict[str, Any]] = []

	loaded_manifests = [load_tracked_run_manifest(manifest_path) for manifest_path in manifests]
	loaded_manifests.sort(key=_manifest_sort_key)

	for manifest in loaded_manifests:
		run_rows.append(_build_run_row(manifest))
		policy_row = _build_policy_row(manifest)
		if policy_row is not None:
			policy_id = str(policy_row.get("policy_id", ""))
			policy_rows_by_id[policy_id] = merge_policy_registry_row(
				policy_rows_by_id.get(policy_id),
				policy_row,
				run_mode=str((manifest.get("run") or {}).get("mode", "")),
			)
		milestone_rows.extend(_build_milestone_rows(manifest))

	policy_rows = list(policy_rows_by_id.values())

	_write_csv_rows(tracking_root / "runs.csv", RUNS_FIELDNAMES, run_rows)
	_write_csv_rows(tracking_root / "policies.csv", POLICIES_FIELDNAMES, policy_rows)
	_write_csv_rows(tracking_root / "milestones.csv", MILESTONES_FIELDNAMES, milestone_rows)
	return {
		"run_count": len(run_rows),
		"policy_count": len(policy_rows),
		"milestone_count": len(milestone_rows),
	}


def promote_run(
	run_reference: str | Path,
	*,
	tracking_dir: str | Path,
	policy_updates: Mapping[str, Any],
) -> dict[str, Any]:
	manifest_path = resolve_run_manifest_path(run_reference, tracking_dir)
	manifest = load_tracked_run_manifest(manifest_path)
	policy = dict(manifest.get("policy") or {})

	for key, value in policy_updates.items():
		if key == "is_best_in_category":
			if value:
				policy[key] = True
			continue
		if value is None:
			continue
		if isinstance(value, str) and value == "":
			continue
		policy[key] = value

	if not str(policy.get("policy_id", "")).strip():
		raise ValueError("Promoting a run requires a non-empty policy_id")

	if not str(policy.get("label", "")).strip():
		policy["label"] = str((manifest.get("run") or {}).get("label", ""))
	if not str(policy.get("algorithm", "")).strip():
		policy["algorithm"] = _infer_algorithm_from_manifest(manifest)

	manifest["policy"] = _normalize_policy_block(policy, fallback_label=str((manifest.get("run") or {}).get("label", "")))
	_write_json(manifest_path, manifest)
	_summary_path = Path((manifest.get("paths") or {}).get("summary_path") or manifest_path.with_name("summary.json"))
	if _summary_path.exists() and manifest.get("summary"):
		_write_json(_summary_path, manifest["summary"])
	_write_policy_package_metadata(manifest)
	counts = rebuild_tracking_registries(tracking_dir)
	return {
		"run_id": str((manifest.get("run") or {}).get("run_id", "")),
		"policy_id": str((manifest.get("policy") or {}).get("policy_id", "")),
		"manifest_path": manifest_path.as_posix(),
		**counts,
	}


def resolve_run_manifest_path(run_reference: str | Path, tracking_dir: str | Path) -> Path:
	reference = Path(run_reference)
	candidates: list[Path] = []
	if reference.exists():
		candidates.append(reference / "manifest.json" if reference.is_dir() else reference)
	tracking_manifest = Path(tracking_dir) / "runs" / str(run_reference) / "manifest.json"
	candidates.append(tracking_manifest)
	for candidate in candidates:
		if candidate.exists() and candidate.is_file():
			return candidate
	raise FileNotFoundError(f"Could not resolve run manifest for {run_reference}")


def load_tracked_run_manifest(manifest_path: str | Path) -> dict[str, Any]:
	manifest_file = Path(manifest_path)
	manifest = _load_json(manifest_file)
	run_dir = manifest_file.parent
	tracking_dir = run_dir.parent.parent
	run_block = dict(manifest.get("run") or {})
	run_block.setdefault("run_id", run_dir.name)
	run_block.setdefault("label", run_dir.name)
	run_block.setdefault("command", "")
	run_block.setdefault("status", "completed")
	run_block.setdefault("notes", "")
	manifest["run"] = run_block

	policy_block = _normalize_policy_block(
		manifest.get("policy") or {},
		fallback_label=str(run_block.get("label", run_dir.name)),
	)
	manifest["policy"] = policy_block

	paths = dict(manifest.get("paths") or {})
	paths["tracking_dir"] = tracking_dir.as_posix()
	paths["run_dir"] = run_dir.as_posix()
	paths["manifest_path"] = manifest_file.as_posix()
	summary_path = Path(paths.get("summary_path") or (run_dir / "summary.json"))
	paths["summary_path"] = summary_path.as_posix()
	paths.setdefault("training_log_jsonl", "")
	paths.setdefault("step_log_jsonl", "")
	paths.setdefault("model_path", "")
	paths.setdefault("model_artifact_path", _derive_model_artifact_path(paths.get("model_path", "")))
	manifest["paths"] = paths

	summary = dict(manifest.get("summary") or {})
	if not summary and summary_path.exists():
		summary = _load_json(summary_path)
	manifest["summary"] = summary
	manifest["milestones"] = list(manifest.get("milestones") or [])
	manifest["environment"] = dict(manifest.get("environment") or {})
	return manifest


def write_tracking_plan_docs(workspace_root: str | Path) -> list[Path]:
	root = Path(workspace_root)
	docs_dir = root / "docs"
	docs_dir.mkdir(parents=True, exist_ok=True)
	policy_plan = docs_dir / "policy_metadata_plan.md"
	milestone_plan = docs_dir / "milestone_coverage_plan.md"
	policy_plan.write_text(POLICY_METADATA_PLAN_TEXT, encoding="utf-8")
	milestone_plan.write_text(MILESTONE_COVERAGE_PLAN_TEXT, encoding="utf-8")
	return [policy_plan, milestone_plan]


def _build_run_row(manifest: Mapping[str, Any]) -> dict[str, Any]:
	run = dict(manifest.get("run") or {})
	policy = _normalize_policy_block(manifest.get("policy") or {}, fallback_label=str(run.get("label", "")))
	paths = dict(manifest.get("paths") or {})
	summary = dict(manifest.get("summary") or {})
	environment = dict(manifest.get("environment") or {})
	return {
		"run_id": run.get("run_id", ""),
		"mode": run.get("mode", ""),
		"status": run.get("status", ""),
		"label": run.get("label", ""),
		"persona": policy.get("persona", ""),
		"training_technique": policy.get("training_technique", ""),
		"policy_id": policy.get("policy_id", ""),
		"started_at": run.get("started_at", ""),
		"finished_at": run.get("finished_at", ""),
		"duration_seconds": run.get("duration_seconds", ""),
		"run_dir": paths.get("run_dir", ""),
		"manifest_path": paths.get("manifest_path", ""),
		"summary_path": paths.get("summary_path", ""),
		"model_path": paths.get("model_path", ""),
		"model_artifact_path": paths.get("model_artifact_path", ""),
		"training_log_jsonl": paths.get("training_log_jsonl", ""),
		"step_log_jsonl": paths.get("step_log_jsonl", ""),
		"environment_fingerprint": environment.get("environment_fingerprint", ""),
		"reward_config_hash": environment.get("reward_config_hash", ""),
		"episode_count": summary.get("episode_count", ""),
		"completed_episode_count": summary.get("completed_episode_count", ""),
		"success_count": summary.get("success_count", ""),
		"success_rate": summary.get("success_rate", ""),
		"mean_episode_reward": summary.get("mean_episode_reward", ""),
		"median_episode_reward": summary.get("median_episode_reward", ""),
		"mean_goal_steps": summary.get("mean_goal_steps", ""),
		"median_goal_steps": summary.get("median_goal_steps", ""),
		"mean_goal_frames": summary.get("mean_goal_frames", ""),
		"median_goal_frames": summary.get("median_goal_frames", ""),
		"mean_goal_timesteps": summary.get("mean_goal_timesteps", ""),
		"median_goal_timesteps": summary.get("median_goal_timesteps", ""),
		"first_success_episode": summary.get("first_success_episode", ""),
		"first_success_timestep": summary.get("first_success_timestep", ""),
		"timesteps_completed": summary.get("timesteps_completed", ""),
		"terminal_reason_counts": summary.get("terminal_reason_counts", {}),
		"notes": run.get("notes", ""),
	}


def _build_policy_row(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
	run = dict(manifest.get("run") or {})
	policy = _normalize_policy_block(manifest.get("policy") or {}, fallback_label=str(run.get("label", "")))
	paths = dict(manifest.get("paths") or {})
	summary = dict(manifest.get("summary") or {})
	environment = dict(manifest.get("environment") or {})
	policy_id = str(policy.get("policy_id", ""))
	model_path = str(paths.get("model_path", ""))
	if not policy_id or not model_path:
		return None
	return {
		"policy_id": policy_id,
		"source_run_id": run.get("run_id", ""),
		"source_run_finished_at": run.get("finished_at", ""),
		"label": policy.get("label", ""),
		"persona": policy.get("persona", ""),
		"objective": policy.get("objective", ""),
		"intended_use": policy.get("intended_use", ""),
		"algorithm": policy.get("algorithm", ""),
		"training_technique": policy.get("training_technique", ""),
		"is_best_in_category": policy.get("is_best_in_category", False),
		"notes": policy.get("notes", ""),
		"manifest_path": _resolve_model_manifest_output_path(model_path, paths.get("model_artifact_path", "")),
		"model_path": model_path,
		"model_artifact_path": paths.get("model_artifact_path", ""),
		"environment_fingerprint": environment.get("environment_fingerprint", ""),
		"reward_config_hash": environment.get("reward_config_hash", ""),
		"timesteps_completed": summary.get("timesteps_completed", ""),
		"success_rate": summary.get("success_rate", ""),
		"mean_episode_reward": summary.get("mean_episode_reward", ""),
		"mean_goal_steps": summary.get("mean_goal_steps", ""),
		"median_goal_steps": summary.get("median_goal_steps", ""),
		"median_goal_timesteps": summary.get("median_goal_timesteps", ""),
		"latest_evaluation_at": run.get("finished_at", "") if str(run.get("mode", "")) == "evaluation" else "",
		"updated_at": run.get("finished_at", ""),
	}


def _manifest_sort_key(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
	run = dict(manifest.get("run") or {})
	return (
		str(run.get("finished_at", "") or ""),
		str(run.get("started_at", "") or ""),
		str(run.get("run_id", "") or ""),
	)


def _build_milestone_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
	run = dict(manifest.get("run") or {})
	mode = str(run.get("mode", ""))
	run_id = str(run.get("run_id", ""))
	rows: list[dict[str, Any]] = []
	for milestone in list(manifest.get("milestones") or []):
		milestone_row = dict(milestone)
		milestone_row.setdefault("run_id", run_id)
		milestone_row.setdefault("mode", mode)
		rows.append({field: milestone_row.get(field, "") for field in MILESTONES_FIELDNAMES})
	return rows


def _normalize_policy_block(policy: Mapping[str, Any], *, fallback_label: str) -> dict[str, Any]:
	data = dict(policy or {})
	return {
		"policy_id": str(data.get("policy_id", "") or ""),
		"label": str(data.get("label", "") or fallback_label),
		"persona": str(data.get("persona", "") or ""),
		"objective": str(data.get("objective", "") or ""),
		"intended_use": str(data.get("intended_use", "") or ""),
		"algorithm": str(data.get("algorithm", "") or ""),
		"training_technique": str(data.get("training_technique", "") or ""),
		"is_best_in_category": bool(data.get("is_best_in_category", False)),
		"notes": str(data.get("notes", "") or ""),
	}


def _infer_algorithm_from_manifest(manifest: Mapping[str, Any]) -> str:
	run = dict(manifest.get("run") or {})
	if run.get("mode") == "smoke":
		return "random_policy"
	training_technique = str((manifest.get("policy") or {}).get("training_technique", "")).lower()
	if training_technique.startswith("ppo"):
		return "PPO"
		
	args = dict(manifest.get("args") or {})
	if args.get("eval_model"):
		return "PPO"
	if args.get("random_only"):
		return "random_policy"
	return "PPO"


def _derive_model_artifact_path(model_path: str) -> str:
	if not model_path:
		return ""
	path = Path(model_path)
	if path.suffix == ".zip":
		return path.as_posix()
	return (path / MODEL_PACKAGE_FILENAME).as_posix()


def _resolve_model_manifest_output_path(model_path: str | Path, model_artifact_path: str | Path | None = None) -> str:
	path = Path(model_path) if model_path else Path()
	artifact_path = Path(model_artifact_path) if model_artifact_path else Path()
	if path and (path.is_dir() or path.suffix == ""):
		return (path / MODEL_PACKAGE_MANIFEST_FILENAME).as_posix()
	if artifact_path and artifact_path.name == MODEL_PACKAGE_FILENAME:
		return (artifact_path.parent / MODEL_PACKAGE_MANIFEST_FILENAME).as_posix()
	if path:
		return path.with_suffix(".manifest.json").as_posix()
	return ""


def _write_policy_package_metadata(manifest: Mapping[str, Any]) -> None:
	paths = dict(manifest.get("paths") or {})
	model_path = str(paths.get("model_path", ""))
	if not model_path:
		return
	model_artifact_path = str(paths.get("model_artifact_path", ""))
	manifest_output_path = Path(_resolve_model_manifest_output_path(model_path, model_artifact_path))
	if not manifest_output_path:
		return
	if manifest_output_path.name == MODEL_PACKAGE_MANIFEST_FILENAME:
		summary_output_path = manifest_output_path.parent / MODEL_PACKAGE_SUMMARY_FILENAME
	else:
		summary_output_path = Path(model_path).with_suffix(".summary.json")
	_write_json(manifest_output_path, manifest)
	if manifest.get("summary"):
		_write_json(summary_output_path, manifest["summary"])


def _load_json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> str | int | float:
	if value is None:
		return ""
	if isinstance(value, bool):
		return "true" if value else "false"
	if isinstance(value, (int, float)):
		return value
	if isinstance(value, (dict, list, tuple)):
		return json.dumps(value, sort_keys=True)
	return str(value)


POLICY_METADATA_PLAN_TEXT = """# Policy-Level Metadata Plan

## Goal

Promote policy metadata from a handful of tags into a deliberate catalog schema so policies can be compared, curated, and selected without reading raw manifests.

## Increment 1: Core Catalog Fields

- Keep `policy_id` as the stable curated identifier.
- Add first-class fields for `label`, `persona`, `objective`, `intended_use`, `algorithm`, `training_technique`, `is_best_in_category`, and `notes`.
- Continue storing environment fingerprint and reward hash alongside the policy so selection always carries compatibility context.

## Increment 2: Promotion Workflow

- Add a `promote-run` command that can take an existing tracked run and curate it into the policy catalog.
- Promotion should update both the run manifest and the policy package manifest so the catalog and the package stay aligned.
- Promotion should be idempotent: promoting the same `policy_id` again should replace the policy row rather than duplicate it.

## Increment 3: Selection Metadata

- Add `is_best_in_category` as the first explicit selection field.
- Next additions should be `status` values like `candidate`, `baseline`, `archived`, and `production`.
- After that, add policy lineage fields such as `replaces_policy_id` and `parent_policy_id` if version chains become important.

## Increment 4: Comparative Metadata

- Promote a small set of derived metrics into policy-facing views, not just run-facing views.
- Candidate fields: current best success rate, median goal steps, median goal timesteps, mean episode reward, and latest evaluation timestamp.
- Keep derived metrics separate from operator-authored metadata so rebuilding registries remains deterministic.

## Increment 5: Workflow Documentation

- Document how a run becomes a policy.
- Document when to mint a new `policy_id` versus updating an existing one.
- Document how to mark a policy as best in category and what evidence is expected.
"""


MILESTONE_COVERAGE_PLAN_TEXT = """# Milestone Coverage Plan

## Goal

Expand milestone tracking from a narrow success-rate view into a comparison framework that supports safe, efficient, and aggressive policy families.

## Increment 1: Completion Milestones

- Keep `first_goal_reached`.
- Add threshold milestones for `median_goal_steps` and `median_goal_timesteps` once enough successful episodes exist.
- Add `first_success_streak` style milestones only after the base success metrics are stable.

## Increment 2: Safety Milestones

- Add thresholds for reduced `out_of_bounds` frequency.
- Add thresholds for reduced high-speed goal entries or unsafe terminal conditions once those metrics exist in summaries.
- Treat safety milestones as separate from performance milestones so a policy can improve in one axis without hiding regressions in the other.

## Increment 3: Efficiency Milestones

- Add milestones for lower median steps-to-goal.
- Add milestones for lower median timesteps-to-goal.
- Add fuel or thrust efficiency milestones once per-episode thrust totals are exposed cleanly in summary data.

## Increment 4: Persona-Aware Milestone Sets

- Safe policies should prioritize safety milestones first, then completion milestones.
- Efficient policies should prioritize steps, timesteps, and thrust-efficiency milestones.
- Aggressive policies should prioritize time-to-goal or speed-oriented milestones, but still surface safety regressions explicitly.

## Increment 5: Registry Presentation

- Keep milestone events in `experiments/milestones.csv`.
- Add a derived summary view later that shows the latest achieved milestone per category for each run or policy.
- Do not overload one milestone table with policy curation state; keep curation in the policy registry.
"""