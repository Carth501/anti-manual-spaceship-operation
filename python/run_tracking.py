from __future__ import annotations

import csv
import json
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


MODEL_PACKAGE_FILENAME = "policy.zip"
MODEL_PACKAGE_MANIFEST_FILENAME = "manifest.json"
MODEL_PACKAGE_SUMMARY_FILENAME = "summary.json"


RUNS_FIELDNAMES = [
	"run_id",
	"mode",
	"status",
	"label",
	"persona",
	"training_technique",
	"policy_id",
	"started_at",
	"finished_at",
	"duration_seconds",
	"run_dir",
	"manifest_path",
	"summary_path",
	"model_path",
	"model_artifact_path",
	"training_log_jsonl",
	"step_log_jsonl",
	"environment_fingerprint",
	"reward_config_hash",
	"episode_count",
	"completed_episode_count",
	"success_count",
	"success_rate",
	"mean_episode_reward",
	"median_episode_reward",
	"mean_goal_steps",
	"median_goal_steps",
	"mean_goal_frames",
	"median_goal_frames",
	"mean_goal_timesteps",
	"median_goal_timesteps",
	"first_success_episode",
	"first_success_timestep",
	"timesteps_completed",
	"terminal_reason_counts",
	"notes",
]

MILESTONES_FIELDNAMES = [
	"run_id",
	"mode",
	"milestone_key",
	"milestone_type",
	"threshold_value",
	"episode",
	"timesteps",
	"current_value",
	"achieved_at",
]

POLICIES_FIELDNAMES = [
	"policy_id",
	"source_run_id",
	"source_run_finished_at",
	"label",
	"persona",
	"objective",
	"intended_use",
	"algorithm",
	"training_technique",
	"is_best_in_category",
	"notes",
	"manifest_path",
	"model_path",
	"model_artifact_path",
	"environment_fingerprint",
	"reward_config_hash",
	"timesteps_completed",
	"success_rate",
	"mean_episode_reward",
	"mean_goal_steps",
	"median_goal_steps",
	"median_goal_timesteps",
	"latest_evaluation_at",
	"updated_at",
]

POLICY_REGISTRY_STABLE_FIELDS = (
	"label",
	"persona",
	"objective",
	"intended_use",
	"algorithm",
	"training_technique",
	"is_best_in_category",
	"notes",
	"manifest_path",
	"model_path",
	"model_artifact_path",
	"environment_fingerprint",
	"reward_config_hash",
)

POLICY_REGISTRY_COMPARISON_FIELDS = (
	"timesteps_completed",
	"success_rate",
	"mean_episode_reward",
	"mean_goal_steps",
	"median_goal_steps",
	"median_goal_timesteps",
)


class EpisodeAccumulator:
	def __init__(self, success_rate_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)) -> None:
		self.success_rate_thresholds = success_rate_thresholds
		self.records: list[dict[str, Any]] = []
		self.terminal_reason_counts: Counter[str] = Counter()
		self.completed_episode_count = 0
		self.success_count = 0
		self._milestones: dict[str, dict[str, Any]] = {}

	def record_episode(
		self,
		*,
		episode: int,
		steps: int,
		frames: int | None,
		reward_total: float,
		terminal_reason: str,
		success: bool,
		timesteps: int | None = None,
		completed_episode: bool = True,
	) -> None:
		reason = terminal_reason or "unknown"
		record = {
			"episode": int(episode),
			"steps": int(steps),
			"frames": int(frames) if frames is not None else None,
			"reward_total": float(reward_total),
			"terminal_reason": reason,
			"success": bool(success),
			"timesteps": int(timesteps) if timesteps is not None else None,
			"completed_episode": bool(completed_episode),
			"recorded_at": utc_now_iso(),
		}
		self.records.append(record)
		self.terminal_reason_counts[reason] += 1

		if not completed_episode:
			return

		self.completed_episode_count += 1
		if success:
			self.success_count += 1
			if "first_goal_reached" not in self._milestones:
				self._milestones["first_goal_reached"] = {
					"milestone_key": "first_goal_reached",
					"milestone_type": "event",
					"threshold_value": 1.0,
					"episode": int(episode),
					"timesteps": int(timesteps) if timesteps is not None else None,
					"current_value": 1.0,
					"achieved_at": record["recorded_at"],
				}

		success_rate = self.success_count / max(self.completed_episode_count, 1)
		for threshold in self.success_rate_thresholds:
			milestone_key = f"success_rate_gte_{int(threshold * 100):02d}"
			if milestone_key in self._milestones or success_rate < threshold:
				continue
			self._milestones[milestone_key] = {
				"milestone_key": milestone_key,
				"milestone_type": "success_rate",
				"threshold_value": threshold,
				"episode": int(episode),
				"timesteps": int(timesteps) if timesteps is not None else None,
				"current_value": round(success_rate, 6),
				"achieved_at": record["recorded_at"],
			}

	def milestone_rows(self, *, run_id: str, mode: str) -> list[dict[str, Any]]:
		rows: list[dict[str, Any]] = []
		for milestone in self._milestones.values():
			rows.append({
				"run_id": run_id,
				"mode": mode,
				**milestone,
			})
		return sorted(rows, key=lambda row: (str(row["achieved_at"]), str(row["milestone_key"])))

	def build_summary(self, *, total_timesteps: int | None = None, wall_seconds: float | None = None) -> dict[str, Any]:
		completed_records = [record for record in self.records if record["completed_episode"]]
		success_records = [record for record in completed_records if record["success"]]
		rewards = [float(record["reward_total"]) for record in completed_records]
		goal_steps = [int(record["steps"]) for record in success_records]
		goal_frames = [int(record["frames"]) for record in success_records if record["frames"] is not None]
		goal_timesteps = [int(record["timesteps"]) for record in success_records if record["timesteps"] is not None]
		first_success = self._milestones.get("first_goal_reached", {})
		latest_timestep = total_timesteps
		if latest_timestep is None:
			latest_timesteps = [int(record["timesteps"]) for record in self.records if record["timesteps"] is not None]
			latest_timestep = max(latest_timesteps) if latest_timesteps else None

		return {
			"episode_count": len(self.records),
			"completed_episode_count": len(completed_records),
			"success_count": len(success_records),
			"success_rate": _safe_ratio(len(success_records), len(completed_records)),
			"mean_episode_reward": _safe_mean(rewards),
			"median_episode_reward": _safe_median(rewards),
			"mean_goal_steps": _safe_mean(goal_steps),
			"median_goal_steps": _safe_median(goal_steps),
			"mean_goal_frames": _safe_mean(goal_frames),
			"median_goal_frames": _safe_median(goal_frames),
			"mean_goal_timesteps": _safe_mean(goal_timesteps),
			"median_goal_timesteps": _safe_median(goal_timesteps),
			"first_success_episode": first_success.get("episode"),
			"first_success_timestep": first_success.get("timesteps"),
			"timesteps_completed": latest_timestep,
			"terminal_reason_counts": dict(self.terminal_reason_counts),
			"milestone_count": len(self._milestones),
			"wall_seconds": round(float(wall_seconds), 3) if wall_seconds is not None else None,
		}


class RunTracker:
	def __init__(
		self,
		*,
		tracking_dir: str | Path,
		mode: str,
		args: Mapping[str, Any],
		environment: Mapping[str, Any],
		run_label: str | None = None,
		persona: str | None = None,
		training_technique: str | None = None,
		policy_id: str | None = None,
		run_notes: str | None = None,
		policy_label: str | None = None,
		policy_objective: str | None = None,
		policy_intended_use: str | None = None,
		policy_algorithm: str | None = None,
		policy_best_in_category: bool = False,
		policy_notes: str | None = None,
	) -> None:
		self.tracking_dir = Path(tracking_dir)
		self.mode = mode
		self.args = dict(args)
		self.environment = json.loads(json.dumps(environment))
		self.run_label = run_label or mode
		self.persona = persona or ""
		self.training_technique = training_technique or ""
		self.policy_id = policy_id or ""
		self.run_notes = run_notes or ""
		self.policy_label = policy_label or self.run_label
		self.policy_objective = policy_objective or ""
		self.policy_intended_use = policy_intended_use or ""
		self.policy_algorithm = policy_algorithm or ""
		self.policy_best_in_category = bool(policy_best_in_category)
		self.policy_notes = policy_notes or self.run_notes
		self.run_id = build_run_id(self.run_label)
		self.run_dir = self.tracking_dir / "runs" / self.run_id
		self.manifest_path = self.run_dir / "manifest.json"
		self.summary_path = self.run_dir / "summary.json"
		self.started_at = utc_now_iso()
		self._started_monotonic = time.monotonic()
		self.episodes = EpisodeAccumulator()
		self._finalized = False

		self.run_dir.mkdir(parents=True, exist_ok=True)
		self._ensure_registry_files()
		self.manifest: dict[str, Any] = {
			"schema_version": 1,
			"run": {
				"run_id": self.run_id,
				"mode": self.mode,
				"label": self.run_label,
				"started_at": self.started_at,
				"notes": self.run_notes,
				"status": "running",
			},
			"policy": {
				"policy_id": self.policy_id,
				"label": self.policy_label,
				"persona": self.persona,
				"objective": self.policy_objective,
				"intended_use": self.policy_intended_use,
				"algorithm": self.policy_algorithm,
				"training_technique": self.training_technique,
				"is_best_in_category": self.policy_best_in_category,
				"notes": self.policy_notes,
			},
			"paths": {
				"tracking_dir": self.tracking_dir.as_posix(),
				"run_dir": self.run_dir.as_posix(),
				"manifest_path": self.manifest_path.as_posix(),
				"summary_path": self.summary_path.as_posix(),
				"model_path": "",
				"model_artifact_path": "",
				"training_log_jsonl": "",
				"step_log_jsonl": "",
			},
			"environment": self.environment,
			"args": self.args,
			"summary": {},
			"milestones": [],
		}
		self._write_json(self.manifest_path, self.manifest)

	def header_record(self) -> dict[str, Any]:
		return {
			"record_type": "run_metadata",
			"run": self.manifest["run"],
			"policy": self.manifest["policy"],
			"paths": self.manifest["paths"],
			"environment": self.environment,
			"args": self.args,
		}

	def attach_paths(
		self,
		*,
		model_path: str | Path | None = None,
		model_artifact_path: str | Path | None = None,
		training_log_path: str | Path | None = None,
		step_log_path: str | Path | None = None,
	) -> None:
		if model_path is not None:
			self.manifest["paths"]["model_path"] = Path(model_path).as_posix()
		if model_artifact_path is not None:
			self.manifest["paths"]["model_artifact_path"] = Path(model_artifact_path).as_posix()
		if training_log_path is not None:
			self.manifest["paths"]["training_log_jsonl"] = Path(training_log_path).as_posix()
		if step_log_path is not None:
			self.manifest["paths"]["step_log_jsonl"] = Path(step_log_path).as_posix()
		self._write_json(self.manifest_path, self.manifest)

	def record_episode(
		self,
		*,
		episode: int,
		steps: int,
		frames: int | None,
		reward_total: float,
		terminal_reason: str,
		success: bool,
		timesteps: int | None = None,
		completed_episode: bool = True,
	) -> None:
		self.episodes.record_episode(
			episode=episode,
			steps=steps,
			frames=frames,
			reward_total=reward_total,
			terminal_reason=terminal_reason,
			success=success,
			timesteps=timesteps,
			completed_episode=completed_episode,
		)

	def finalize(
		self,
		*,
		status: str,
		total_timesteps: int | None = None,
		model_path: str | Path | None = None,
		model_artifact_path: str | Path | None = None,
		training_log_path: str | Path | None = None,
		step_log_path: str | Path | None = None,
		error_message: str | None = None,
	) -> None:
		if self._finalized:
			return

		self.attach_paths(
			model_path=model_path,
			model_artifact_path=model_artifact_path,
			training_log_path=training_log_path,
			step_log_path=step_log_path,
		)
		finished_at = utc_now_iso()
		duration_seconds = round(time.monotonic() - self._started_monotonic, 3)
		summary = self.episodes.build_summary(total_timesteps=total_timesteps, wall_seconds=duration_seconds)
		milestones = self.episodes.milestone_rows(run_id=self.run_id, mode=self.mode)
		self.manifest["run"]["status"] = status
		self.manifest["run"]["finished_at"] = finished_at
		self.manifest["run"]["duration_seconds"] = duration_seconds
		if error_message:
			self.manifest["run"]["error_message"] = error_message
		self.manifest["summary"] = summary
		self.manifest["milestones"] = milestones
		self._write_json(self.manifest_path, self.manifest)
		self._write_json(self.summary_path, summary)
		self._upsert_runs_registry(summary, finished_at, duration_seconds)
		self._replace_milestone_rows(milestones)
		if self.policy_id and self.manifest["paths"]["model_path"]:
			self._upsert_policy_registry(summary, finished_at)
		if self.manifest["paths"]["model_path"]:
			self._write_model_sidecars()
		self._finalized = True

	def _ensure_registry_files(self) -> None:
		self.tracking_dir.mkdir(parents=True, exist_ok=True)
		_ensure_csv_file(self.tracking_dir / "runs.csv", RUNS_FIELDNAMES)
		_ensure_csv_file(self.tracking_dir / "milestones.csv", MILESTONES_FIELDNAMES)
		_ensure_csv_file(self.tracking_dir / "policies.csv", POLICIES_FIELDNAMES)

	def _upsert_runs_registry(self, summary: dict[str, Any], finished_at: str, duration_seconds: float) -> None:
		row = {
			"run_id": self.run_id,
			"mode": self.mode,
			"status": self.manifest["run"]["status"],
			"label": self.run_label,
			"persona": self.persona,
			"training_technique": self.training_technique,
			"policy_id": self.policy_id,
			"started_at": self.started_at,
			"finished_at": finished_at,
			"duration_seconds": duration_seconds,
			"run_dir": self.run_dir.as_posix(),
			"manifest_path": self.manifest_path.as_posix(),
			"summary_path": self.summary_path.as_posix(),
			"model_path": self.manifest["paths"]["model_path"],
			"model_artifact_path": self.manifest["paths"].get("model_artifact_path", ""),
			"training_log_jsonl": self.manifest["paths"]["training_log_jsonl"],
			"step_log_jsonl": self.manifest["paths"]["step_log_jsonl"],
			"environment_fingerprint": self.environment.get("environment_fingerprint", ""),
			"reward_config_hash": self.environment.get("reward_config_hash", ""),
			"episode_count": summary.get("episode_count"),
			"completed_episode_count": summary.get("completed_episode_count"),
			"success_count": summary.get("success_count"),
			"success_rate": summary.get("success_rate"),
			"mean_episode_reward": summary.get("mean_episode_reward"),
			"median_episode_reward": summary.get("median_episode_reward"),
			"mean_goal_steps": summary.get("mean_goal_steps"),
			"median_goal_steps": summary.get("median_goal_steps"),
			"mean_goal_frames": summary.get("mean_goal_frames"),
			"median_goal_frames": summary.get("median_goal_frames"),
			"mean_goal_timesteps": summary.get("mean_goal_timesteps"),
			"median_goal_timesteps": summary.get("median_goal_timesteps"),
			"first_success_episode": summary.get("first_success_episode"),
			"first_success_timestep": summary.get("first_success_timestep"),
			"timesteps_completed": summary.get("timesteps_completed"),
			"terminal_reason_counts": summary.get("terminal_reason_counts", {}),
			"notes": self.manifest["run"].get("notes", self.run_notes),
		}
		_upsert_csv_rows(
			self.tracking_dir / "runs.csv",
			fieldnames=RUNS_FIELDNAMES,
			key_fields=("run_id",),
			rows=[row],
		)

	def _replace_milestone_rows(self, milestones: list[dict[str, Any]]) -> None:
		_upsert_csv_rows(
			self.tracking_dir / "milestones.csv",
			fieldnames=MILESTONES_FIELDNAMES,
			key_fields=("run_id", "milestone_key"),
			rows=milestones,
			remove_matching={"run_id": self.run_id},
		)

	def _upsert_policy_registry(self, summary: dict[str, Any], finished_at: str) -> None:
		policy_manifest_path = _resolve_model_manifest_output_path(
			self.manifest["paths"].get("model_path", ""),
			self.manifest["paths"].get("model_artifact_path", ""),
		)
		candidate_row = {
			"policy_id": self.policy_id,
			"source_run_id": self.run_id,
			"source_run_finished_at": finished_at,
			"label": self.policy_label,
			"persona": self.persona,
			"objective": self.policy_objective,
			"intended_use": self.policy_intended_use,
			"algorithm": self.policy_algorithm,
			"training_technique": self.training_technique,
			"is_best_in_category": self.policy_best_in_category,
			"notes": self.policy_notes,
			"manifest_path": policy_manifest_path,
			"model_path": self.manifest["paths"]["model_path"],
			"model_artifact_path": self.manifest["paths"].get("model_artifact_path", ""),
			"environment_fingerprint": self.environment.get("environment_fingerprint", ""),
			"reward_config_hash": self.environment.get("reward_config_hash", ""),
			"timesteps_completed": summary.get("timesteps_completed"),
			"success_rate": summary.get("success_rate"),
			"mean_episode_reward": summary.get("mean_episode_reward"),
			"mean_goal_steps": summary.get("mean_goal_steps"),
			"median_goal_steps": summary.get("median_goal_steps"),
			"median_goal_timesteps": summary.get("median_goal_timesteps"),
			"latest_evaluation_at": finished_at if self.mode == "evaluation" else "",
			"updated_at": finished_at,
		}
		existing_row = _load_csv_row_by_key(self.tracking_dir / "policies.csv", key_field="policy_id", key_value=self.policy_id)
		row = merge_policy_registry_row(existing_row, candidate_row, run_mode=self.mode)
		_upsert_csv_rows(
			self.tracking_dir / "policies.csv",
			fieldnames=POLICIES_FIELDNAMES,
			key_fields=("policy_id",),
			rows=[row],
		)

	def _write_model_sidecars(self) -> None:
		model_path = Path(self.manifest["paths"]["model_path"])
		model_artifact_path = Path(self.manifest["paths"].get("model_artifact_path") or model_path)
		if model_path.is_dir() or model_artifact_path.name == MODEL_PACKAGE_FILENAME:
			package_dir = model_path if model_path.is_dir() else model_artifact_path.parent
			manifest_sidecar = package_dir / MODEL_PACKAGE_MANIFEST_FILENAME
			summary_sidecar = package_dir / MODEL_PACKAGE_SUMMARY_FILENAME
		else:
			manifest_sidecar = model_path.with_suffix(".manifest.json")
			summary_sidecar = model_path.with_suffix(".summary.json")
		self._write_json(manifest_sidecar, self.manifest)
		self._write_json(summary_sidecar, self.manifest["summary"])

	def _write_json(self, path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_run_id(label: str) -> str:
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	slug = slugify(label)
	return f"{timestamp}-{slug}-{uuid.uuid4().hex[:8]}"


def slugify(value: str) -> str:
	normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
	return normalized or "run"


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _ensure_csv_file(path: Path, fieldnames: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists():
		return
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()


def _load_csv_row_by_key(path: Path, *, key_field: str, key_value: str) -> dict[str, str] | None:
	if not path.exists():
		return None
	with path.open("r", encoding="utf-8", newline="") as handle:
		for row in csv.DictReader(handle):
			if row.get(key_field, "") == key_value:
				return row
	return None


def merge_policy_registry_row(
	existing_row: Mapping[str, Any] | None,
	candidate_row: Mapping[str, Any],
	*,
	run_mode: str,
) -> dict[str, Any]:
	existing = {field: _csv_value((existing_row or {}).get(field)) for field in POLICIES_FIELDNAMES}
	candidate = {field: _csv_value(candidate_row.get(field)) for field in POLICIES_FIELDNAMES}
	merged = dict(existing)

	merged["policy_id"] = str(candidate.get("policy_id", "") or existing.get("policy_id", ""))

	if run_mode == "evaluation":
		for field in POLICY_REGISTRY_STABLE_FIELDS:
			if not merged.get(field, "") and candidate.get(field, ""):
				merged[field] = candidate[field]
	else:
		for field in POLICY_REGISTRY_STABLE_FIELDS:
			if candidate.get(field, "") != "":
				merged[field] = candidate[field]

	if run_mode == "evaluation":
		if not merged.get("source_run_id", "") and candidate.get("source_run_id", ""):
			merged["source_run_id"] = candidate["source_run_id"]
		if not merged.get("source_run_finished_at", "") and candidate.get("source_run_finished_at", ""):
			merged["source_run_finished_at"] = candidate["source_run_finished_at"]
	else:
		if candidate.get("source_run_id", ""):
			merged["source_run_id"] = candidate["source_run_id"]
		if candidate.get("source_run_finished_at", ""):
			merged["source_run_finished_at"] = candidate["source_run_finished_at"]

	for field in POLICY_REGISTRY_COMPARISON_FIELDS:
		merged[field] = candidate.get(field, "")

	candidate_updated_at = str(candidate.get("updated_at", "") or "")
	merged["updated_at"] = _latest_iso_timestamp(existing.get("updated_at", ""), candidate_updated_at)
	if run_mode == "evaluation":
		merged["latest_evaluation_at"] = _latest_iso_timestamp(existing.get("latest_evaluation_at", ""), candidate_updated_at)
	else:
		merged["latest_evaluation_at"] = str(existing.get("latest_evaluation_at", "") or "")

	return {field: merged.get(field, "") for field in POLICIES_FIELDNAMES}


def _upsert_csv_rows(
	path: Path,
	*,
	fieldnames: list[str],
	key_fields: tuple[str, ...],
	rows: list[dict[str, Any]],
	remove_matching: dict[str, Any] | None = None,
) -> None:
	existing_rows: list[dict[str, Any]] = []
	if path.exists():
		with path.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			existing_rows = list(reader)

	new_keys = {_row_key(row, key_fields) for row in rows}
	filtered_rows: list[dict[str, Any]] = []
	for existing_row in existing_rows:
		if remove_matching and all(existing_row.get(key, "") == _csv_value(value) for key, value in remove_matching.items()):
			continue
		if _row_key(existing_row, key_fields) in new_keys:
			continue
		filtered_rows.append(existing_row)

	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in filtered_rows:
			writer.writerow({field: row.get(field, "") for field in fieldnames})
		for row in rows:
			writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _row_key(row: Mapping[str, Any], key_fields: tuple[str, ...]) -> tuple[str, ...]:
	return tuple(_csv_value(row.get(field)) for field in key_fields)


def _latest_iso_timestamp(*values: Any) -> str:
	timestamps = [str(value) for value in values if str(value)]
	return max(timestamps, default="")


def _csv_value(value: Any) -> str | int | float:
	if value is None:
		return ""
	if isinstance(value, bool):
		return "true" if value else "false"
	if isinstance(value, (int, float)):
		return value
	if isinstance(value, Path):
		return value.as_posix()
	if isinstance(value, (dict, list, tuple)):
		return json.dumps(value, sort_keys=True)
	return str(value)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
	if denominator <= 0:
		return None
	return round(numerator / denominator, 6)


def _safe_mean(values: list[int] | list[float]) -> float | None:
	if not values:
		return None
	return round(float(mean(values)), 6)


def _safe_median(values: list[int] | list[float]) -> float | None:
	if not values:
		return None
	return round(float(median(values)), 6)