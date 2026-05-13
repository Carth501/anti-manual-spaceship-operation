from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_DIR) not in sys.path:
	sys.path.insert(0, str(PROJECT_PYTHON_DIR))

from run_tracking import RunTracker  # noqa: E402
from tracking_admin import rebuild_tracking_registries  # noqa: E402


class PolicyRegistryTests(unittest.TestCase):
	def test_tracker_persists_policy_input_config(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			tracking_dir = Path(temp_dir) / "experiments"
			tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="training",
				args={},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="thruster-aware-baseline",
			)

			tracker.attach_policy_input_config(
				{
					"input_adapter": "thruster_set_encoder_v1",
					"global_feature_count": 7,
					"thruster_feature_count": 12,
				}
			)

			manifest = json.loads(tracker.manifest_path.read_text(encoding="utf-8"))
			self.assertEqual(
				manifest["training"]["policy_input_config"]["input_adapter"],
				"thruster_set_encoder_v1",
			)
			self.assertEqual(
				tracker.header_record()["training"]["policy_input_config"]["thruster_feature_count"],
				12,
			)

	def test_tracker_writes_policy_comparison_fields(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			tracking_dir = Path(temp_dir) / "experiments"
			model_dir = Path(temp_dir) / "models" / "safe_policy"
			model_dir.mkdir(parents=True, exist_ok=True)
			(model_dir / "policy.zip").write_bytes(b"")

			tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="training",
				args={},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="safe-baseline",
				persona="safe",
				training_technique="ppo_reward_baseline",
				policy_id="safe-docker-v1",
				policy_label="Safe Docker V1",
				policy_algorithm="PPO",
			)
			source_run_id = tracker.run_id
			tracker.record_episode(
				episode=1,
				steps=10,
				frames=80,
				reward_total=5.0,
				terminal_reason="goal_reached",
				success=True,
				timesteps=64,
			)
			tracker.record_episode(
				episode=2,
				steps=14,
				frames=112,
				reward_total=7.0,
				terminal_reason="goal_reached",
				success=True,
				timesteps=96,
			)
			tracker.finalize(
				status="completed",
				total_timesteps=160,
				model_path=model_dir,
				model_artifact_path=model_dir / "policy.zip",
			)

			rows = _read_csv_rows(tracking_dir / "policies.csv")
			self.assertEqual(len(rows), 1)
			self.assertEqual(rows[0]["policy_id"], "safe-docker-v1")
			self.assertEqual(rows[0]["source_run_id"], source_run_id)
			self.assertNotEqual(rows[0]["source_run_finished_at"], "")
			self.assertEqual(rows[0]["latest_evaluation_at"], "")
			self.assertEqual(rows[0]["updated_at"], rows[0]["source_run_finished_at"])
			self.assertEqual(rows[0]["mean_episode_reward"], "6.0")
			self.assertEqual(rows[0]["median_goal_timesteps"], "80.0")

	def test_evaluation_preserves_source_timestamps(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			tracking_dir = Path(temp_dir) / "experiments"
			model_dir = Path(temp_dir) / "models" / "safe_policy"
			model_dir.mkdir(parents=True, exist_ok=True)
			(model_dir / "policy.zip").write_bytes(b"")

			training_tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="training",
				args={},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="safe-baseline",
				persona="safe",
				training_technique="ppo_reward_baseline",
				policy_id="safe-docker-v1",
				policy_label="Safe Docker V1",
				policy_algorithm="PPO",
			)
			training_tracker.record_episode(
				episode=1,
				steps=12,
				frames=96,
				reward_total=4.0,
				terminal_reason="goal_reached",
				success=True,
				timesteps=72,
			)
			training_tracker.finalize(
				status="completed",
				total_timesteps=72,
				model_path=model_dir,
				model_artifact_path=model_dir / "policy.zip",
			)

			source_row = _read_csv_rows(tracking_dir / "policies.csv")[0]

			evaluation_tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="evaluation",
				args={"eval_model": model_dir.as_posix()},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="safe-eval",
				persona="safe",
				training_technique="policy_evaluation",
				policy_id="safe-docker-v1",
				policy_algorithm="PPO",
			)
			evaluation_tracker.record_episode(
				episode=2,
				steps=20,
				frames=160,
				reward_total=2.0,
				terminal_reason="timeout",
				success=False,
				timesteps=144,
			)
			evaluation_tracker.finalize(
				status="completed",
				total_timesteps=144,
				model_path=model_dir,
				model_artifact_path=model_dir / "policy.zip",
			)

			rows = _read_csv_rows(tracking_dir / "policies.csv")
			self.assertEqual(len(rows), 1)
			self.assertEqual(rows[0]["source_run_id"], source_row["source_run_id"])
			self.assertEqual(rows[0]["source_run_finished_at"], source_row["source_run_finished_at"])
			self.assertEqual(rows[0]["label"], "Safe Docker V1")
			self.assertNotEqual(rows[0]["latest_evaluation_at"], "")
			self.assertEqual(rows[0]["updated_at"], rows[0]["latest_evaluation_at"])
			self.assertEqual(rows[0]["mean_episode_reward"], "2.0")

	def test_rebuild_collapses_policy_history_with_timestamp_semantics(self) -> None:
		with tempfile.TemporaryDirectory() as temp_dir:
			tracking_dir = Path(temp_dir) / "experiments"
			model_dir = Path(temp_dir) / "models" / "safe_policy"
			model_dir.mkdir(parents=True, exist_ok=True)
			(model_dir / "policy.zip").write_bytes(b"")

			training_tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="training",
				args={},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="safe-baseline",
				persona="safe",
				training_technique="ppo_reward_baseline",
				policy_id="safe-docker-v1",
				policy_label="Safe Docker V1",
				policy_algorithm="PPO",
			)
			training_tracker.record_episode(
				episode=1,
				steps=12,
				frames=96,
				reward_total=4.0,
				terminal_reason="goal_reached",
				success=True,
				timesteps=72,
			)
			training_tracker.finalize(
				status="completed",
				total_timesteps=72,
				model_path=model_dir,
				model_artifact_path=model_dir / "policy.zip",
			)

			evaluation_tracker = RunTracker(
				tracking_dir=tracking_dir,
				mode="evaluation",
				args={"eval_model": model_dir.as_posix()},
				environment={
					"environment_fingerprint": "env-fingerprint",
					"reward_config_hash": "reward-hash",
				},
				run_label="safe-eval",
				persona="safe",
				training_technique="policy_evaluation",
				policy_id="safe-docker-v1",
				policy_algorithm="PPO",
			)
			evaluation_tracker.record_episode(
				episode=2,
				steps=20,
				frames=160,
				reward_total=2.0,
				terminal_reason="timeout",
				success=False,
				timesteps=144,
			)
			evaluation_tracker.finalize(
				status="completed",
				total_timesteps=144,
				model_path=model_dir,
				model_artifact_path=model_dir / "policy.zip",
			)

			_rewrite_finished_at(training_tracker.manifest_path, "2026-05-12T10:00:00Z")
			_rewrite_finished_at(evaluation_tracker.manifest_path, "2026-05-12T11:00:00Z")

			counts = rebuild_tracking_registries(tracking_dir)

			rows = _read_csv_rows(tracking_dir / "policies.csv")
			self.assertEqual(counts["policy_count"], 1)
			self.assertEqual(len(rows), 1)
			self.assertEqual(rows[0]["source_run_id"], training_tracker.run_id)
			self.assertEqual(rows[0]["source_run_finished_at"], "2026-05-12T10:00:00Z")
			self.assertEqual(rows[0]["latest_evaluation_at"], "2026-05-12T11:00:00Z")
			self.assertEqual(rows[0]["updated_at"], "2026-05-12T11:00:00Z")
			self.assertEqual(rows[0]["label"], "Safe Docker V1")
			self.assertEqual(rows[0]["mean_episode_reward"], "2.0")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
	with path.open("r", encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle))


def _rewrite_finished_at(manifest_path: Path, finished_at: str) -> None:
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	manifest.setdefault("run", {})["finished_at"] = finished_at
	manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
	unittest.main()