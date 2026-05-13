from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_DIR) not in sys.path:
	sys.path.insert(0, str(PROJECT_PYTHON_DIR))

from train import _build_thruster_policy_input_config_from_metadata, _validate_model_against_env  # noqa: E402


class _FakeSpace:
	def __init__(self, shape: tuple[int, ...]) -> None:
		self.shape = shape


class _FakeModel:
	def __init__(self, observation_shape: tuple[int, ...], action_shape: tuple[int, ...]) -> None:
		self.observation_space = _FakeSpace(observation_shape)
		self.action_space = _FakeSpace(action_shape)


class _FakeEnv:
	def __init__(
		self,
		observation_shape: tuple[int, ...],
		action_shape: tuple[int, ...],
		metadata: dict[str, object],
	) -> None:
		self.observation_space = _FakeSpace(observation_shape)
		self.action_space = _FakeSpace(action_shape)
		self._metadata = metadata

	def get_environment_metadata(self) -> dict[str, object]:
		return copy.deepcopy(self._metadata)


class PolicyCompatibilityTests(unittest.TestCase):
	def test_thruster_policy_input_config_builds_structured_thruster_rows(self) -> None:
		metadata = _build_environment_metadata(
			thruster_count=2,
			thruster_config={
				"center_of_mass_local": [2.0, 0.0, -2.0],
				"direct_throttle_slew_rate": 0.5,
				"thrusters": [
					{
						"index": 0,
						"enabled": True,
						"position_local": [2.0, 0.0, 0.0],
						"thrust_direction_local": [1.0, 0.0, 0.0],
						"linear_response": [1.0, 0.0, 0.0],
						"angular_response": [0.0, 1.0, 0.0],
						"max_force": 10.0,
					},
				],
			},
		)

		config = _build_thruster_policy_input_config_from_metadata(metadata)

		self.assertEqual(config["input_adapter"], "thruster_set_encoder_v1")
		self.assertEqual(config["thruster_count"], 2)
		self.assertEqual(config["enabled_thruster_count"], 1)
		self.assertEqual(config["global_feature_count"], len(config["global_feature_values"]))
		self.assertEqual(len(config["thruster_feature_rows"]), 2)
		self.assertEqual(config["thruster_feature_count"], len(config["thruster_feature_names"]))
		self.assertIn("max_force", config["thruster_feature_names"])
		self.assertEqual(
			config["thruster_feature_rows"][0][config["thruster_feature_names"].index("present")],
			1.0,
		)
		self.assertEqual(
			config["thruster_feature_rows"][0][config["thruster_feature_names"].index("max_force")],
			1.0,
		)
		self.assertEqual(
			config["thruster_feature_rows"][1][config["thruster_feature_names"].index("present")],
			0.0,
		)
		self.assertEqual(
			config["thruster_feature_rows"][1][config["thruster_feature_names"].index("enabled")],
			0.0,
		)

	def test_policy_input_schema_mismatch_raises(self) -> None:
		live_metadata = _build_environment_metadata(
			thruster_count=2,
			thruster_config={
				"center_of_mass_local": [0.0, 0.0, 0.0],
				"direct_throttle_slew_rate": 1.0,
				"thrusters": [
					{
						"index": 0,
						"enabled": True,
						"position_local": [1.0, 0.0, 0.0],
						"thrust_direction_local": [0.0, 1.0, 0.0],
						"linear_response": 1.0,
						"angular_response": 1.0,
						"max_force": 10.0,
					},
				],
			},
		)
		training_block = {
			"policy_input_config": {
				"input_adapter": "thruster_set_encoder_v1",
				"global_feature_names": ["different_global_feature"],
				"thruster_feature_names": ["present", "enabled"],
				"pooling": "mean_present",
			}
		}
		with tempfile.TemporaryDirectory() as temp_dir:
			model_path = _write_policy_package(
				Path(temp_dir),
				_build_manifest_environment(thruster_count=2),
				training_block=training_block,
			)
			with self.assertRaisesRegex(SystemExit, "global policy input schema"):
				_validate_model_against_env(
					_FakeModel(observation_shape=(25,), action_shape=(13,)),
					_FakeEnv(observation_shape=(25,), action_shape=(13,), metadata=live_metadata),
					model_path=model_path,
				)

	def test_reward_config_drift_warns_but_does_not_raise(self) -> None:
		metadata = _build_environment_metadata(reward_config_hash="live-reward")
		with tempfile.TemporaryDirectory() as temp_dir:
			model_path = _write_policy_package(
				Path(temp_dir),
				_build_manifest_environment(reward_config_hash="saved-reward"),
			)
			buffer = io.StringIO()
			with redirect_stdout(buffer):
				_validate_model_against_env(
					_FakeModel(observation_shape=(25,), action_shape=(13,)),
					_FakeEnv(observation_shape=(25,), action_shape=(13,), metadata=metadata),
					model_path=model_path,
				)
			self.assertIn("warning: saved model reward configuration differs", buffer.getvalue())

	def test_observation_schema_mismatch_raises(self) -> None:
		live_metadata = _build_environment_metadata(
			observation_fields=[
				"goal_offset_local_x",
				"goal_offset_local_y",
				"goal_offset_local_z",
				"linear_velocity_local_x",
				"linear_velocity_local_y",
				"linear_velocity_local_z",
				"angular_velocity_local_x",
				"angular_velocity_local_y",
				"angular_velocity_local_z",
				"relative_speed",
				*[f"thruster_throttle_{index:02d}" for index in range(13)],
				"is_inside_goal",
				"is_goal_completed",
			],
		)
		saved_metadata = _build_manifest_environment(
			observation_fields=[
				"goal_offset_local_x",
				"goal_offset_local_y",
				"goal_offset_local_z",
				"linear_velocity_local_x",
				"linear_velocity_local_y",
				"linear_velocity_local_z",
				"angular_velocity_local_x",
				"angular_velocity_local_y",
				"angular_velocity_local_z",
				"goal_speed",
				*[f"thruster_throttle_{index:02d}" for index in range(13)],
				"is_inside_goal",
				"is_goal_completed",
			],
		)
		with tempfile.TemporaryDirectory() as temp_dir:
			model_path = _write_policy_package(Path(temp_dir), saved_metadata)
			with self.assertRaisesRegex(SystemExit, "observation schema"):
				_validate_model_against_env(
					_FakeModel(observation_shape=(25,), action_shape=(13,)),
					_FakeEnv(observation_shape=(25,), action_shape=(13,), metadata=live_metadata),
					model_path=model_path,
				)

	def test_action_shape_mismatch_raises(self) -> None:
		with self.assertRaisesRegex(SystemExit, "action shape"):
			_validate_model_against_env(
				_FakeModel(observation_shape=(25,), action_shape=(12,)),
				_FakeEnv(observation_shape=(25,), action_shape=(13,), metadata=_build_environment_metadata()),
			)


def _write_policy_package(
	root: Path,
	environment_block: dict[str, object],
	training_block: dict[str, object] | None = None,
) -> Path:
	package_dir = root / "policy_package"
	package_dir.mkdir(parents=True, exist_ok=True)
	(package_dir / "policy.zip").write_bytes(b"")
	manifest = {
		"environment": environment_block,
		"policy": {},
		"training": dict(training_block or {}),
		"run": {"label": "test-run"},
		"paths": {"model_path": package_dir.as_posix(), "model_artifact_path": (package_dir / "policy.zip").as_posix()},
	}
	(package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
	return package_dir


def _build_environment_metadata(
	*,
	observation_fields: list[str] | None = None,
	thruster_count: int = 13,
	thruster_config: dict[str, object] | None = None,
	default_action_frames: int = 8,
	environment_fingerprint: str = "environment-fingerprint",
	reward_config_hash: str = "reward-config-hash",
) -> dict[str, object]:
	fields = observation_fields or [
		"goal_offset_local_x",
		"goal_offset_local_y",
		"goal_offset_local_z",
		"linear_velocity_local_x",
		"linear_velocity_local_y",
		"linear_velocity_local_z",
		"angular_velocity_local_x",
		"angular_velocity_local_y",
		"angular_velocity_local_z",
		"relative_speed",
		*[f"thruster_throttle_{index:02d}" for index in range(13)],
		"is_inside_goal",
		"is_goal_completed",
	]
	return {
		"thruster_count": thruster_count,
		"thruster_config": dict(thruster_config or {}),
		"observation_schema": {"fields": fields},
		"default_action_frames": default_action_frames,
		"environment_fingerprint": environment_fingerprint,
		"reward_config_hash": reward_config_hash,
	}


def _build_manifest_environment(
	*,
	observation_fields: list[str] | None = None,
	thruster_count: int = 13,
	thruster_config: dict[str, object] | None = None,
	default_action_frames: int = 8,
	environment_fingerprint: str = "environment-fingerprint",
	reward_config_hash: str = "reward-config-hash",
) -> dict[str, object]:
	return _build_environment_metadata(
		observation_fields=observation_fields,
		thruster_count=thruster_count,
		thruster_config=thruster_config,
		default_action_frames=default_action_frames,
		environment_fingerprint=environment_fingerprint,
		reward_config_hash=reward_config_hash,
	)


if __name__ == "__main__":
	unittest.main()