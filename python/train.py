from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from godot_env import GodotThrusterEnv
from run_tracking import RunTracker
from tracking_admin import promote_run, rebuild_tracking_registries


MODEL_PACKAGE_FILENAME = "policy.zip"
MODEL_PACKAGE_MANIFEST_FILENAME = "manifest.json"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train or smoke-test the Godot thruster RL environment.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8765)
	parser.add_argument("--frames-per-step", type=int, default=8)
	parser.add_argument("--launch-project", action="store_true")
	parser.add_argument("--godot-executable", default=None)
	parser.add_argument("--no-headless", action="store_true")
	parser.add_argument(
		"--watch",
		action="store_true",
		help="Open the Godot window while training and optionally slow steps so you can watch.",
	)
	parser.add_argument(
		"--watch-step-delay",
		type=float,
		default=0.05,
		help="Extra delay in seconds after each env step when --watch is enabled.",
	)
	parser.add_argument("--connect-timeout", type=float, default=30.0)
	parser.add_argument("--random-only", action="store_true")
	parser.add_argument(
		"--eval-model",
		default=None,
		help="Load a saved PPO model and run evaluation episodes instead of training.",
	)
	parser.add_argument("--eval-episodes", type=int, default=3)
	parser.add_argument("--eval-steps", type=int, default=300)
	parser.add_argument(
		"--stochastic-eval",
		action="store_true",
		help="Sample actions during evaluation instead of using deterministic policy output.",
	)
	parser.add_argument(
		"--log-steps",
		action="store_true",
		help="Print a concise per-step summary during random smoke or evaluation runs.",
	)
	parser.add_argument(
		"--log-step-details",
		action="store_true",
		help="Print the full debug payload returned by RLBridge for each smoke or evaluation step.",
	)
	parser.add_argument(
		"--log-jsonl",
		default=None,
		help="Write one JSON object per step to the given path during random smoke or evaluation runs.",
	)
	parser.add_argument("--smoke-episodes", type=int, default=3)
	parser.add_argument("--smoke-steps", type=int, default=300)
	parser.add_argument("--timesteps", type=int, default=100_000)
	parser.add_argument(
		"--rebuild-tracking",
		action="store_true",
		help="Rebuild runs.csv, policies.csv, and milestones.csv from tracked run manifests and exit.",
	)
	parser.add_argument(
		"--promote-run",
		default=None,
		help="Promote an existing tracked run by run id or manifest path and upsert it into policies.csv.",
	)
	parser.add_argument(
		"--tracking-dir",
		default="experiments",
		help="Directory used for run manifests and CSV experiment registries.",
	)
	parser.add_argument(
		"--run-label",
		default=None,
		help="Human-readable label stored with the tracked run.",
	)
	parser.add_argument(
		"--persona",
		default=None,
		help="Optional behavior tag such as aggressive, efficient, or safe.",
	)
	parser.add_argument(
		"--training-technique",
		default=None,
		help="Optional label for the training structure or technique used for this run.",
	)
	parser.add_argument(
		"--policy-id",
		default=None,
		help="Optional curated policy identifier to upsert into policies.csv or use during promotion.",
	)
	parser.add_argument(
		"--policy-label",
		default=None,
		help="Curated display label for a policy package or promoted run.",
	)
	parser.add_argument(
		"--policy-objective",
		default=None,
		help="Short statement of what the policy is optimized to do.",
	)
	parser.add_argument(
		"--policy-intended-use",
		default=None,
		help="Operational note describing when this policy should be used.",
	)
	parser.add_argument(
		"--policy-algorithm",
		default=None,
		help="Algorithm label stored with the policy metadata, for example PPO.",
	)
	parser.add_argument(
		"--policy-best-in-category",
		action="store_true",
		help="Mark the policy as the current best option in its category.",
	)
	parser.add_argument(
		"--policy-notes",
		default=None,
		help="Curated notes stored with the policy metadata.",
	)
	parser.add_argument(
		"--run-notes",
		default=None,
		help="Optional freeform notes stored in the run manifest.",
	)
	parser.add_argument(
		"--training-log-jsonl",
		default=None,
		help="Write per-episode PPO training summaries to this JSONL file. Defaults to logs/<model_name>_training.jsonl.",
	)
	parser.add_argument(
		"--ppo-n-steps",
		type=int,
		default=256,
		help="Rollout length for PPO. Lower this for short training smoke tests.",
	)
	parser.add_argument(
		"--ppo-batch-size",
		type=int,
		default=64,
		help="Batch size for PPO updates.",
	)
	parser.add_argument("--model-output", default="models/ppo_thruster_agent")
	return parser.parse_args()


def _default_training_log_path(model_output: str) -> Path:
	model_path = Path(model_output)
	return Path("logs") / f"{model_path.stem}_training.jsonl"


def build_env(args: argparse.Namespace) -> GodotThrusterEnv:
	watch_mode = args.watch
	return GodotThrusterEnv(
		host=args.host,
		port=args.port,
		step_frames=args.frames_per_step,
		launch_project=args.launch_project,
		godot_executable=args.godot_executable,
		headless=not (args.no_headless or watch_mode),
		realtime_delay=args.watch_step_delay if watch_mode else 0.0,
		connect_timeout=args.connect_timeout,
	)


def _resolve_model_reference_path(raw_path: str | Path) -> Path:
	path = Path(raw_path)
	if path.suffix == ".zip":
		return path
	return path


def _resolve_model_artifact_path(raw_path: str | Path) -> Path:
	path = Path(raw_path)
	if path.is_dir():
		return path / MODEL_PACKAGE_FILENAME
	if path.suffix == ".zip":
		return path
	return path / MODEL_PACKAGE_FILENAME


def _resolve_existing_model_path(raw_path: str | Path) -> Path:
	path = Path(raw_path)
	if path.is_dir():
		return path / MODEL_PACKAGE_FILENAME
	if path.exists():
		return path
	package_model_path = path / MODEL_PACKAGE_FILENAME
	if package_model_path.exists():
		return package_model_path
	zip_path = _resolve_model_artifact_path(path)
	if zip_path.exists():
		return zip_path
	legacy_zip_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
	if legacy_zip_path.exists():
		return legacy_zip_path
	return package_model_path if path.suffix != ".zip" else path


def _resolve_policy_manifest_path(model_path: str | Path) -> Path:
	path = Path(model_path)
	if path.is_dir():
		return path / MODEL_PACKAGE_MANIFEST_FILENAME
	if path.name == MODEL_PACKAGE_FILENAME:
		return path.parent / MODEL_PACKAGE_MANIFEST_FILENAME
	return path.with_suffix(".manifest.json")


def _infer_run_mode(args: argparse.Namespace) -> str:
	if args.eval_model:
		return "evaluation"
	if args.random_only:
		return "smoke"
	return "training"


def _default_run_label(args: argparse.Namespace) -> str:
	if args.run_label:
		return args.run_label
	if args.eval_model:
		return f"{Path(args.eval_model).stem}-evaluation"
	if args.random_only:
		return "random-smoke"
	return Path(args.model_output).stem


def _default_training_technique(args: argparse.Namespace) -> str:
	if args.training_technique:
		return args.training_technique
	if args.eval_model:
		return "policy_evaluation"
	if args.random_only:
		return "random_smoke"
	return "ppo"


def _default_policy_algorithm(args: argparse.Namespace) -> str:
	if args.policy_algorithm:
		return args.policy_algorithm
	if args.random_only:
		return "random_policy"
	return "PPO"


def _build_policy_updates_from_args(args: argparse.Namespace) -> dict[str, Any]:
	updates: dict[str, Any] = {}
	if args.policy_id is not None:
		updates["policy_id"] = args.policy_id
	if args.policy_label is not None:
		updates["label"] = args.policy_label
	if args.persona is not None:
		updates["persona"] = args.persona
	if args.policy_objective is not None:
		updates["objective"] = args.policy_objective
	if args.policy_intended_use is not None:
		updates["intended_use"] = args.policy_intended_use
	if args.policy_algorithm is not None:
		updates["algorithm"] = args.policy_algorithm
	if args.training_technique is not None:
		updates["training_technique"] = args.training_technique
	if args.policy_best_in_category:
		updates["is_best_in_category"] = True
	if args.policy_notes is not None:
		updates["notes"] = args.policy_notes
	return updates


def build_run_tracker(args: argparse.Namespace, env: GodotThrusterEnv) -> RunTracker:
	return RunTracker(
		tracking_dir=args.tracking_dir,
		mode=_infer_run_mode(args),
		args=vars(args).copy(),
		environment=env.get_environment_metadata(),
		run_label=_default_run_label(args),
		persona=args.persona,
		training_technique=_default_training_technique(args),
		policy_id=args.policy_id,
		run_notes=args.run_notes,
		policy_label=args.policy_label or _default_run_label(args),
		policy_objective=args.policy_objective,
		policy_intended_use=args.policy_intended_use,
		policy_algorithm=_default_policy_algorithm(args),
		policy_best_in_category=args.policy_best_in_category,
		policy_notes=args.policy_notes,
	)


def _run_tracking_admin_command(args: argparse.Namespace) -> bool:
	if args.promote_run:
		result = promote_run(
			args.promote_run,
			tracking_dir=args.tracking_dir,
			policy_updates=_build_policy_updates_from_args(args),
		)
		print(
			"promoted_run run_id=%s policy_id=%s runs=%d policies=%d milestones=%d"
			% (
				result.get("run_id", ""),
				result.get("policy_id", ""),
				int(result.get("run_count", 0)),
				int(result.get("policy_count", 0)),
				int(result.get("milestone_count", 0)),
			)
		)
		return True

	if args.rebuild_tracking:
		counts = rebuild_tracking_registries(args.tracking_dir)
		print(
			"rebuilt_tracking runs=%d policies=%d milestones=%d"
			% (
				int(counts.get("run_count", 0)),
				int(counts.get("policy_count", 0)),
				int(counts.get("milestone_count", 0)),
			)
		)
		return True

	return False


def _load_policy_manifest(model_path: Path) -> dict[str, Any] | None:
	manifest_path = _resolve_policy_manifest_path(model_path)
	if not manifest_path.exists():
		return None
	try:
		return json.loads(manifest_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as exc:
		raise SystemExit(f"Could not load saved policy manifest: {manifest_path}") from exc


def _format_step_summary(step_index: int, reward: float, total_reward: float, debug: dict) -> str:
	reward_terms = dict(debug.get("reward_terms", {}))
	return (
		"step=%d reward=%+.3f total=%+.3f dist=%.3f rel_speed=%.3f progress=%+.4f "
		"thr_sum=%.3f thr_max=%.3f reason=%s"
		% (
			step_index,
			reward,
			total_reward,
			float(debug.get("goal_distance", 0.0)),
			float(debug.get("relative_speed", 0.0)),
			float(reward_terms.get("progress", 0.0)),
			float(debug.get("throttle_sum", 0.0)),
			float(debug.get("throttle_max", 0.0)),
			debug.get("terminal_reason", "unknown"),
		)
	)


def run_random_smoke(
	env: GodotThrusterEnv,
	episodes: int,
	max_steps: int,
	log_steps: bool = False,
	log_step_details: bool = False,
	log_jsonl: str | None = None,
	tracker: RunTracker | None = None,
) -> None:
	jsonl_path = Path(log_jsonl) if log_jsonl else None
	jsonl_file = None
	if jsonl_path is not None:
		jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		jsonl_file = jsonl_path.open("w", encoding="utf-8")
		if tracker is not None:
			tracker.attach_paths(step_log_path=jsonl_path)
			jsonl_file.write(json.dumps(tracker.header_record()) + "\n")

	try:
		for episode in range(episodes):
			observation, info = env.reset()
			total_reward = 0.0
			last_info = info
			for step in range(max_steps):
				action = env.action_space.sample().astype(np.float32)
				observation, reward, terminated, truncated, last_info = env.step(action)
				total_reward += reward
				debug_payload = dict(last_info.get("debug", {}))

				if log_steps or log_step_details:
					print(_format_step_summary(step + 1, reward, total_reward, debug_payload))
				if log_step_details:
					print("debug=%s" % debug_payload)

				if jsonl_file is not None:
					jsonl_record = {
						"episode": episode,
						"step": step + 1,
						"reward": reward,
						"episode_total_reward": total_reward,
						"terminated": terminated,
						"truncated": truncated,
						"observation": observation.astype(float).tolist(),
						"debug": debug_payload,
					}
					jsonl_file.write(json.dumps(jsonl_record) + "\n")

				if terminated or truncated:
					terminal_reason = str(last_info.get("terminal_reason", "unknown"))
					success = terminal_reason == "goal_reached"
					if tracker is not None:
						tracker.record_episode(
							episode=episode,
							steps=step + 1,
							frames=int(last_info.get("episode_frames", 0)),
							reward_total=total_reward,
							terminal_reason=terminal_reason,
							success=success,
						)
					print(
						"episode=%d steps=%d reward=%.3f reason=%s goal_distance=%.3f relative_speed=%.3f"
						% (
							episode,
							step + 1,
							total_reward,
							terminal_reason,
							float(last_info.get("goal_distance", 0.0)),
							float(last_info.get("relative_speed", 0.0)),
						)
					)
					break
			else:
				if tracker is not None:
					tracker.record_episode(
						episode=episode,
						steps=max_steps,
						frames=int(last_info.get("episode_frames", 0)),
						reward_total=total_reward,
						terminal_reason="max_steps",
						success=False,
					)
				print(
					"episode=%d steps=%d reward=%.3f reason=max_steps"
					% (episode, max_steps, total_reward)
				)
	finally:
		if jsonl_file is not None:
			jsonl_file.close()


def run_ppo_training(
	env: GodotThrusterEnv,
	timesteps: int,
	model_output: str,
	ppo_n_steps: int,
	ppo_batch_size: int,
	training_log_jsonl: str | None,
	tracker: RunTracker | None,
) -> tuple[Path, Path, int]:
	try:
		from stable_baselines3 import PPO
		from stable_baselines3.common.callbacks import BaseCallback
	except ImportError as exc:
		raise SystemExit(
			"stable-baselines3 is not installed. Start with --random-only, or install a supported RL stack. "
			"If Python 3.14 package support lags, use Python 3.12 or 3.13 for training."
		) from exc

	class EpisodeRewardLogger(BaseCallback):
		def __init__(self, log_path: Path) -> None:
			super().__init__(verbose=0)
			self.log_path = log_path
			self.log_file = None
			self.episode_index = 0
			self.episode_reward_total = 0.0
			self.episode_step_count = 0
			self.episode_reward_terms: dict[str, float] = {}
			self.last_info: dict[str, object] = {}

		def _on_training_start(self) -> None:
			self.log_path.parent.mkdir(parents=True, exist_ok=True)
			self.log_file = self.log_path.open("w", encoding="utf-8")
			if tracker is not None:
				tracker.attach_paths(training_log_path=self.log_path)
				self.log_file.write(json.dumps(tracker.header_record()) + "\n")
			print("training_log_jsonl=%s" % self.log_path.as_posix())

		def _on_step(self) -> bool:
			rewards = self.locals.get("rewards")
			dones = self.locals.get("dones")
			infos = self.locals.get("infos")
			if rewards is None or dones is None or infos is None or len(infos) == 0:
				return True

			reward = float(rewards[0])
			done = bool(dones[0])
			info = dict(infos[0] or {})
			self.last_info = info
			reward_terms = dict(info.get("reward_terms", {}))

			self.episode_reward_total += reward
			self.episode_step_count += 1
			for key, value in reward_terms.items():
				if isinstance(value, (int, float)):
					self.episode_reward_terms[key] = self.episode_reward_terms.get(key, 0.0) + float(value)

			if not done:
				return True

			record = {
				"episode": self.episode_index,
				"timesteps": self.num_timesteps,
				"episode_total_reward": self.episode_reward_total,
				"episode_steps": self.episode_step_count,
				"episode_frames": int(info.get("episode_frames", 0)),
				"terminal_reason": info.get("terminal_reason", "unknown"),
				"goal_distance": float(info.get("goal_distance", 0.0)),
				"relative_speed": float(info.get("relative_speed", 0.0)),
				"is_goal_completed": bool(info.get("is_goal_completed", False)),
				"is_inside_goal": bool(info.get("is_inside_goal", False)),
				"reward_terms": self.episode_reward_terms,
			}
			self.log_file.write(json.dumps(record) + "\n")
			self.log_file.flush()
			if tracker is not None:
				tracker.record_episode(
					episode=self.episode_index,
					steps=self.episode_step_count,
					frames=int(info.get("episode_frames", 0)),
					reward_total=self.episode_reward_total,
					terminal_reason=str(info.get("terminal_reason", "unknown")),
					success=bool(info.get("is_goal_completed", False)),
					timesteps=self.num_timesteps,
				)
			print(
				"train_episode=%d steps=%d reward=%.3f reason=%s goal_distance=%.3f relative_speed=%.3f"
				% (
					self.episode_index,
					self.episode_step_count,
					self.episode_reward_total,
					info.get("terminal_reason", "unknown"),
					float(info.get("goal_distance", 0.0)),
					float(info.get("relative_speed", 0.0)),
				)
			)

			self.episode_index += 1
			self.episode_reward_total = 0.0
			self.episode_step_count = 0
			self.episode_reward_terms = {}
			self.last_info = {}
			return True

		def _on_training_end(self) -> None:
			if self.log_file is not None:
				if self.episode_step_count > 0:
					record = {
						"episode": self.episode_index,
						"timesteps": self.num_timesteps,
						"episode_total_reward": self.episode_reward_total,
						"episode_steps": self.episode_step_count,
						"episode_frames": int(self.last_info.get("episode_frames", 0)),
						"terminal_reason": self.last_info.get("terminal_reason", "training_stopped"),
						"goal_distance": float(self.last_info.get("goal_distance", 0.0)),
						"relative_speed": float(self.last_info.get("relative_speed", 0.0)),
						"is_goal_completed": bool(self.last_info.get("is_goal_completed", False)),
						"is_inside_goal": bool(self.last_info.get("is_inside_goal", False)),
						"completed_episode": False,
						"reward_terms": self.episode_reward_terms,
					}
					self.log_file.write(json.dumps(record) + "\n")
					self.log_file.flush()
					if tracker is not None:
						tracker.record_episode(
							episode=self.episode_index,
							steps=self.episode_step_count,
							frames=int(self.last_info.get("episode_frames", 0)),
							reward_total=self.episode_reward_total,
							terminal_reason=str(self.last_info.get("terminal_reason", "training_stopped")),
							success=bool(self.last_info.get("is_goal_completed", False)),
							timesteps=self.num_timesteps,
							completed_episode=False,
						)
				self.log_file.close()
				self.log_file = None

	model_reference_path = _resolve_model_reference_path(model_output)
	artifact_path = _resolve_model_artifact_path(model_output)
	artifact_path.parent.mkdir(parents=True, exist_ok=True)
	training_log_path = Path(training_log_jsonl) if training_log_jsonl else _default_training_log_path(model_output)
	rollout_steps = max(int(ppo_n_steps), 2)
	batch_size = max(int(ppo_batch_size), 2)
	batch_size = min(batch_size, rollout_steps)
	model = PPO(
		"MlpPolicy",
		env,
		verbose=1,
		n_steps=rollout_steps,
		batch_size=batch_size,
	)
	model.learn(total_timesteps=timesteps, callback=EpisodeRewardLogger(training_log_path))
	model.save(artifact_path.as_posix())
	if tracker is not None:
		tracker.attach_paths(
			model_path=model_reference_path,
			model_artifact_path=artifact_path,
			training_log_path=training_log_path,
		)
	return model_reference_path, training_log_path, int(getattr(model, "num_timesteps", timesteps))


def _get_space_shape(space: object) -> tuple[int, ...]:
	shape = getattr(space, "shape", ())
	if shape is None:
		return ()
	return tuple(shape)


def _validate_model_against_env(model: object, env: GodotThrusterEnv, model_path: Path | None = None) -> None:
	model_observation_shape = _get_space_shape(getattr(model, "observation_space", None))
	env_observation_shape = _get_space_shape(env.observation_space)
	if model_observation_shape != env_observation_shape:
		raise SystemExit(
			"Saved model observation shape %s does not match environment shape %s"
			% (model_observation_shape, env_observation_shape)
		)

	model_action_shape = _get_space_shape(getattr(model, "action_space", None))
	env_action_shape = _get_space_shape(env.action_space)
	if model_action_shape != env_action_shape:
		raise SystemExit(
			"Saved model action shape %s does not match environment shape %s"
			% (model_action_shape, env_action_shape)
		)

	if model_path is None:
		return

	manifest = _load_policy_manifest(model_path)
	if manifest is None:
		return

	manifest_environment = dict(manifest.get("environment") or {})
	current_environment = env.get_environment_metadata()
	manifest_fields = tuple(
		str(field_name)
		for field_name in (manifest_environment.get("observation_schema") or {}).get("fields", [])
	)
	current_fields = tuple(
		str(field_name)
		for field_name in (current_environment.get("observation_schema") or {}).get("fields", [])
	)
	if manifest_fields and current_fields and manifest_fields != current_fields:
		raise SystemExit("Saved model observation schema does not match the live environment schema")

	manifest_step_frames = manifest_environment.get("default_action_frames")
	current_step_frames = current_environment.get("default_action_frames")
	if (
		manifest_step_frames is not None
		and current_step_frames is not None
		and int(manifest_step_frames) != int(current_step_frames)
	):
		raise SystemExit(
			"Saved model step-frame configuration %s does not match environment setting %s"
			% (manifest_step_frames, current_step_frames)
		)

	manifest_fingerprint = str(manifest_environment.get("environment_fingerprint", ""))
	current_fingerprint = str(current_environment.get("environment_fingerprint", ""))
	if manifest_fingerprint and current_fingerprint and manifest_fingerprint != current_fingerprint:
		raise SystemExit("Saved model environment fingerprint does not match the live environment")

	manifest_reward_hash = str(manifest_environment.get("reward_config_hash", ""))
	current_reward_hash = str(current_environment.get("reward_config_hash", ""))
	if manifest_reward_hash and current_reward_hash and manifest_reward_hash != current_reward_hash:
		print(
			"warning: saved model reward configuration differs from the live environment; "
			"evaluation may still run, but results are not directly comparable"
		)


def run_policy_evaluation(
	env: GodotThrusterEnv,
	model_input: str,
	episodes: int,
	max_steps: int,
	deterministic: bool = True,
	log_steps: bool = False,
	log_step_details: bool = False,
	log_jsonl: str | None = None,
	tracker: RunTracker | None = None,
) -> None:
	try:
		from stable_baselines3 import PPO
	except ImportError as exc:
		raise SystemExit(
			"stable-baselines3 is required for saved-model evaluation. "
			"If Python 3.14 package support lags, use Python 3.12 or 3.13 for evaluation."
		) from exc

	model_path = _resolve_existing_model_path(model_input)
	if not model_path.exists():
		raise SystemExit("Saved model not found: %s" % model_path)

	model = PPO.load(model_path.as_posix())
	_validate_model_against_env(model, env, model_path=model_path)

	jsonl_path = Path(log_jsonl) if log_jsonl else None
	jsonl_file = None
	if jsonl_path is not None:
		jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		jsonl_file = jsonl_path.open("w", encoding="utf-8")
		if tracker is not None:
			tracker.attach_paths(step_log_path=jsonl_path)
			jsonl_file.write(json.dumps(tracker.header_record()) + "\n")

	success_count = 0
	try:
		for episode in range(episodes):
			observation, info = env.reset()
			total_reward = 0.0
			last_info = info
			for step in range(max_steps):
				action, _ = model.predict(observation, deterministic=deterministic)
				action_array = np.asarray(action, dtype=np.float32).reshape(-1)
				observation, reward, terminated, truncated, last_info = env.step(action_array)
				total_reward += reward
				debug_payload = dict(last_info.get("debug", {}))

				if log_steps or log_step_details:
					print(_format_step_summary(step + 1, reward, total_reward, debug_payload))
				if log_step_details:
					print("debug=%s" % debug_payload)

				if jsonl_file is not None:
					jsonl_record = {
						"mode": "eval",
						"episode": episode,
						"step": step + 1,
						"reward": reward,
						"episode_total_reward": total_reward,
						"terminated": terminated,
						"truncated": truncated,
						"action": action_array.astype(float).tolist(),
						"observation": observation.astype(float).tolist(),
						"debug": debug_payload,
					}
					jsonl_file.write(json.dumps(jsonl_record) + "\n")

				if terminated or truncated:
					success = last_info.get("terminal_reason") == "goal_reached"
					if success:
						success_count += 1
					if tracker is not None:
						tracker.record_episode(
							episode=episode,
							steps=step + 1,
							frames=int(last_info.get("episode_frames", 0)),
							reward_total=total_reward,
							terminal_reason=str(last_info.get("terminal_reason", "unknown")),
							success=success,
						)
					print(
						"eval_episode=%d steps=%d reward=%.3f reason=%s goal_distance=%.3f relative_speed=%.3f success=%s"
						% (
							episode,
							step + 1,
							total_reward,
							last_info.get("terminal_reason", "unknown"),
							float(last_info.get("goal_distance", 0.0)),
							float(last_info.get("relative_speed", 0.0)),
							"yes" if success else "no",
						)
					)
					break
			else:
				if tracker is not None:
					tracker.record_episode(
						episode=episode,
						steps=max_steps,
						frames=int(last_info.get("episode_frames", 0)),
						reward_total=total_reward,
						terminal_reason="max_steps",
						success=False,
					)
				print(
					"eval_episode=%d steps=%d reward=%.3f reason=max_steps success=no"
					% (episode, max_steps, total_reward)
				)
	finally:
		if jsonl_file is not None:
			jsonl_file.close()

	print(
		"eval_summary episodes=%d successes=%d success_rate=%.3f deterministic=%s"
		% (episodes, success_count, success_count / max(episodes, 1), "yes" if deterministic else "no")
	)


def main() -> None:
	args = parse_args()
	if _run_tracking_admin_command(args):
		return
	if args.random_only and args.eval_model:
		raise SystemExit("Use either --random-only or --eval-model, not both.")

	env = build_env(args)
	tracker = build_run_tracker(args, env)
	try:
		if args.eval_model:
			run_policy_evaluation(
				env,
				model_input=args.eval_model,
				episodes=args.eval_episodes,
				max_steps=args.eval_steps,
				deterministic=not args.stochastic_eval,
				log_steps=args.log_steps,
				log_step_details=args.log_step_details,
				log_jsonl=args.log_jsonl,
				tracker=tracker,
			)
			tracker.finalize(status="completed", step_log_path=args.log_jsonl)
			return

		run_random_smoke(
			env,
			episodes=args.smoke_episodes,
			max_steps=args.smoke_steps,
			log_steps=args.log_steps,
			log_step_details=args.log_step_details,
			log_jsonl=args.log_jsonl,
			tracker=tracker if args.random_only else None,
		)
		if args.random_only:
			tracker.finalize(status="completed", step_log_path=args.log_jsonl)
			return

		saved_model_path, training_log_path, completed_timesteps = run_ppo_training(
			env,
			timesteps=args.timesteps,
			model_output=args.model_output,
			ppo_n_steps=args.ppo_n_steps,
			ppo_batch_size=args.ppo_batch_size,
			training_log_jsonl=args.training_log_jsonl,
			tracker=tracker,
		)
		tracker.finalize(
			status="completed",
			total_timesteps=completed_timesteps,
			model_path=saved_model_path,
			training_log_path=training_log_path,
		)
	except BaseException as exc:
		tracker.finalize(
			status="failed",
			step_log_path=args.log_jsonl,
			error_message=str(exc),
		)
		raise
	finally:
		env.close()


if __name__ == "__main__":
	main()