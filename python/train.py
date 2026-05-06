from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from godot_env import GodotThrusterEnv


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train or smoke-test the Godot thruster RL environment.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8765)
	parser.add_argument("--frames-per-step", type=int, default=4)
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
		"--log-steps",
		action="store_true",
		help="Print a concise per-step summary during random smoke runs.",
	)
	parser.add_argument(
		"--log-step-details",
		action="store_true",
		help="Print the full debug payload returned by RLBridge for each step.",
	)
	parser.add_argument(
		"--log-jsonl",
		default=None,
		help="Write one JSON object per step to the given path during random smoke runs.",
	)
	parser.add_argument("--smoke-episodes", type=int, default=3)
	parser.add_argument("--smoke-steps", type=int, default=300)
	parser.add_argument("--timesteps", type=int, default=100_000)
	parser.add_argument("--model-output", default="models/ppo_thruster_agent")
	return parser.parse_args()


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
) -> None:
	jsonl_path = Path(log_jsonl) if log_jsonl else None
	jsonl_file = None
	if jsonl_path is not None:
		jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		jsonl_file = jsonl_path.open("w", encoding="utf-8")

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
					print(
						"episode=%d steps=%d reward=%.3f reason=%s goal_distance=%.3f relative_speed=%.3f"
						% (
							episode,
							step + 1,
							total_reward,
							last_info.get("terminal_reason", "unknown"),
							float(last_info.get("goal_distance", 0.0)),
							float(last_info.get("relative_speed", 0.0)),
						)
					)
					break
			else:
				print(
					"episode=%d steps=%d reward=%.3f reason=max_steps"
					% (episode, max_steps, total_reward)
				)
	finally:
		if jsonl_file is not None:
			jsonl_file.close()


def run_ppo_training(env: GodotThrusterEnv, timesteps: int, model_output: str) -> None:
	try:
		from stable_baselines3 import PPO
	except ImportError as exc:
		raise SystemExit(
			"stable-baselines3 is not installed. Start with --random-only, or install a supported RL stack. "
			"If Python 3.14 package support lags, use Python 3.12 or 3.13 for training."
		) from exc

	output_path = Path(model_output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=timesteps)
	model.save(output_path.as_posix())


def main() -> None:
	args = parse_args()
	env = build_env(args)
	try:
		run_random_smoke(
			env,
			episodes=args.smoke_episodes,
			max_steps=args.smoke_steps,
			log_steps=args.log_steps,
			log_step_details=args.log_step_details,
			log_jsonl=args.log_jsonl,
		)
		if args.random_only:
			return

		run_ppo_training(env, timesteps=args.timesteps, model_output=args.model_output)
	finally:
		env.close()


if __name__ == "__main__":
	main()