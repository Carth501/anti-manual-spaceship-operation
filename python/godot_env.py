from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BRIDGE_PROTOCOL_VERSION = 1
OBSERVATION_FIXED_FIELDS = 12


class GodotProtocolError(RuntimeError):
	pass


def resolve_godot_executable(raw_path: str | os.PathLike[str] | None) -> str | None:
	if raw_path is None:
		return None

	path = Path(raw_path).expanduser()
	if path.is_dir():
		candidates = [
			*sorted(path.glob("*_console.exe")),
			*sorted(path.glob("*.exe")),
		]
		for candidate in candidates:
			if candidate.is_file():
				return str(candidate)
		return None

	if path.is_file():
		return str(path)

	return None


class GodotThrusterEnv(gym.Env):
	metadata = {"render_modes": []}

	def __init__(
		self,
		host: str = "127.0.0.1",
		port: int = 8765,
		step_frames: int = 8,
		launch_project: bool = False,
		godot_executable: str | None = None,
		headless: bool = True,
		realtime_delay: float = 0.0,
		connect_timeout: float = 30.0,
		project_path: str | Path | None = None,
	) -> None:
		super().__init__()
		self.host = host
		self.port = port
		self.step_frames = step_frames
		self.launch_project = launch_project
		self.godot_executable = resolve_godot_executable(
			godot_executable or os.environ.get("GODOT_BIN")
		)
		self.headless = headless
		self.realtime_delay = max(float(realtime_delay), 0.0)
		self.connect_timeout = connect_timeout
		self.project_path = Path(project_path or Path(__file__).resolve().parents[1])

		self._socket: socket.socket | None = None
		self._socket_file = None
		self._process: subprocess.Popen[bytes] | None = None

		if self.launch_project:
			self._process = self._launch_godot_process()

		self._connect()
		hello_response = self._send_command({"command": "hello"})
		self.bridge_version = int(hello_response.get("version", -1))
		if self.bridge_version != BRIDGE_PROTOCOL_VERSION:
			raise GodotProtocolError(
				f"Expected RL bridge protocol version {BRIDGE_PROTOCOL_VERSION}, got {self.bridge_version}"
			)

		self.bridge_metadata = self._normalize_bridge_metadata(hello_response)
		self.thruster_count = int(self.bridge_metadata["thruster_count"])
		if self.thruster_count <= 0:
			raise GodotProtocolError(
				f"Expected a positive thruster count from the RL bridge, got {self.thruster_count}"
			)

		self.observation_schema_fields = tuple(
			str(field_name) for field_name in self.bridge_metadata["observation_schema"]["fields"]
		)
		self.expected_observation_size = len(self.observation_schema_fields)
		if self.expected_observation_size <= 0:
			self.expected_observation_size = self.thruster_count + OBSERVATION_FIXED_FIELDS
		self.default_action_frames = int(self.bridge_metadata.get("default_action_frames", step_frames))
		self.environment_fingerprint = str(self.bridge_metadata["environment_fingerprint"])
		self.reward_config_hash = str(self.bridge_metadata["reward_config_hash"])
		initial_observation = self._coerce_observation(hello_response["observation"], context="hello")

		self.action_space = spaces.Box(
			low=0.0,
			high=1.0,
			shape=(self.thruster_count,),
			dtype=np.float32,
		)
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=initial_observation.shape,
			dtype=np.float32,
		)

	def reset(
		self,
		*,
		seed: int | None = None,
		options: dict[str, Any] | None = None,
	) -> tuple[np.ndarray, dict[str, Any]]:
		super().reset(seed=seed)
		response = self._send_command({"command": "reset"})
		observation = self._coerce_observation(response["observation"], context="reset")
		info = dict(response.get("info", {}))
		if options:
			info.update(options)
		return observation, info

	def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
		action_array = np.asarray(action, dtype=np.float32).reshape(-1)
		if action_array.shape != (self.thruster_count,):
			raise ValueError(
				f"Expected action shape {(self.thruster_count,)}, got {action_array.shape}"
			)

		response = self._send_command(
			{
				"command": "step",
				"action": np.clip(action_array, 0.0, 1.0).astype(float).tolist(),
				"frames": self.step_frames,
			}
		)
		observation = self._coerce_observation(response["observation"], context="step")
		reward = float(response["reward"])
		done = bool(response["done"])
		info = dict(response.get("info", {}))
		terminal_reason = str(info.get("terminal_reason", ""))
		terminated = done and terminal_reason != "timeout"
		truncated = done and terminal_reason == "timeout"
		if self.realtime_delay > 0.0:
			time.sleep(self.realtime_delay)
		return observation, reward, terminated, truncated, info

	def close(self) -> None:
		if self._socket_file is not None:
			try:
				self._send_command({"command": "close"})
			except OSError:
				pass
			except GodotProtocolError:
				pass

		if self._socket_file is not None:
			self._socket_file.close()
			self._socket_file = None

		if self._socket is not None:
			self._socket.close()
			self._socket = None

		if self._process is not None:
			self._process.terminate()
			try:
				self._process.wait(timeout=5.0)
			except subprocess.TimeoutExpired:
				self._process.kill()
			self._process = None

	def get_environment_metadata(self) -> dict[str, Any]:
		return copy.deepcopy(self.bridge_metadata)

	def _normalize_bridge_metadata(self, hello_response: dict[str, Any]) -> dict[str, Any]:
		metadata = dict(hello_response)
		thruster_count = int(metadata.get("thruster_count", 0))
		observation_schema = dict(metadata.get("observation_schema") or {})
		observation_fields = observation_schema.get("fields")
		if not isinstance(observation_fields, list) or not observation_fields:
			observation_fields = _default_observation_fields(thruster_count)
		observation_schema["fields"] = [str(field_name) for field_name in observation_fields]
		observation_schema.setdefault("fixed_field_count", max(len(observation_schema["fields"]) - thruster_count - 2, 0))
		observation_schema.setdefault("thruster_field_count", thruster_count)
		observation_schema.setdefault("flag_field_count", 2)
		metadata["observation_schema"] = observation_schema
		metadata.setdefault("reward_config", {})
		metadata.setdefault("goal_config", {})
		metadata.setdefault("ship_config", {})
		metadata.setdefault("thruster_config", {})
		environment_contract = {
			"bridge_version": int(metadata.get("version", -1)),
			"thruster_count": thruster_count,
			"default_action_frames": int(metadata.get("default_action_frames", self.step_frames)),
			"episode_frame_limit": metadata.get("episode_frame_limit"),
			"physics_ticks_per_second": metadata.get("physics_ticks_per_second"),
			"scene_path": metadata.get("scene_path"),
			"spawn_origin": metadata.get("spawn_origin"),
			"observation_schema": observation_schema,
			"goal_config": metadata["goal_config"],
			"ship_config": metadata["ship_config"],
			"thruster_config": metadata["thruster_config"],
		}
		metadata["environment_fingerprint"] = _sha256_json(environment_contract)
		metadata["reward_config_hash"] = _sha256_json(metadata["reward_config"])
		return metadata

	def _launch_godot_process(self) -> subprocess.Popen[bytes]:
		if not self.godot_executable:
			raise ValueError(
				"Set GODOT_BIN or pass godot_executable when launch_project=True. The value can be either the Godot .exe or a directory containing it."
			)

		command = [self.godot_executable]
		if self.headless:
			command.append("--headless")
		# Without --headless Godot opens its normal game window, which is useful
		# for watching the agent train even though observations still come from
		# the structured TCP bridge rather than rendered pixels.
		command.extend(["--path", str(self.project_path)])

		creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
		return subprocess.Popen(
			command,
			cwd=self.project_path,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			creationflags=creationflags,
		)

	def _connect(self) -> None:
		deadline = time.monotonic() + self.connect_timeout
		last_error: OSError | None = None
		while time.monotonic() < deadline:
			if self._process is not None:
				exit_code = self._process.poll()
				if exit_code is not None:
					raise RuntimeError(
						"Godot exited before the RL bridge became available. "
						f"Exit code: {exit_code}. Launch Godot directly to inspect startup errors."
					)

			try:
				self._socket = socket.create_connection((self.host, self.port), timeout=1.0)
				self._socket.settimeout(self.connect_timeout)
				self._socket_file = self._socket.makefile("rwb")
				return
			except OSError as exc:
				last_error = exc
				time.sleep(0.2)

		timeout_message = f"Could not connect to Godot RL bridge at {self.host}:{self.port}"
		if self.launch_project and not self.headless:
			timeout_message += ". Watch mode can take longer to open; try --connect-timeout 90"
		raise TimeoutError(timeout_message) from last_error

	def _send_command(self, command: dict[str, Any]) -> dict[str, Any]:
		if self._socket_file is None:
			raise OSError("Socket is not connected")

		encoded = json.dumps(command).encode("utf-8") + b"\n"
		self._socket_file.write(encoded)
		self._socket_file.flush()

		response_line = self._socket_file.readline()
		if not response_line:
			raise ConnectionError("Godot RL bridge closed the connection")

		response = json.loads(response_line.decode("utf-8"))
		if not response.get("ok", False):
			raise GodotProtocolError(
				f"{response.get('error', 'protocol_error')}: {response.get('message', 'Unknown error')}"
			)
		return response

	def _coerce_observation(self, raw_observation: list[float], *, context: str) -> np.ndarray:
		if not isinstance(raw_observation, list):
			raise GodotProtocolError(
				f"Expected {context} observation to be a JSON array, got {type(raw_observation).__name__}"
			)

		observation = np.asarray(raw_observation, dtype=np.float32).reshape(-1)
		if observation.shape != (self.expected_observation_size,):
			raise GodotProtocolError(
				"Expected %s observation shape %s for %d thrusters, got %s"
				% (
					context,
					(self.expected_observation_size,),
					self.thruster_count,
					observation.shape,
				)
			)

		return observation


def _default_observation_fields(thruster_count: int) -> list[str]:
	fields = [
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
	]
	fields.extend(f"thruster_throttle_{index:02d}" for index in range(thruster_count))
	fields.extend(["is_inside_goal", "is_goal_completed"])
	return fields


def _sha256_json(payload: Any) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()