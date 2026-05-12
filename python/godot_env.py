from __future__ import annotations

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

		self.thruster_count = int(hello_response["thruster_count"])
		if self.thruster_count <= 0:
			raise GodotProtocolError(
				f"Expected a positive thruster count from the RL bridge, got {self.thruster_count}"
			)

		self.expected_observation_size = self.thruster_count + OBSERVATION_FIXED_FIELDS
		self.default_action_frames = int(hello_response.get("default_action_frames", step_frames))
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