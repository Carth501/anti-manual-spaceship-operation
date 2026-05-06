# Anti-Manual Spaceship Operation

Godot-based spaceship control sandbox with an external Python reinforcement-learning bridge.

## Overview

This project treats the Godot scene as a reinforcement-learning environment for a spaceship with individually addressable thrusters. Godot owns the physics simulation, scene state, and episode reset logic. Python owns the training loop.

The current integration is state-based rather than image-based:

- Godot exposes a TCP bridge through `RLBridge`.
- Python connects through a Gymnasium-compatible wrapper.
- Each action is one continuous throttle value per active thruster.
- Observations are structured numeric state, not rendered pixels.

This makes headless runs the default for fast training, while still allowing a visible watch mode for debugging and demos.

## Current Status

- External Python smoke tests are working.
- Godot can be launched directly by the Python wrapper.
- Headless and watchable training modes both work.
- The in-game UI includes a training HUD that shows trainer connection state, episode counters, reward totals, and the last terminal reason.
- The first curriculum is soft docking only. Obstacle avoidance is not yet part of the RL task.

## Architecture

The RL loop currently looks like this:

1. Python launches Godot or connects to an already running Godot instance.
2. Python sends a `hello`, `reset`, or `step` command over TCP.
3. Godot applies one throttle value per thruster and advances physics for a fixed number of frames.
4. Godot computes observation, reward, and done state.
5. Godot sends that result back to Python as JSON.

The main pieces are:

- `scripts/rl_bridge.gd`: synchronous TCP bridge that turns the scene into an RL environment
- `scripts/control_interface.gd`: switches between manual controls and RL thruster control
- `scripts/thruster_controller.gd`: applies per-thruster throttle values to the ship
- `python/godot_env.py`: Gymnasium-compatible Python client for the Godot bridge
- `python/train.py`: smoke-test and training entry point

## Action, Observation, and Reward

### Action Space

- One continuous value in the range `[0, 1]` per active thruster
- The action size is discovered from the live Godot scene at startup
- The current ship setup exposes 13 thrusters

### Observation Layout

Observations are mostly ship-local values so the policy learns control relationships instead of memorizing world orientation.

Current observation layout:

1. Goal offset in ship-local coordinates: `x, y, z`
2. Linear velocity in ship-local coordinates: `x, y, z`
3. Angular velocity in ship-local coordinates: `x, y, z`
4. Relative speed to the goal
5. Current throttle for each thruster
6. Inside-goal flag
7. Goal-complete flag

### Reward Shaping

The current reward is a simple shaped signal for the first docking curriculum:

- positive reward for reducing distance to the goal
- penalty for high relative speed
- penalty for high total thruster use
- small living penalty per step
- large success bonus for entering the goal zone under the speed threshold
- penalty for going out of bounds

### Episode Termination

Episodes currently end when one of these happens:

- the ship reaches the goal correctly
- the ship leaves the training boundary
- the episode hits the frame limit

## Requirements

### Godot

- Godot 4.6 or compatible 4.x console build
- The wrapper accepts either the Godot executable itself or a folder containing it

Example validated path on this machine:

```powershell
C:\Users\carth\Godot\Godot 4.6
```

### Python

- Python is used for the environment wrapper and training scripts
- `gymnasium` and `numpy` are enough for smoke tests
- `stable-baselines3` is intended for PPO training, but Python 3.14 support may lag upstream
- If PPO install support is inconsistent, prefer Python 3.12 or 3.13 for training

## Setup

### 1. Create a Virtual Environment

Windows PowerShell example:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python Dependencies

```powershell
python -m pip install -r python/requirements.txt
```

If you want PPO training on a Python version that supports it, install Stable-Baselines3 as well.

### 3. Point Python at Godot

Either pass the path on the command line:

```powershell
python python/train.py --launch-project --godot-executable "C:\Path\To\Godot"
```

Or set an environment variable:

```powershell
$env:GODOT_BIN = "C:\Path\To\Godot"
```

## Running the Environment

### Headless Smoke Test

This is the fastest way to confirm the bridge is working.

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --random-only --smoke-episodes 1 --smoke-steps 50
```

### Watch Mode

Watch mode opens the normal Godot window and slows down the step loop slightly so you can inspect what the ship is doing.

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --watch --watch-step-delay 0.05 --random-only --smoke-episodes 1 --smoke-steps 50
```

If the first visible launch is slow and you hit a bridge timeout, retry with a longer startup window:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --watch --watch-step-delay 0.05 --connect-timeout 90 --random-only --smoke-episodes 1 --smoke-steps 50
```

Notes:

- Watch mode is still using structured state observations over TCP.
- The agent is not learning from pixels.
- Headless mode remains the better choice for long training runs.
- Add `--log-steps` to print a concise per-step summary during smoke runs.
- Add `--log-step-details` to print the full debug payload returned by `RLBridge`.
- Add `--log-jsonl logs/random_smoke.jsonl` to capture one JSON object per step for later analysis.

### Connect to an Already Running Godot Instance

If Godot is already open and the RL bridge is active, omit `--launch-project`:

```powershell
python python/train.py --random-only --smoke-episodes 1 --smoke-steps 50
```

## Training HUD

When the normal game window is visible, the UI now shows a training panel with:

- trainer connection status
- current phase
- episode number
- episode frame count
- total reward for the current episode
- reward from the most recent step
- last terminal reason

This panel is for debugging and inspection only. It does not affect the RL logic.

## Training Notes

- The script currently runs a short random-policy smoke phase before optional PPO training.
- The PPO path lives in `python/train.py` and expects Stable-Baselines3 to be importable.
- The current default step size is 4 physics frames per action.
- The environment is currently single-instance and local.

## Project Layout

- `scenes/space.tscn`: main environment scene
- `scenes/user_interface.tscn`: in-game UI including the training HUD
- `scripts/rl_bridge.gd`: TCP bridge between Godot and Python
- `scripts/ui_manager.gd`: updates HUD and manual flight UI
- `scripts/thruster_controller.gd`: thruster application logic
- `scripts/thruster_point.gd`: per-thruster force and torque behavior
- `python/godot_env.py`: Python environment wrapper
- `python/train.py`: smoke test and training entry point
- `python/requirements.txt`: Python dependency baseline

## Current Limitations

- The first RL task is docking only; obstacle avoidance is not yet part of the environment logic.
- The large obstacle meshes in the scene are not yet part of the RL curriculum.
- Training currently depends on a live Godot process rather than an exported standalone simulator binary.
- The observation schema is code-defined rather than separately versioned in docs or tests.

## TODO

- [ ] Add a slower visual evaluation mode for replaying a saved policy in the Godot window.
- [ ] Add obstacle collision handling and fold it into the training curriculum.
- [ ] Add saved-model evaluation and policy replay tooling.
