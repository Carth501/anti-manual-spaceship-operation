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
py -3.13 -m venv .venv313
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If you already have an older repo-local environment, keep it isolated and use `.venv313` for RL training and policy evaluation.

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

### Saved Policy Evaluation

Once you have a saved PPO model, you can replay it through the same bridge without starting a new training run.

Headless evaluation:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --eval-model models/ppo_thruster_agent --eval-episodes 3 --eval-steps 300
```

Visible replay:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --watch --watch-step-delay 0.05 --eval-model models/ppo_thruster_agent --eval-episodes 3 --eval-steps 300
```

Notes:

- Evaluation checks the saved model action and observation shapes against the live environment before rollout starts.
- Use `--stochastic-eval` if you want sampled rather than deterministic actions.
- `--log-steps`, `--log-step-details`, and `--log-jsonl` work during evaluation too.

## Baseline Training

The first useful PPO run for this project should stay simple:

- use the fixed-start docking task that already exists in the scene
- train headless for speed
- keep the pre-training smoke phase short
- save a checkpoint that you can replay in watch mode afterward

Recommended baseline command:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --smoke-episodes 1 --smoke-steps 8 --timesteps 100000 --ppo-n-steps 256 --ppo-batch-size 64 --model-output models/ppo_baseline
```

What this does:

- runs one short random smoke episode first so the bridge is exercised before PPO starts
- trains PPO for 100,000 timesteps with a modest rollout size
- writes the saved model to `models/ppo_baseline.zip`
- writes per-episode PPO training summaries to `logs/ppo_baseline_training.jsonl` by default

Each PPO training JSONL record includes:

- episode reward total
- episode step and frame counts
- terminal reason or incomplete-training marker
- final goal distance and relative speed
- aggregated reward terms for that episode

If you want a different training-log path, pass `--training-log-jsonl`.

For a quick training smoke test instead of a real baseline, reduce the timesteps and rollout size:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --smoke-episodes 1 --smoke-steps 8 --timesteps 64 --ppo-n-steps 32 --ppo-batch-size 32 --model-output models/ppo_smoke_agent
```

### Can I Watch?

Yes, but watch mode is mainly for inspection, not for the main long run.

Visible training:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --watch --watch-step-delay 0.02 --smoke-episodes 1 --smoke-steps 8 --timesteps 100000 --ppo-n-steps 256 --ppo-batch-size 64 --model-output models/ppo_baseline_watch
```

Recommended workflow:

1. Train headless.
2. Save a checkpoint.
3. Replay the saved model in watch mode.

Example replay command:

```powershell
python python/train.py --launch-project --godot-executable "C:\Users\carth\Godot\Godot 4.6" --watch --watch-step-delay 0.05 --eval-model models/ppo_baseline.zip --eval-episodes 3 --eval-steps 300
```

### What To Watch In The Results

During PPO training, Stable-Baselines3 prints periodic summaries. The most useful fields for a first pass are:

- `total_timesteps`: confirms training is still advancing
- `fps`: rough throughput of the training loop
- `approx_kl`: how aggressively the policy is changing between updates
- `clip_fraction`: how often PPO clipping is active; this helps indicate whether updates are too mild or too aggressive
- `entropy_loss`: a rough proxy for policy randomness; very high exploration should reduce over time
- `policy_gradient_loss`: policy update signal
- `value_loss`: critic fit quality; large unstable jumps can indicate reward-scaling or learning instability
- `explained_variance`: critic usefulness; for this project, it may stay noisy early and is more useful as a trend than a target number

Also pay attention to the episode summaries printed by this project before or during evaluation:

- total episode reward
- terminal reason such as `goal_reached`, `out_of_bounds`, `timeout`, or `max_steps`
- final goal distance
- final relative speed

For the first baseline, useful signs of progress are:

- total reward becomes less negative or starts trending positive on some episodes
- more episodes end closer to the goal
- fewer episodes end far away or immediately drift out of bounds
- replayed checkpoints show more directed movement instead of random thruster use

Signs that the reward or training setup likely still needs work:

- almost every episode times out with little change in behavior
- the ship spins or jitters in place without approaching the goal
- the policy saturates many thrusters continuously
- reward looks flat even though the motion visibly changes

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
- The recommended training interpreter for this repo is Python 3.13 in `.venv313`.
- `--ppo-n-steps` and `--ppo-batch-size` exist mainly so short PPO smoke tests and modest baseline runs are easier to control from the CLI.

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

- [ ] Add obstacle collision handling and fold it into the training curriculum.
- [ ] Add a parser or plotter for JSONL step logs so reward, distance, and thruster usage can be graphed.
- [ ] Add per-episode aggregate summaries such as mean throttle use, closest approach, and top-used thrusters.
