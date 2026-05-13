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
- New runs now write manifests and CSV registries under `experiments/`.
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

Note: the current observation still does not include explicit thruster geometry features such as each thruster's fixed force direction or torque contribution. The policy is still expected to infer that mapping from interaction, and exposing those geometry terms remains a useful future improvement if direct per-thruster learning stays sample-inefficient.

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

Relevant args:

- `--launch-project`: start the Godot project from Python before connecting
- `--godot-executable PATH`: point at the Godot executable or a folder that contains it

Or set an environment variable:

```powershell
$env:GODOT_BIN = "C:\Path\To\Godot"
```

## Running the Environment

### Experiment Tracking

New runs are tracked automatically. The first pass starts clean: existing `logs/` and `models/` artifacts are not backfilled.

Tracked outputs currently include:

- `experiments/runs.csv`: one row per tracked run
- `experiments/milestones.csv`: milestone hits such as first goal or success-rate thresholds
- `experiments/policies.csv`: curated policy registry for runs promoted during training with `--policy-id` or later with `--promote-run`
- `experiments/runs/<run_id>/manifest.json`: full run metadata, environment contract, and CLI arguments
- `experiments/runs/<run_id>/summary.json`: derived per-run summary metrics

Training runs now save policies as package directories by default. For a model output like `models/ppo_baseline`, the package layout is:

- `models/ppo_baseline/policy.zip`: the saved Stable-Baselines3 policy
- `models/ppo_baseline/manifest.json`: policy metadata and environment compatibility data
- `models/ppo_baseline/summary.json`: derived summary metrics for that policy package

Legacy flat files such as `models/ppo_baseline.zip` with nearby metadata are still loadable for backward compatibility.

Current run-level metadata includes:

- persona tags such as `aggressive`, `efficient`, or `safe`
- training-technique labels for comparing reward structures or curricula
- environment fingerprint and reward-config hash
- average and median goal steps, frames, and timesteps when goals are reached
- milestone timestamps and timesteps for threshold-based comparisons

Current policy-level metadata includes:

- curated policy id and label
- persona, objective, intended use, and algorithm
- training technique and best-in-category flag
- policy notes, environment fingerprint, reward-config hash, and source run id
- policy-facing comparison fields including success rate, mean episode reward, mean and median goal steps, median goal timesteps, source run finished timestamp, latest evaluation timestamp, and the last policy-row update timestamp

Useful tracking flags:

- `--tracking-dir experiments`: change where manifests and CSV registries are written
- `--run-label baseline-v1`: human-readable label for the tracked run
- `--persona safe`: behavior category tag for the run or promoted policy
- `--training-technique reward_shaping_v2`: label for the training structure or recipe
- `--policy-id safe-docker-v1`: optional curated policy identifier to upsert into `policies.csv`
- `--policy-label "Safe Docker V1"`: curated display label for the policy catalog
- `--policy-objective "Dock reliably without high-speed goal entries"`: short optimization target
- `--policy-intended-use "Reference safe docking baseline"`: operator-facing usage note
- `--policy-algorithm PPO`: algorithm label stored with policy metadata
- `--policy-best-in-category`: mark the policy as the current best choice in its category
- `--policy-notes "Promoted after regression checks"`: curated notes stored with the policy metadata
- `--run-notes "reduced speed penalty near goal"`: freeform notes stored in the manifest

Focused planning docs for the next increments live in:

- `docs/policy_metadata_plan.md`
- `docs/milestone_coverage_plan.md`

Useful admin commands:

- `--promote-run RUN_ID_OR_PATH`: curate an existing tracked run into the policy catalog
- `--tracking-dir PATH`: choose the tracking directory; defaults to `experiments`
- `--policy-id NAME`: upsert the curated policy row when promoting
- `--rebuild-tracking`: rebuild `runs.csv`, `policies.csv`, and `milestones.csv` from tracked run manifests and exit
- combine `--promote-run` with the `--policy-*`, `--persona`, and `--training-technique` flags when you want curated metadata on the promoted entry

### Promotion Workflow

`experiments/runs.csv` is the full run ledger. `experiments/policies.csv` is the curated shortlist you use when choosing which saved policy should stay in active consideration. Promotion is the step that turns one tracked run into one curated policy entry.

Recommended workflow:

1. Run training or evaluation with tracking enabled so the run writes `experiments/runs/<run_id>/manifest.json` and `summary.json`.
2. Review the run in `experiments/runs.csv` and, when needed, open the run summary to inspect success rate, goal timing metrics, and terminal-reason counts.
3. Promote the run with `--promote-run <run_id>` and the curated policy fields you want to keep with that artifact.
4. Use the same `--policy-id` when you want to revise metadata for the same curated entry. Use a new `--policy-id` when you want a distinct catalog entry for side-by-side comparison.

Promotion currently does all of the following in one step:

- updates the `policy` block inside `experiments/runs/<run_id>/manifest.json`
- writes matching metadata into the saved policy package manifest under `models/<policy>/manifest.json` or the legacy sidecar location
- writes the promoted run summary into the package `summary.json` next to the saved model
- rebuilds `experiments/runs.csv`, `experiments/policies.csv`, and `experiments/milestones.csv` from tracked run manifests

Two common promotion patterns are supported:

- promote during training by passing `--policy-id` and the other `--policy-*` flags directly on the training command
- promote later with `--promote-run` after you have inspected the finished run and decided it belongs in the curated catalog

Use `--rebuild-tracking` any time you want to regenerate the CSV registries from the run manifests. The manifests are the source of truth; the CSVs are rebuildable views.

### Comparison Workflow

There is no dedicated compare subcommand yet. The intended comparison workflow is to use the tracked CSVs and summaries as layered views of the same data.

Recommended workflow:

1. Start in `experiments/runs.csv` when you want to compare every tracked run, including candidates that have not been curated into the policy catalog yet.
2. Filter to compatible runs first. Matching `environment_fingerprint` is the safest apples-to-apples check, and matching `reward_config_hash` means the reward shaping was also identical.
3. Compare `success_rate`, `median_goal_steps`, `median_goal_timesteps`, `mean_episode_reward`, and `first_success_timestep` together instead of ranking by reward alone.
4. Open the run `summary.json` when two runs look close. `terminal_reason_counts` is often the fastest way to spot hidden failure modes such as repeated timeouts or out-of-bounds endings.
5. Once a run deserves to stay in the curated set, compare it in `experiments/policies.csv` against other promoted policies in the same persona or intended-use bucket.
6. Use `experiments/milestones.csv` when you care about when a threshold was first reached, not just the final aggregate metrics.

For policy rows, interpret the timestamps this way:

- `source_run_finished_at`: when the originating non-evaluation source run completed
- `latest_evaluation_at`: when the policy was last refreshed by an evaluation run
- `updated_at`: when the policy row was last refreshed by any tracked run

Use the artifacts this way:

- `experiments/runs.csv`: broad experiment comparison across all tracked runs
- `experiments/policies.csv`: curated policy comparison after promotion
- `experiments/milestones.csv`: threshold-achievement timeline for each run
- `experiments/runs/<run_id>/summary.json`: detailed rollup for one tracked run
- `models/<policy>/summary.json`: the promoted policy's summary stored next to the saved model package

PowerShell examples:

Top compatible safe runs:

```powershell
Import-Csv experiments/runs.csv |
	Where-Object { $_.persona -eq "safe" -and $_.environment_fingerprint -eq "<fingerprint>" } |
	Sort-Object @{ Expression = { if ($_.success_rate) { [double]$_.success_rate } else { 0.0 } }; Descending = $true }, @{ Expression = { if ($_.median_goal_steps) { [double]$_.median_goal_steps } else { [double]::PositiveInfinity } } } |
	Select-Object -First 10 run_id, label, training_technique, success_rate, median_goal_steps, first_success_timestep
```

Top curated safe policies:

```powershell
Import-Csv experiments/policies.csv |
	Where-Object { $_.persona -eq "safe" } |
	Sort-Object @{ Expression = { if ($_.is_best_in_category -eq "true") { 1 } else { 0 } }; Descending = $true }, @{ Expression = { if ($_.success_rate) { [double]$_.success_rate } else { 0.0 } }; Descending = $true }, @{ Expression = { if ($_.median_goal_steps) { [double]$_.median_goal_steps } else { [double]::PositiveInfinity } } } |
	Format-Table policy_id, label, source_run_id, success_rate, median_goal_steps, is_best_in_category
```

### Build Commands From Flags

Start every smoke, evaluation, or PPO run with:

```powershell
python python/train.py [flags]
```

Placeholders used below:

- `PATH`: filesystem path such as the Godot executable folder, a model directory, or a JSONL output path
- `N`: integer count such as episodes, steps, timesteps, frames, or PPO rollout size
- `X`: floating-point seconds for delays or timeouts
- `LABEL`, `TAG`, `NAME`, `TEXT`: freeform metadata stored with tracked runs or promoted policies

Recommended values below are starting points for the current docking curriculum, not hard limits.

Shared launch and connection flags:

- `--launch-project`: start the Godot project from Python before connecting. Recommended: enable this for most local runs; omit it only when Godot is already running and the bridge is ready.
- `--godot-executable PATH`: point at the Godot executable or a folder that contains it. Recommended: use a local Godot 4.6 install path, and prefer the console build for headless runs.
- `--host HOST`: TCP host for the RL bridge; defaults to `127.0.0.1`. Recommended: keep `127.0.0.1` unless you intentionally run the bridge on another machine or interface.
- `--port PORT`: TCP port for the RL bridge; defaults to `8765`. Recommended: keep `8765` unless it conflicts, then choose another unused high port such as `8766-8799`.
- `--connect-timeout X`: wait up to `X` seconds for the bridge connection. Recommended: `15-30` for headless launches, `30-90` for first visible launches, and `5-15` when connecting to an already running instance.
- `--frames-per-step N`: advance `N` physics frames for each environment step; larger values make each action last longer in simulation time. Default: `8`. Recommended: `4-12` for most runs, with `8` as the baseline starting point.

### Headless Smoke Test

This is the fastest way to confirm the bridge is working.

Relevant args:

- `--random-only`: run the random-policy smoke path and exit instead of training PPO. Recommended: enable this for bridge validation runs and omit it for PPO training.
- `--smoke-episodes N`: run `N` smoke episodes. Default: `3`. Recommended: `1-3`, with `1` as the fastest bridge check.
- `--smoke-steps N`: cap each smoke episode at `N` environment steps. Default: `300`. Recommended: `25-100` for standalone smoke checks, with `50` as a good starting point.
- `--run-label LABEL`: optional tracked run label for this smoke pass. Recommended: a short hyphenated slug with `2-4` words, such as `smoke-baseline` or `bridge-check`.
- `--persona TAG`: optional behavior tag stored with the run, such as `safe`. Recommended: reuse a short stable vocabulary such as `safe`, `aggressive`, or `efficient`.
- `--training-technique NAME`: optional label for the smoke recipe, such as `smoke_validation`. Recommended: use a short slug that identifies the setup, usually `1-4` underscore-separated terms.

### Watch Mode

Watch mode opens the normal Godot window and slows down the step loop slightly so you can inspect what the ship is doing.

Relevant args:

- `--watch`: open the Godot instance in a visible window and disable headless mode. Recommended: use this for debugging, demos, and replay inspection; leave it off for long training runs.
- `--watch-step-delay X`: add `X` seconds of real-time delay after each environment step while `--watch` is enabled. Default: `0.05`. Recommended: `0.02-0.10` for readable playback, or `0.0-0.02` if you want a visible but faster run.
- `--connect-timeout X`: increase startup wait time if the first visible launch is slow. Recommended: `30-90` for watch mode, especially on the first startup.
- `--log-steps`: print a concise per-step summary during smoke or evaluation runs. Recommended: enable it for short visible debugging runs and disable it for long replays.
- `--log-step-details`: print the full debug payload returned by `RLBridge` for each step. Recommended: use it only for very short debugging passes, typically `1` episode with low step counts, because output grows quickly.
- `--log-jsonl PATH`: write one JSON object per step to `PATH` for later analysis. Recommended: store these under `logs/` with a scenario-specific filename such as `logs/random_smoke.jsonl`.

Notes:

- Watch mode is still using structured state observations over TCP.
- The agent is not learning from pixels.
- Headless mode remains the better choice for long training runs.

### Connect to an Already Running Godot Instance

If Godot is already open and the RL bridge is active, omit `--launch-project`:

Relevant args:

- omit `--launch-project`: connect to the existing bridge instead of starting Godot. Recommended: do this only after you have manually opened Godot and confirmed the bridge is listening.
- `--host HOST`: use a non-default host if the bridge is not on `127.0.0.1`. Recommended: keep `127.0.0.1` for local runs.
- `--port PORT`: use a non-default TCP port if needed. Recommended: keep `8765` unless the default port is already in use.
- `--connect-timeout X`: control how long Python waits for the bridge to accept the connection. Recommended: `5-15` is usually enough when the instance is already running.
- combine this with `--random-only` for smoke, `--eval-model PATH` for replay, or the PPO flags below for training. Recommended: pair it with `--smoke-episodes 1-3` for smoke or `--eval-episodes 3-10` for replay.

### Saved Policy Evaluation

Once you have a saved PPO model, you can replay it through the same bridge without starting a new training run. With the default package layout, pass the package directory to `--eval-model`.

Headless evaluation args:

- `--eval-model PATH`: load the saved PPO policy at `PATH` and run evaluation instead of training. Recommended: point at a package directory under `models/`, such as `models/ppo_baseline`; the loader also accepts direct `.zip` paths.
- `--eval-episodes N`: run `N` evaluation episodes. Default: `3`. Recommended: `3-10` for quick checks and `10-20` when comparing candidate policies.
- `--eval-steps N`: cap each evaluation episode at `N` environment steps. Default: `300`. Recommended: `200-400` so the policy has time to finish the docking task without hiding repeated timeouts.
- `--stochastic-eval`: sample actions during evaluation instead of using deterministic policy output. Recommended: leave this off for regressions and side-by-side comparisons; enable it only when you want to inspect action variability.

Visible replay adds:

- `--watch`: open the Godot instance in a visible window and disable headless mode during replay. Recommended: use it for visual inspection, not for bulk evaluation sweeps.
- `--watch-step-delay X`: add `X` seconds of delay after each evaluation step so the replay is readable. Recommended: `0.02-0.10`, with `0.05` as a good default.
- `--connect-timeout X`: increase startup wait time if the first visible launch is slow. Recommended: `30-90`.
- `--log-steps`: print concise per-step summaries during replay. Recommended: turn it on when replaying `1-3` episodes interactively.
- `--log-step-details`: print the full debug payload for each replay step. Recommended: keep it for one short debug replay only.
- `--log-jsonl PATH`: write per-step evaluation data to `PATH`. Recommended: save under `logs/` with a policy-specific name such as `logs/ppo_baseline_eval.jsonl`.

Notes:

- Evaluation checks the saved model action and observation shapes against the live environment before rollout starts.
- If the model package contains `manifest.json`, evaluation also checks observation schema order, action-frame settings, and environment fingerprint. Reward-config drift is warned about but does not block evaluation.
- Use `--stochastic-eval` if you want sampled rather than deterministic actions.
- `--log-steps`, `--log-step-details`, and `--log-jsonl` work during evaluation too.

## Baseline Training

The first useful PPO run for this project should stay simple:

- use the fixed-start docking task that already exists in the scene
- train headless for speed
- keep the pre-training smoke phase short
- save a checkpoint that you can replay in watch mode afterward

Relevant args:

- `--smoke-episodes N`: run `N` random smoke episodes before PPO training starts. Default: `3`. Recommended: `1-2`, with `1` as the normal baseline choice.
- `--smoke-steps N`: cap each pre-training smoke episode at `N` environment steps. Default: `300`. Recommended: `8-32`, with `8` for minimal overhead and `16-32` when you want a slightly stronger preflight check.
- `--timesteps N`: train PPO for `N` total environment timesteps. Default: `100000`. Recommended: `50_000-200_000` for a first useful baseline, or `32-512` for a pure training smoke test.
- `--ppo-n-steps N`: use rollout length `N` for each PPO update; smaller values are useful for short smoke runs. Default: `256`. Recommended: `128-512` for baseline runs and `32-64` for smoke tests.
- `--ppo-batch-size N`: use batch size `N` for PPO updates. Default: `64`. Recommended: `32-128` for baseline runs, keep it less than or equal to `--ppo-n-steps`, and use `32` for tiny smoke tests.
- `--model-output PATH`: save the trained policy package under `PATH`. Recommended: write to a descriptive directory under `models/`, such as `models/ppo_baseline_safe`, and omit any file extension.
- `--training-log-jsonl PATH`: write per-episode PPO summaries to `PATH`; if omitted, the default is `logs/<model_name>_training.jsonl`. Recommended: keep these under `logs/` with a name derived from the model slug.

Tracking and policy metadata you can add to the same training command:

- `--run-label LABEL`: human-readable run name stored in tracking output. Recommended: a short slug with `2-5` terms, such as `baseline-safe-v1`.
- `--persona TAG`: behavior category for the run or promoted policy, such as `safe`. Recommended: keep the vocabulary small and reusable, for example `safe`, `aggressive`, or `efficient`.
- `--training-technique NAME`: label for the reward recipe, curriculum, or training setup. Recommended: use `1-4` concise slug terms such as `ppo_reward_baseline`.
- `--policy-id NAME`: curated policy identifier to upsert into `policies.csv`. Recommended: use a stable slug with a version suffix, such as `safe-docker-v1`.
- `--policy-label LABEL`: curated display label for the promoted policy. Recommended: `2-6` human-readable words.
- `--policy-objective TEXT`: short statement of what the policy is optimized to do. Recommended: a single sentence fragment of roughly `5-15` words.
- `--policy-intended-use TEXT`: operator-facing note describing when to use the policy. Recommended: a short sentence fragment of roughly `5-15` words.
- `--policy-algorithm NAME`: algorithm label stored with policy metadata, such as `PPO`. Recommended: use `PPO` for current runs unless you add another training path.
- `--policy-best-in-category`: mark the policy as the current best option in its category. Recommended: set this on only one current preferred policy per persona or use-case bucket.
- `--policy-notes TEXT`: curated notes stored with the policy metadata. Recommended: `1-2` short sentences summarizing why the policy was promoted.
- `--run-notes TEXT`: freeform notes stored only with the tracked run manifest. Recommended: `1-2` short sentences describing what changed in this run.

What this does:

- runs the configured smoke pass first so the bridge is exercised before PPO starts
- trains PPO for the total from `--timesteps` using the rollout and batch sizes you choose
- writes the saved policy package under the path from `--model-output`
- writes per-episode PPO training summaries to the default or custom JSONL path
- writes a run manifest and summary under `experiments/runs/<run_id>/`
- updates `experiments/runs.csv` and, when `--policy-id` is set, `experiments/policies.csv`

Each PPO training JSONL record includes:

- a metadata header record as the first line
- episode reward total
- episode step and frame counts
- terminal reason or incomplete-training marker
- final goal distance and relative speed
- aggregated reward terms for that episode

If you want a different training-log path, pass `--training-log-jsonl`.

For a quick training smoke test instead of a real baseline, reduce these values:

- `--timesteps N`: use a small total such as `32-512`; `64` is a reasonable smoke-test starting point.
- `--ppo-n-steps N`: use a short rollout such as `32-64`.
- `--ppo-batch-size N`: use a small batch such as `32`, and keep it less than or equal to `--ppo-n-steps`.
- `--model-output PATH`: write to a disposable output path such as `models/ppo_smoke_agent`.
- keep `--smoke-episodes N` in the `1-2` range and `--smoke-steps N` in the `8-16` range.

### Can I Watch?

Yes, but watch mode is mainly for inspection, not for the main long run.

Visible training adds these flags on top of the baseline training flags:

- `--watch`: open the Godot instance in a visible window and disable headless mode during training. Recommended: use it only for inspection runs, not the main long baseline.
- `--watch-step-delay X`: slow each environment step by `X` seconds so motion is readable. Recommended: `0.02-0.05` for visible training, which is usually enough to follow the ship without dragging the run too far.
- keep the same `--smoke-episodes N`, `--smoke-steps N`, `--timesteps N`, `--ppo-n-steps N`, `--ppo-batch-size N`, and `--model-output PATH` values you would use for the matching headless run. Recommended: if you insist on visible training, stay near the low end of the baseline ranges so the run remains manageable.

Recommended workflow:

1. Train headless.
2. Save a checkpoint.
3. Replay the saved model in watch mode with `--eval-model PATH`, `--watch`, and `--watch-step-delay X`.

### Promote a Tracked Run

Relevant args:

- `--promote-run RUN_ID_OR_PATH`: select the tracked run to promote by run id or manifest path. Recommended: use the full run id from `experiments/runs.csv`; use a manifest path mainly for automation.
- `--tracking-dir PATH`: directory holding `runs.csv`, `policies.csv`, `milestones.csv`, and run manifests; defaults to `experiments`. Recommended: keep `experiments` unless you intentionally maintain separate registries.
- `--policy-id NAME`: curated policy identifier to upsert into `policies.csv`. Recommended: a stable slug with a version suffix such as `safe-docker-v1`.
- `--policy-label LABEL`: curated display label for the promoted policy. Recommended: `2-6` human-readable words.
- `--persona TAG`: behavior category for the curated entry. Recommended: reuse the same small vocabulary as training runs, such as `safe`, `aggressive`, or `efficient`.
- `--training-technique NAME`: label for the training recipe used by the promoted run. Recommended: `1-4` slug terms matching the originating run.
- `--policy-objective TEXT`: short statement of the optimization target. Recommended: `5-15` words.
- `--policy-intended-use TEXT`: operator-facing note describing when to use the policy. Recommended: `5-15` words.
- `--policy-algorithm NAME`: algorithm label stored with the policy metadata. Recommended: use `PPO` for the current training pipeline.
- `--policy-best-in-category`: mark the promoted policy as the preferred entry in its category. Recommended: set this on only the single policy you currently want operators to prefer.
- `--policy-notes TEXT`: curated notes stored with the promoted policy. Recommended: `1-2` short sentences on why the run earned promotion.

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
- The current default step size is 8 physics frames per action.
- Direct thruster commands are now slew-limited so RL control targets change smoothly instead of teleporting from one throttle value to another in a single physics tick.
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
