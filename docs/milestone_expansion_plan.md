# Milestone Expansion Plan

## Goal

Expand milestone tracking from success-rate-only signals into a practical comparison layer for safe, efficient, and aggressive policy families.

## Scope

- Add milestone types for completion quality, safety, and efficiency.
- Keep milestone generation centralized in `EpisodeAccumulator` and summary-building code.
- Continue storing milestone events in `experiments/milestones.csv`.

## Out Of Scope

- Persona-specific reward shaping changes.
- Visualization dashboards.
- Automatic policy promotion based on milestones.

## Metrics To Add First

### Completion Metrics

- `median_goal_steps`
- `median_goal_timesteps`
- `median_goal_frames`

### Safety Metrics

- `out_of_bounds_rate`
- `timeout_rate`
- optional later: `unsafe_goal_entry_rate` once that signal exists explicitly

### Efficiency Metrics

- `mean_thruster_usage_per_episode`
- `median_thruster_usage_on_success`
- optional later: thrust-per-meter or thrust-per-success metrics

## Implementation Steps

### Increment 1: Episode Summary Inputs

- Extend training and evaluation episode recording so per-episode summaries include the inputs needed for milestone comparisons.
- Reuse `reward_terms` where possible instead of adding parallel instrumentation.
- Capture per-episode thrust totals from reward or debug payloads if they are already available; otherwise add the smallest needed summary field.

### Increment 2: Derived Summary Metrics

- Extend `EpisodeAccumulator.build_summary()` in `python/run_tracking.py` to compute the new safety and efficiency aggregates.
- Keep new summary fields nullable until enough completed episodes exist to compute them safely.

### Increment 3: Milestone Generation

- Add threshold milestones for `median_goal_steps` and `median_goal_timesteps`.
- Add rate-based milestones for improved `out_of_bounds_rate` and reduced `timeout_rate`.
- Keep milestone keys stable and machine-readable so rebuilds produce identical rows.

### Increment 4: Registry Surface

- Add the new summary metrics to `experiments/runs.csv` once they are stable.
- Keep detailed milestone events in `experiments/milestones.csv`.
- Consider a later derived summary file for the latest milestone per run or per policy, but not in this slice.

### Increment 5: Persona Mapping

- Safe policies should be evaluated primarily on safety milestones first.
- Efficient policies should be evaluated primarily on timesteps, steps, and thrust-use milestones.
- Aggressive policies should be evaluated on time-to-goal style milestones, but with visible safety regression metrics next to them.

## Files Likely To Change

- `python/run_tracking.py`
- `python/train.py`
- `scripts/rl_bridge.gd` only if a missing per-episode signal must be exposed
- `README.md`

## Validation

Prefer synthetic accumulator tests over full training jobs for most of this slice. The risky part is metric derivation and milestone emission, not PPO itself.

### Summary Derivation Cases

1. Feed episodes with zero successes and verify `median_goal_steps`, `median_goal_timesteps`, and `median_goal_frames` stay unset.
2. Feed mixed success and failure episodes and verify completion metrics are computed from successful episodes only.
3. Feed episodes with known `out_of_bounds` and timeout terminal states and verify `out_of_bounds_rate` and `timeout_rate` are computed exactly.
4. Feed episodes with known thrust totals and verify `mean_thruster_usage_per_episode` and `median_thruster_usage_on_success` match expected aggregates.
5. Feed episodes where thrust totals are missing and verify efficiency fields stay nullable instead of falling back to incorrect zeros.

### Milestone Emission Cases

1. Verify a run can trigger a completion milestone without triggering a safety milestone.
2. Verify a run can trigger a safety milestone without improving completion metrics.
3. Verify milestone keys remain stable when the same summary is rebuilt from stored manifests.
4. Verify zero-success runs do not emit completion-quality milestones.

### Registry And Rebuild Cases

1. Run a short smoke or PPO job and verify the new summary fields appear without breaking existing `experiments/runs.csv` rows.
2. Rebuild tracking and confirm the new milestone rows are reproduced deterministically.
3. Verify old manifests that lack the new summary fields remain rebuildable, with the new columns left blank.
4. Verify milestone rows stay append-safe and machine-readable after multiple rebuilds.

### Failure And Edge Cases

1. Verify summary building never divides by zero when episode counts or success counts are zero.
2. Verify partial data does not create contradictory metrics, such as success-rate milestones with missing completion medians.
3. Verify the training path still works when the bridge does not yet expose an optional efficiency signal, with null metrics rather than runtime failure.

### Suggested Test Split

1. Unit-style tests for `EpisodeAccumulator.build_summary()` using synthetic episode fixtures.
2. Unit-style tests for milestone generation from prebuilt summary dictionaries.
3. One end-to-end smoke job to confirm CSV compatibility and one rebuild test to confirm determinism.

## Open Questions

- Should timeout reduction milestones be absolute thresholds, relative improvements, or both?
- Should efficiency milestones be based on raw thrust sum, normalized thrust sum, or both?
