# Milestone Coverage Plan

Focused next-step plan: `docs/milestone_expansion_plan.md`

## Goal

Expand milestone tracking from a narrow success-rate view into a comparison framework that supports safe, efficient, and aggressive policy families.

## Increment 1: Completion Milestones

- Keep `first_goal_reached`.
- Add threshold milestones for `median_goal_steps` and `median_goal_timesteps` once enough successful episodes exist.
- Add `first_success_streak` style milestones only after the base success metrics are stable.

## Increment 2: Safety Milestones

- Add thresholds for reduced `out_of_bounds` frequency.
- Add thresholds for reduced high-speed goal entries or unsafe terminal conditions once those metrics exist in summaries.
- Treat safety milestones as separate from performance milestones so a policy can improve in one axis without hiding regressions in the other.

## Increment 3: Efficiency Milestones

- Add milestones for lower median steps-to-goal.
- Add milestones for lower median timesteps-to-goal.
- Add fuel or thrust efficiency milestones once per-episode thrust totals are exposed cleanly in summary data.

## Increment 4: Persona-Aware Milestone Sets

- Safe policies should prioritize safety milestones first, then completion milestones.
- Efficient policies should prioritize steps, timesteps, and thrust-efficiency milestones.
- Aggressive policies should prioritize time-to-goal or speed-oriented milestones, but still surface safety regressions explicitly.

## Increment 5: Registry Presentation

- Keep milestone events in `experiments/milestones.csv`.
- Add a derived summary view later that shows the latest achieved milestone per category for each run or policy.
- Do not overload one milestone table with policy curation state; keep curation in the policy registry.
