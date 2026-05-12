# Policy-Level Metadata Plan

Focused next-step plan: `docs/policy_status_lineage_plan.md`

## Goal

Promote policy metadata from a handful of tags into a deliberate catalog schema so policies can be compared, curated, and selected without reading raw manifests.

## Increment 1: Core Catalog Fields

- Keep `policy_id` as the stable curated identifier.
- Add first-class fields for `label`, `persona`, `objective`, `intended_use`, `algorithm`, `training_technique`, `is_best_in_category`, and `notes`.
- Continue storing environment fingerprint and reward hash alongside the policy so selection always carries compatibility context.

## Increment 2: Promotion Workflow

- Add a `promote-run` command that can take an existing tracked run and curate it into the policy catalog.
- Promotion should update both the run manifest and the policy package manifest so the catalog and the package stay aligned.
- Promotion should be idempotent: promoting the same `policy_id` again should replace the policy row rather than duplicate it.

## Increment 3: Selection Metadata

- Add `is_best_in_category` as the first explicit selection field.
- Next additions should be `status` values like `candidate`, `baseline`, `archived`, and `production`.
- After that, add policy lineage fields such as `replaces_policy_id` and `parent_policy_id` if version chains become important.

## Increment 4: Comparative Metadata

- Promote a small set of derived metrics into policy-facing views, not just run-facing views.
- Candidate fields: current best success rate, median goal steps, median goal timesteps, mean episode reward, and latest evaluation timestamp.
- Keep derived metrics separate from operator-authored metadata so rebuilding registries remains deterministic.

## Increment 5: Workflow Documentation

- Document how a run becomes a policy.
- Document when to mint a new `policy_id` versus updating an existing one.
- Document how to mark a policy as best in category and what evidence is expected.
