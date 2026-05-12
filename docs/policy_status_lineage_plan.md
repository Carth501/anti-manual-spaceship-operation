# Policy Status And Lineage Plan

## Goal

Add the next layer of policy curation on top of the existing policy catalog so policies can be ranked, replaced, archived, and traced over time without losing their link to source runs.

## Scope

- Add first-class policy curation fields for lifecycle state.
- Add lineage fields so one policy can explicitly replace or derive from another.
- Keep the existing run tracking and promotion workflow intact.
- Keep all changes rebuild-safe so `--rebuild-tracking` can regenerate registries from manifests.

## Out Of Scope

- Automatic best-policy selection.
- Bulk migration of legacy flat model packages.
- UI work beyond manifest and CSV registry support.

## Target Metadata

- `status`: one of `candidate`, `baseline`, `production`, `archived`
- `parent_policy_id`: direct ancestor when a policy is derived from an earlier curated policy
- `replaces_policy_id`: policy superseded by the current one
- `evaluation_notes`: optional operator-facing summary of why the policy was promoted or archived
- `status_changed_at`: timestamp of the latest lifecycle update

## Implementation Steps

### Increment 1: Manifest And Registry Schema

- Extend policy metadata in manifests and `experiments/policies.csv` with the new lifecycle and lineage fields.
- Ensure package manifests under `models/<policy>/manifest.json` store the same policy block as the run manifest.
- Make rebuild logic in `python/tracking_admin.py` preserve these fields deterministically.

### Increment 2: CLI Surface

- Extend `python/train.py` promotion flags with `--policy-status`, `--parent-policy-id`, `--replaces-policy-id`, and `--evaluation-notes`.
- Allow lifecycle updates through `--promote-run` without forcing a new `policy_id` each time.
- Keep promotion idempotent for repeated updates to the same policy.

### Increment 3: Promotion Rules

- When `replaces_policy_id` is set, do not mutate the replaced policy automatically in phase 1.
- Require operators to archive or demote the replaced policy explicitly in a later command or follow-up promotion.
- Always preserve `source_run_id` so curated policy metadata never breaks traceability back to the training run.

### Increment 4: Documentation

- Document when to create a new `policy_id` versus updating an existing one.
- Document how `parent_policy_id` differs from `replaces_policy_id`.
- Document expected usage of each lifecycle state.

## Files Likely To Change

- `python/run_tracking.py`
- `python/tracking_admin.py`
- `python/train.py`
- `README.md`

## Validation

Use mostly manifest-level and registry-level tests here. The core behavior is deterministic data movement, so most cases should be covered without launching Godot.

### Schema And Round-Trip Cases

1. Write policy metadata with all lifecycle and lineage fields populated and verify the same values round-trip through the run manifest, package manifest, and `experiments/policies.csv`.
2. Write policy metadata with all new fields omitted and verify rebuild keeps optional fields blank rather than inventing defaults.
3. Rebuild tracking from manifests that predate these fields and verify legacy policies remain readable and produce stable blank columns.

### Promotion Workflow Cases

1. Promote an existing run into a `candidate` policy and verify the new fields appear in both the package manifest and `experiments/policies.csv`.
2. Update the same `policy_id` to `baseline` and verify the policy row is replaced rather than duplicated.
3. Re-promote the same policy with revised `evaluation_notes` only and verify the notes update without changing unrelated lineage fields.
4. Promote a follow-on run with `--parent-policy-id` and verify lineage is preserved after `--rebuild-tracking`.
5. Promote a replacement policy with `--replaces-policy-id` and verify the replacement link survives rebuild.
6. Promote a replacement policy and verify the replaced policy is not silently archived or rewritten in phase 1.

### Negative And Edge Cases

1. Pass an invalid `--policy-status` value and verify the command fails before any manifest or CSV files are modified.
2. Run `--promote-run` against a missing or unreadable run manifest and verify the failure is clean and leaves existing policy artifacts untouched.
3. Promote a policy without `--parent-policy-id` or `--replaces-policy-id` and verify those fields stay empty instead of inheriting stale values from a prior promotion.
4. Rebuild tracking when one policy manifest is missing optional lifecycle fields and verify the rest of the registry is still reconstructed deterministically.

### Suggested Test Split

1. Unit-style tests for manifest normalization and CSV row generation in `python/tracking_admin.py` and `python/run_tracking.py`.
2. CLI-level tests for `python/train.py --promote-run ...` using temporary manifests and package directories.
3. One rebuild regression test that proves the same manifest set reproduces the same `policies.csv` rows across repeated runs.

## Open Questions

- Should `production` imply `is_best_in_category=true`, or should those stay independent?
- Should a later command enforce uniqueness for one `production` policy per persona or category?
