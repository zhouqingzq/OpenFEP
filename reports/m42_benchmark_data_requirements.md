# M4.2 Benchmark Data Requirements

## Current Repository State

The repository contains runnable benchmark scaffolding for:

- `confidence_database`
- `iowa_gambling_task`

What is complete:

- Registry discovery and manifest validation
- Adapter registration and basic benchmark execution
- Smoke-fixture preprocessing and deterministic test coverage
- Acceptance reporting with explicit blocker/status output

What is not complete inside the repository:

- Acceptance-grade external bundles for either benchmark
- Enough real subject coverage to make formal M4.2 benchmark claims
- A passing M4.2 acceptance state

## Smoke Fixture vs External Bundle

The files under `data/benchmarks/<benchmark_id>/` are repository fixtures for smoke testing.

They are intentionally limited:

- Small
- Repo-local
- Suitable for adapter/registry/report validation
- Not acceptable as formal benchmark evidence

Acceptance-grade evaluation requires a real external bundle. In the current repo state, both bundled manifests are marked so they cannot be mistaken for acceptance-ready data.

## Where External Bundles Should Live

Use one of these two registry layouts:

1. Import into the default registry path:

```text
data/benchmarks/<benchmark_id>/
```

2. Point `SEGMENTUM_BENCHMARK_ROOT` at an external registry root containing:

```text
<external-root>/<benchmark_id>/manifest.json
<external-root>/<benchmark_id>/<data_file from manifest>
```

Recommended import command:

```bash
python -m segmentum.benchmark_cli import <bundle_dir_or_zip> --root <external_registry_root>
```

For the local raw Confidence Database directory already placed at the repo root, a conversion helper now exists:

```bash
py -3.11 scripts/build_confidence_external_bundle.py
```

By default this writes an ignored local registry under:

```text
external_benchmark_registry/confidence_database/
```

That generated bundle is suitable for `validate` and for confidence-side benchmark runs without polluting git history.

Recommended validation command:

```bash
python -m segmentum.benchmark_cli validate confidence_database --root <external_registry_root>
python -m segmentum.benchmark_cli validate iowa_gambling_task --root <external_registry_root>
```

## Data Expectations

`confidence_database` external bundle must provide:

- Real external source material
- Non-smoke manifest (`smoke_test_only: false`)
- `source_type: external_bundle`
- Valid JSONL records matching the confidence schema
- Enough independent subjects/sessions for held-out evaluation beyond the repo fixture

`iowa_gambling_task` external bundle must provide:

- Real external source material
- Non-smoke manifest (`smoke_test_only: false`)
- `source_type: external_bundle`
- Valid JSONL records matching the IGT schema
- Enough independent subjects for held-out evaluation beyond the repo fixture

## When M4.2 Can Be Considered Truly Passed

M4.2 is only truly passed when all of the following are true:

- Both benchmarks resolve to external bundles, not repo smoke fixtures
- Both manifests validate cleanly
- Reports mark both benchmarks as `acceptance_ready` before the acceptance run
- The acceptance report finishes with `acceptance_state: acceptance_pass`
- No blocker findings remain for missing external bundles or inadequate evidence

## Current Blockers

Current blocker for `confidence_database`:

- No longer blocked on missing external bundle if you use the generated local bundle under `external_benchmark_registry/confidence_database/`

Current blocker for `iowa_gambling_task`:

- No longer blocked on missing external bundle if you use the generated local bundle under `external_benchmark_registry/iowa_gambling_task/`

After both local external bundles are generated, the next practical blocker is full-report runtime/performance on the large confidence bundle, not bundle availability.
