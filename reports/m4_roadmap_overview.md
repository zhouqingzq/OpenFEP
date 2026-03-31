# M4 Roadmap Overview

## Overall Goal

M4 turns cognitive style from a descriptive personality layer into a measurable behavioral interface that can be:

- serialized,
- injected into action selection,
- evaluated on benchmark tasks,
- stress-tested across contexts,
- and validated for longitudinal stability.

The phase is intentionally structured as a staged validation ladder rather than a jump from toy tasks straight into open-world tooling.

## Milestone Sequence

### M4.1: Cognitive Variable Operationalization and Unified Parameter Interface

Focus:

- define the core parameter family,
- make it serializable and decision-relevant,
- and expose it through decision logging.

The parameter family should explicitly include two source-trust dimensions:

- `source_precision_gain`, which controls how strongly indirect or linguistic prediction-error signals compete with direct perceptual error,
- `source_authority_weighting`, which controls how much source reliability history differentiates one indirect source from another.

### M4.2: Cognitive Benchmark Environment Setup

Focus:

- connect the parameter family to benchmark protocols,
- standardize observation-action-confidence interfaces,
- and ensure reproducible closed-loop evaluation.

### M4.3: Single-Task Behavioral Fit and Initial Falsification

Focus:

- fit the repository benchmark slice at trial level,
- compare the cognitive-style agent against a tiered baseline set,
- and fail early if the architecture only beats trivial baselines.

Baseline policy:

- lower baselines must be beaten,
- at least one competitive baseline should be matched or exceeded,
- and a per-subject or per-condition best-fit model is treated as an upper ceiling reference.

### M4.4: Cross-Task Stability Check

Focus:

- test whether shared parameters remain credible across multiple tasks,
- and separate stable style terms from task-local fit terms.

### M4.5: Controlled Complex Environment Validation

Focus:

- bridge from benchmark tasks into a richer but still controlled environment,
- preserve quantifiability,
- and test cross-context behavioral transfer without open-world tooling noise.

Recommended environment family:

- MiniGrid or an equivalent partially observable grid world with resource gathering, threat avoidance, and information search.

This stage should also test:

- graceful degradation under compute or attention limits,
- and cue-following behavior under partially reliable hints to expose source-trust effects.

### M4.6: Longitudinal Stability and Style Differentiation

Focus:

- quantify whether style is stable over time,
- reproducible across seeds,
- distinct across parameter profiles,
- and partially recoverable after major perturbation.

Recommended quantitative criteria:

- ICC for within-profile stability,
- clustering purity for reproducibility,
- effect size for between-profile separation,
- cosine similarity for post-perturbation recovery.

## Boundary to M5

Open-world tool use, filesystem operations, network retrieval, and task-queue orchestration belong to M5.

M4 should end once cognitive-style effects are credible across benchmark and controlled complex environments with quantitative stability evidence.
