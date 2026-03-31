# M4.3 Work Prompt

Implement a deterministic single-task fitting milestone over the repository Confidence Database slice.

Requirements:
- Fit trial-level choice and confidence with a parameterized cognitive agent.
- Compare against a three-tier baseline set.
- Lower baselines that must be beaten:
  - standard Signal Detection Theory with fixed confidence mapping,
  - random action with task-agnostic confidence.
- Competitive baselines that should be matched or exceeded:
  - Fleming and Daw style metacognitive-noise modeling,
  - the best-performing model reported by the Confidence Database source study when reproducible in-repo,
  - a standard drift-diffusion model with post-decisional evidence accumulation.
- Upper baseline:
  - an independently fit per-subject or per-condition optimum-parameter model used as a free-parameter ceiling.
- Produce held-out metrics, ablation evidence, stress evidence, failure analysis, and a machine-readable acceptance report.
- Enforce these acceptance rules:
  - all lower baselines must be beaten or M4.3 fails,
  - at least one competitive baseline must be matched or exceeded on the primary metric, with Brier score difference under 5 percent counting as parity,
  - if every competitive baseline is worse by more than 15 percent, M4.4 should be blocked pending architecture review.
- Regress M4.2 and M4.1 acceptance.
