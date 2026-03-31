# M4.6 Work Prompt

Implement a longitudinal stability milestone for cognitive-style differentiation.

Requirements:
- Run multiple seeds across multiple style profiles.
- Measure within-profile restart drift, between-profile divergence, and recovery-retention after corruption.
- Quantify acceptance with explicit thresholds:
  - style stability over at least 500 decision cycles using ICC, with above 0.7 counted as stable,
  - style reproducibility across 5 seed-separated runs using clustering purity, with at least 85 percent purity counted as strong evidence,
  - non-trivial style differentiation using a between-profile versus within-profile effect size target, with Cohen's d above 0.8 counted as strong evidence,
  - perturbation recovery using post-recovery cosine similarity, with above 0.7 counted as good recovery.
- Produce divergence, recovery, stress, and acceptance artifacts.
- Regress M4.5, M4.4, M4.3, and continuity/restart acceptance evidence.
