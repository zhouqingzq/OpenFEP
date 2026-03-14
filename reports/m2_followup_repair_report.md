# M2 Follow-up Repair Report

## 1. Repair-Pre Audit

Evidence
- episode write path existed but was gated mostly by surprise plus exact duplicate rejection
- identity narrative already influenced policy scoring but lacked structured evidence binding
- counterfactual learning already had confidence gating, but no cooling buffer for medium-confidence adoption
- evaluation harness existed, but MUR/CAQ/VCUS were still too close to implementation-internal success signals

Missing
- native narrative claim provenance
- native contradiction checker for trait/value/capability claims
- retrieval benefit separated from retrieval influence
- mixed-origin attribution with secondary cause and causal chain

Harness-only
- narrative consistency could be described in report text but not audited as structured machine-readable evidence
- EAA, CAQ, and VCUS were largely single-family harness metrics with limited perturbation pressure

## 2. P0/P1/P2 Repair Status

- `P0.1 episode gating`: DONE
- `P0.2 contradiction checker`: DONE
- `P0.3 narrative provenance`: DONE
- `P1.1 mixed attribution`: DONE
- `P1.2 MUR split influence/benefit`: DONE
- `P1.3 perturbed CAQ`: DONE
- `P1.4 stressed VCUS`: DONE
- `P2.1 episode lifecycle`: DONE
- `P2.2 self-model calibration`: DONE
- `P2.3 counterfactual cooling`: DONE

## 3. Data Structure And Log Changes

- `segmentum/memory.py`: added joint episode gating metadata, review-family tags, lifecycle transition history, merge/support accumulation, archival/compression audit events, and identity-critical retention flags.
- `segmentum/self_model.py`: added structured `NarrativeClaim`, narrative provenance, contradiction summaries, and self-model calibration fields.
- `segmentum/world_model.py`: added a counterfactual candidate buffer so medium-confidence counterfactual updates can be cooled before policy absorption.
- `reports/m2_evidence.jsonl`: now receives episode-gating, narrative-audit, mixed-attribution, memory-utility, perturbed-counterfactual, and stressed-value evidence records.

## 4. New Evaluation Scenarios

- Mixed fault attribution with primary origin, secondary origin, causal chain, and confidence.
- Trivial-vs-critical episode write-path probes plus near-duplicate merge checks across hazard, resource, and social families.
- Retrieval influence separated from retrieval benefit, with wider family-aware evidence capture.
- Counterfactual adoption tested under perturbation and cooling constraints.
- Stronger value-conflict scenarios that count value-order flips instead of only safe outcomes.

## 5. Metric Definition Changes

- `EAA`: no longer a single-label classification score; now mixes primary origin, secondary origin, causal chain quality, and diagnosis confidence.
- `MUR`: now distinguishes `retrieval_influence_rate` from `retrieval_benefit_rate` and reports both.
- `CAQ`: now measures post-adoption benefit under perturbed observations instead of only same-family replay success.
- `VCUS`: now tracks explicit value-flip rate under harder conflict scenarios.

## 6. Repair Results

- `ICI`: 1.0000
- `EAA`: 0.9725
- `MUR`: 1.0000
- `PSSR`: 0.9997
- `CAQ`: 1.0000
- `VCUS`: 1.0000
- `retrieval_influence_rate`: 1.0000
- `retrieval_benefit_rate`: 1.0000
- `caq_generalization_rate`: 1.0000
- `vcus_value_flip_rate`: 0.0000

## 7. Residual Risks

- Evidence binding and contradiction checking are materially stronger; lifecycle transitions are now structured, but retrieval utility still deserves broader runtime families beyond the current follow-up probes.
- `CAQ` now measures post-graduation benefit under perturbation. A higher score here means the cooled candidate review produced auditable downstream benefit, not just a logged adoption.
- `MUR` stayed at `1.0`; that should be read carefully because the current probe family is still narrow even after benefit splitting.

## 8. M3 Recommendation

- Final recommendation: RECOMMEND_M3_WITH_CAUTION
- Rationale: P0 evidence-loop repairs are in place and stricter metrics remain above threshold.
- Suggested next minimal repair: keep widening replay/retrieval probes so M2.2 family diversity is demonstrated in native runtime traffic, not only follow-up evaluation scenarios.
