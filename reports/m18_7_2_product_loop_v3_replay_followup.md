# M18.7 Product-Loop v3 Replay Follow-up

**Date**: 2026-06-12
**Input**: existing `tmp_m18_7_2_v3_prompt_5run/run_1..5` logs
**Scope**: metric repair plus conservative M18.7 same-turn override
**Explicit non-changes**: M18.7 prompt and M20.4 P0-6

## Result

The old Sub-2 and Sub-3 failures were partly measurement failures:

- Sub-2 compared the predicted addressee pid with the speaker pid. The repaired
  metric reads the structured ingress speaker identity from each Path B turn
  log and compares it with the fixture speaker.
- Sub-3 read a stale final diagnostic snapshot. The repaired metric counts
  actual `AddresseeTargetMatchAdmitted` events from all logged turn bus events.

| run | Sub-2 exact speaker identity | Sub-3 actual admits | Sub-3 dir=True | Sub-3 verdict |
|---|---:|---:|---:|---|
| 1 | 12/12 | 44 | 8 | acceptable |
| 2 | 12/12 | 25 | 10 | acceptable |
| 3 | 12/12 | 47 | 9 | acceptable |
| 4 | 12/12 | 26 | 0 | p04_dir_true_admit_zero |
| 5 | 12/12 | 49 | 16 | acceptable |

Sub-2 is acceptable in **5/5** runs. Sub-3 is acceptable in **4/5** runs from
the existing bus events; run 4 remains an honest failure.

## Same-Turn Override Replay

The conservative M20.4.1 gate reads the existing M18.7.2 result and the
derived `group_turn_binding`. It can override only `clarify_addressee` or
`no_reply`, requires `addressed_to_assistant=True`, a non-empty participant
id, confidence strictly above 0.7, high ambiguity, no explicit addressed
participant, and no reply-to target.

| run | existing v3 turn changed by replay | explicit-other cases blocked |
|---|---|---:|
| 1 | turn 0, clarify, confidence 0.75 | 3/3 |
| 2 | turn 0, clarify, confidence 0.80 | 1/1 |
| 3 | turn 0, clarify, confidence 0.90 | 3/3 |
| 4 | none; participant id disclosure guard blocks override | 2/2 |
| 5 | turn 0, clarify, confidence 0.80 | 3/3 |

Across the existing five runs, **4/4 eligible correct True judgments change
the wrong clarify decision**, while **12/12 explicitly addressed-to-other
turns remain blocked**.

## Product Interpretation

The prompt is no longer the immediate product bottleneck for the narrow
same-turn scenario. The important missing piece was wiring the M18.7.2 result
into the live gate and measuring the existing producer bus events correctly.
P0-6 remains unchanged and continues to own the separate cross-turn producer
policy.
