# M14.4 Streamlit Implicit Idle Proactive Delivery

## Implementation

M14.4 adds an open-page Streamlit delivery adapter without adding a second
generation stack.

- `segmentum.dialogue.runtime.m14_4_implicit_idle` computes idle seconds from
  `temporal_state.last_user_turn_at` with `last_turn_at` fallback.
- Streamlit now drains queued outreach first, runs idle introspection, then calls
  `maybe_propose_proactive_turn(implicit_idle_request=True)` and
  `run_proactive_turn` through the existing M13.3 path.
- The helper records `M14ImplicitIdleProactiveCheckEvent` audit rows and compact
  UI diagnostics for last check, suppression reason, profile, and proposal id.
- Manual "让胡桃继续" delivery was removed; hidden local MVP bootstrap enables
  proactive, implicit idle, idle introspection, and background continuity.

## Acceptance Notes

Silence alone is still insufficient. A visible proactive message requires a
traceable target accepted by M14.3: scheduled outreach, memory-EFE outreach,
correction follow-up, or evidence-backed boredom exploration. Vague
`open_items[].next_check = "later"` remains diagnostic-only.

`streamlit_open_chat` is available for demos and skips session/cooldown caps
only. Safety, delivery assessor, typing/pending-turn guards, idle threshold, and
traceable-target gates remain active.

## Validation

Targeted acceptance:

```text
pytest tests/test_m14_4_streamlit_implicit_idle.py \
  tests/test_m13_3_ui_initiative.py \
  tests/test_m14_3_proactive_alignment.py
42 passed
```

Focused checks cover policy profile relaxation, bounded-default regressions,
idle computation, throttle, user-active suppression, event shape, persistence,
and a negative case proving repeated `later` keyword cues do not create a target.

## Audit Findings

- No new keyword cue path was added for proactive target selection.
- No synthetic open-item shortcut was added; vague open items remain blocked by
  M14.3 unless the legacy compatibility flag is explicitly enabled.
- The first implementation order would have throttled before idle introspection
  could create a target. This was fixed by ordering queued outreach, then idle
  introspection, then implicit idle delivery.
- Regression audit found that the M14.3 traceable-target tightening had
  accidentally suppressed M14.0 reflection focus for non-outreach open items.
  The fallback planner now may reflect on retrieved open items even when they
  are not eligible for proactive outreach.
- Regression audit also found brittle substring grounding: `An` could match
  `and`, and generic `follow/up` tokens made synthetic fixtures look grounded.
  Generic tokens are now filtered, scheduled outreach is allowed to rely on its
  durable intent plus delivery assessor, and explicit assessor rejection takes
  precedence over ungrounded-reply diagnostics.
