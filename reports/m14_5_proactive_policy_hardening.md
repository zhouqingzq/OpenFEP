# M14.5 Proactive Policy Hardening

## Defaults

Production-facing initiative defaults are conservative:

- `implicit_idle_delivery = false`
- `proactive_policy_profile = "bounded_default"`
- `max_proactive_per_session = 1`
- `cooldown_turns = 2`

The demo profile `streamlit_open_chat` is available only through
`SEGMENTUM_PROACTIVE_PROFILE=streamlit_open_chat` or the explicit runtime setter
`set_proactive_policy_profile("streamlit_open_chat")`.

## Local MVP UI

The Streamlit page intentionally hides the four proactive opt-in checkboxes per
current product request. When a persona is loaded and the MVP loop is active, the
local page automatically enables bounded proactive delivery, implicit idle
delivery, idle introspection, and background continuity. This is a UI bootstrap
choice, not a change to the underlying fresh-state defaults.

If `SEGMENTUM_PROACTIVE_PROFILE` is unset, the hidden UI bootstrap keeps
`bounded_default`; session caps and cooldowns remain active. Setting
`SEGMENTUM_PROACTIVE_PROFILE=streamlit_open_chat` enables the demo cap/cooldown
relaxation and the UI diagnostics show the active profile.

## Recommended Production Values

- `max_proactive_per_session = 1`
- `cooldown_turns = 2`
- `idle_threshold_seconds >= 120`
- implicit idle attempt throttle `>= 45s`
- keep M14.3 traceable target selection and delivery assessor enabled

## Validation

M14.5 regression coverage is included in
`tests/test_m14_4_streamlit_implicit_idle.py`:

- fresh initiative state remains conservative,
- `streamlit_open_chat` only appears through env/setter,
- bounded-default still enforces session cap and cooldown,
- M14.4 implicit idle wiring still delivers when the profile and target allow it.

Expanded targeted regression:

```text
pytest tests/test_m13_3_ui_initiative.py tests/test_m13_4_idle_tick.py \
  tests/test_m14_0_conscious_idle_reflector.py \
  tests/test_m14_1_background_self_continuity.py \
  tests/test_m14_2_self_loop_daemon.py tests/test_m14_2_scheduled_outreach.py \
  tests/test_m14_3_proactive_alignment.py \
  tests/test_m14_4_streamlit_implicit_idle.py tests/test_m13_6_memory_efe.py
143 passed
```

## Go / No-Go

Entering the next milestone is reasonable after targeted tests pass, with two
known product caveats:

- Full `python -m pytest` is expected to exceed ten minutes in this repo; use
  targeted suites plus `--collect-only` for broad smoke.
- The Streamlit auto-enable behavior is intentionally local-product behavior and
  should not be treated as production consent semantics.
