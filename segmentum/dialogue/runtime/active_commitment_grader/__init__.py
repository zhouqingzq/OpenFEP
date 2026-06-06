"""M20.2 routing stubs (one module per correction level).

Each module exposes a `route(decision, *, owner_state_snapshot, at)`
function that emits the `GradedCorrectionRouted` audit event and
(optionally) calls into the existing owner write path.

M20.2 ships no-op routing stubs. The actual owner write paths
(M19.1 traction, M19.3 promotion, M9.0 control_guidance, M17.4
precision EMA, M15.1 episode aggregation) are owned by their
respective modules. A future M20.2.1 milestone wires the real
write paths into the routing stubs.

M20.2 owns the dispatcher; M20.2.1 owns the actual write path
migration. M20.2.1 may replace the body of each `route` function
without changing its signature.
"""

from .expire import route_expire
from .microadjust import route_microadjust
from .next_turn import route_next_turn
from .revoke import route_revoke
from .same_turn import route_same_turn
from .slow_promote import route_slow_promote


__all__ = [
    "route_expire",
    "route_microadjust",
    "route_next_turn",
    "route_revoke",
    "route_same_turn",
    "route_slow_promote",
]
