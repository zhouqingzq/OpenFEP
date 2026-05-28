export type ConnectionBadge = "connecting" | "live" | "resyncing" | "offline";

export interface ConnectionStateInput {
  wsOpen: boolean;
  subscribed: boolean;
  reconnecting: boolean;
  explicitOffline?: boolean;
}

export function connectionBadge(state: ConnectionStateInput): ConnectionBadge {
  if (state.explicitOffline) {
    return "offline";
  }
  if (state.reconnecting) {
    return "resyncing";
  }
  if (state.wsOpen && state.subscribed) {
    return "live";
  }
  if (state.wsOpen || state.reconnecting) {
    return "connecting";
  }
  return "offline";
}

export function badgeLabel(badge: ConnectionBadge): string {
  switch (badge) {
    case "connecting":
      return "连接中";
    case "live":
      return "在线";
    case "resyncing":
      return "重连同步";
    case "offline":
      return "离线";
    default:
      return badge;
  }
}
