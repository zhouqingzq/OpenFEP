export interface StatusBarInput {
  personaId: string;
  sessionId: string;
  gatewayUrl: string;
  wsOpen: boolean;
  subscribed: boolean;
  reconnecting: boolean;
  runnerPhase?: string;
}

export function formatStatusBar(input: StatusBarInput): string {
  const conn = connectionLabel(input.wsOpen, input.subscribed, input.reconnecting);
  const runner = input.runnerPhase ? ` runner=${input.runnerPhase}` : "";
  return `[${input.personaId}/${input.sessionId}] gateway=${input.gatewayUrl} conn=${conn}${runner}`;
}

export function connectionLabel(wsOpen: boolean, subscribed: boolean, reconnecting: boolean): string {
  if (reconnecting) {
    return "reconnecting";
  }
  if (wsOpen && subscribed) {
    return "live";
  }
  if (wsOpen) {
    return "connecting";
  }
  return "offline";
}
