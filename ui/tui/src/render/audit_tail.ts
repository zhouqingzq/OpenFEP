import type { WsServerMessage } from "@segments/consciousness-client";

import { type TranscriptLine } from "./transcript.js";

const MAX_AUDIT_TAIL = 40;
const DELIVERY_SURFACE_REFRESH_SECONDS = 30;

export interface AuditTailOptions {
  verbose?: boolean;
}

export class AuditTail {
  private lines: TranscriptLine[] = [];
  private lastSuppressionText = "";
  private lastSuppressionAt = 0;
  private readonly verbose: boolean;

  constructor(options?: AuditTailOptions) {
    this.verbose = options?.verbose === true;
  }

  push(message: WsServerMessage): TranscriptLine | null {
    const kind = String(message.kind ?? "");
    if (kind === "RunnerSuppression") {
      const reason = String(message.payload?.reason_code ?? "unknown");
      const now = message.at ?? Math.floor(Date.now() / 1000);
      if (reason === this.lastSuppressionText && now - this.lastSuppressionAt < 8) {
        return null;
      }
      this.lastSuppressionText = reason;
      this.lastSuppressionAt = now;
      const line: TranscriptLine = {
        role: "suppression",
        text: reason,
        at: message.at,
      };
      this.append(line);
      return line;
    }
    if (kind === "RunnerHealth") {
      if (!this.verbose) {
        return null;
      }
      const ready = message.payload?.delivery_surface_ready;
      const reason = String(message.payload?.delivery_surface_reason ?? "").trim();
      const summary = reason ? `RunnerHealth: ${reason}` : `RunnerHealth: ready=${String(ready)}`;
      const line: TranscriptLine = { role: "audit", text: summary, at: message.at, meta: kind };
      this.append(line);
      return line;
    }
    if (kind === "AuditEvent") {
      if (String(message.payload?.audit_type ?? "") === "turn_progress") {
        return null;
      }
      const summary = summarizeAuditPayload(message.payload ?? {});
      const line: TranscriptLine = {
        role: "audit",
        text: summary,
        at: message.at,
        meta: kind,
      };
      this.append(line);
      return line;
    }
    return null;
  }

  list(): readonly TranscriptLine[] {
    return this.lines;
  }

  private append(line: TranscriptLine): void {
    this.lines.push(line);
    if (this.lines.length > MAX_AUDIT_TAIL) {
      this.lines.splice(0, this.lines.length - MAX_AUDIT_TAIL);
    }
  }
}

function summarizeAuditPayload(payload: Record<string, unknown>): string {
  const eventType = String(payload.event_type ?? payload.type ?? payload.kind ?? "audit");
  const reason = String(payload.reason_code ?? payload.reason ?? "").trim();
  if (reason) {
    return `${eventType}: ${reason}`;
  }
  const detail = String(payload.detail ?? payload.message ?? "").trim();
  if (detail) {
    return `${eventType}: ${detail.slice(0, 160)}`;
  }
  return eventType;
}

export { DELIVERY_SURFACE_REFRESH_SECONDS };
