import type { WsServerMessage } from "@segments/consciousness-client";

import { type TranscriptLine, formatTranscriptLine } from "./transcript.js";

const MAX_AUDIT_TAIL = 40;

export class AuditTail {
  private lines: TranscriptLine[] = [];

  push(message: WsServerMessage): TranscriptLine | null {
    const kind = String(message.kind ?? "");
    if (kind === "RunnerSuppression") {
      const reason = String(message.payload?.reason_code ?? "unknown");
      const line: TranscriptLine = {
        role: "suppression",
        text: reason,
        at: message.at,
        meta: kind,
      };
      this.append(line);
      return line;
    }
    if (kind === "AuditEvent" || kind === "RunnerHealth") {
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

  recentText(limit = 12): string {
    return this.lines
      .slice(-limit)
      .map((line) => formatTranscriptLine(line, { color: false }))
      .join("\n");
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
