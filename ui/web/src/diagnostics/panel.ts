import type { SnapshotResponse, WsServerMessage } from "@segments/consciousness-client";

const MAX_AUDIT_ROWS = 40;

export interface AuditRow {
  at: number;
  kind: string;
  summary: string;
}

export class DiagnosticsPanel {
  private readonly auditList: HTMLElement;
  private readonly hintsEl: HTMLElement;
  private readonly snapshotPre: HTMLElement;
  private audits: AuditRow[] = [];

  constructor(root: HTMLElement) {
    root.innerHTML = `
      <section class="diagnostics-panel">
        <h2>Diagnostics</h2>
        <div class="runtime-hints"></div>
        <h3>Audit tail</h3>
        <ul class="audit-list"></ul>
        <h3>Snapshot JSON</h3>
        <pre class="snapshot-json"></pre>
      </section>
    `;
    this.hintsEl = root.querySelector(".runtime-hints") as HTMLElement;
    this.auditList = root.querySelector(".audit-list") as HTMLElement;
    this.snapshotPre = root.querySelector(".snapshot-json") as HTMLElement;
  }

  pushAudit(message: WsServerMessage): void {
    if (message.kind !== "AuditEvent" && message.kind !== "RunnerHealth") {
      return;
    }
    const summary = JSON.stringify(message.payload ?? {}).slice(0, 240);
    this.audits.push({ at: message.at, kind: message.kind, summary });
    if (this.audits.length > MAX_AUDIT_ROWS) {
      this.audits = this.audits.slice(-MAX_AUDIT_ROWS);
    }
    this.renderAudits();
  }

  updateSnapshot(snapshot: SnapshotResponse): void {
    const hints = snapshot.runtime_hints ?? {};
    const lines = [
      `last_turn_index: ${String(hints.last_turn_index ?? "-")}`,
      `initiative_enabled: ${String(hints.initiative_enabled ?? "-")}`,
      `runner_kind: ${String(hints.runner_kind ?? "-")}`,
    ];
    this.hintsEl.textContent = lines.join("\n");
    this.snapshotPre.textContent = JSON.stringify(snapshot, null, 2);
  }

  snapshotText(): string {
    return this.snapshotPre.textContent ?? "";
  }

  private renderAudits(): void {
    this.auditList.innerHTML = "";
    for (const row of this.audits.slice().reverse()) {
      const item = document.createElement("li");
      item.textContent = `[${row.at}] ${row.kind}: ${row.summary}`;
      this.auditList.appendChild(item);
    }
  }
}
