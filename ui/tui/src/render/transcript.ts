export type TranscriptRole = "user" | "assistant" | "proactive" | "audit" | "suppression";

export interface TranscriptLine {
  role: TranscriptRole;
  text: string;
  at?: number;
  meta?: string;
}

export interface TranscriptLabels {
  user?: string;
  assistant?: string;
  proactive?: string;
  audit?: string;
  suppression?: string;
}

const DEFAULT_ROLE_LABELS: Record<TranscriptRole, string> = {
  user: "user",
  assistant: "assistant",
  proactive: "proactive",
  audit: "audit",
  suppression: "suppression",
};

const ANSI: Record<TranscriptRole, string> = {
  user: "\x1b[36m",
  assistant: "\x1b[32m",
  proactive: "\x1b[35m",
  audit: "\x1b[90m",
  suppression: "\x1b[33m",
};

const RESET = "\x1b[0m";

export function roleLabel(role: TranscriptRole, labels?: TranscriptLabels): string {
  const custom = labels?.[role];
  if (custom && custom.trim()) {
    return custom.trim();
  }
  return DEFAULT_ROLE_LABELS[role];
}

export function ansiColorForRole(role: TranscriptRole): string {
  return ANSI[role];
}

export function formatTranscriptLine(
  line: TranscriptLine,
  options?: { color?: boolean; labels?: TranscriptLabels },
): string {
  const label = roleLabel(line.role, options?.labels).padEnd(11, " ");
  const meta = line.meta ? ` (${line.meta})` : "";
  const body = `${label}| ${line.text}${meta}`;
  if (options?.color === false) {
    return body;
  }
  return `${ansiColorForRole(line.role)}${body}${RESET}`;
}

export function formatTranscriptBlock(
  lines: TranscriptLine[],
  options?: { color?: boolean; labels?: TranscriptLabels },
): string {
  return lines.map((line) => formatTranscriptLine(line, options)).join("\n");
}

export function transcriptRoleFromEvent(event: string): TranscriptRole {
  if (event === "proactive_turn") {
    return "proactive";
  }
  if (event === "assistant_message") {
    return "assistant";
  }
  return "user";
}
