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
const DEFAULT_LABEL_DISPLAY_WIDTH = 11;

/** Approximate terminal display width (CJK counts as 2). */
export function displayWidth(text: string): number {
  let width = 0;
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0;
    if (
      (code >= 0x1100 && code <= 0x115f) ||
      (code >= 0x2e80 && code <= 0xa4cf) ||
      (code >= 0xac00 && code <= 0xd7a3) ||
      (code >= 0xf900 && code <= 0xfaff) ||
      (code >= 0xfe10 && code <= 0xfe19) ||
      (code >= 0xfe30 && code <= 0xfe6f) ||
      (code >= 0xff00 && code <= 0xff60) ||
      (code >= 0xffe0 && code <= 0xffe6) ||
      (code >= 0x20000 && code <= 0x2fffd) ||
      (code >= 0x30000 && code <= 0x3fffd)
    ) {
      width += 2;
    } else if (code >= 0x20 && code !== 0x7f) {
      width += 1;
    }
  }
  return width;
}

export function resolveTranscriptLabelWidth(labels?: TranscriptLabels): number {
  const widths = Object.keys(DEFAULT_ROLE_LABELS).map((role) =>
    displayWidth(roleLabel(role as TranscriptRole, labels)),
  );
  return Math.max(DEFAULT_LABEL_DISPLAY_WIDTH, ...widths);
}

function padLabel(label: string, width: number): string {
  const padding = width - displayWidth(label);
  return padding > 0 ? `${label}${" ".repeat(padding)}` : label;
}

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
  options?: { color?: boolean; labels?: TranscriptLabels; labelWidth?: number },
): string {
  const labelWidth = options?.labelWidth ?? resolveTranscriptLabelWidth(options?.labels);
  const label = padLabel(roleLabel(line.role, options?.labels), labelWidth);
  const meta = line.meta ? ` (${line.meta})` : "";
  const body = `${label}| ${line.text}${meta}`;
  if (options?.color === false) {
    return body;
  }
  return `${ansiColorForRole(line.role)}${body}${RESET}`;
}

export function formatTranscriptBlock(
  lines: TranscriptLine[],
  options?: { color?: boolean; labels?: TranscriptLabels; labelWidth?: number },
): string {
  const labelWidth = options?.labelWidth ?? resolveTranscriptLabelWidth(options?.labels);
  return lines.map((line) => formatTranscriptLine(line, { ...options, labelWidth })).join("\n");
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
