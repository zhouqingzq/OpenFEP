export const DEFAULT_REPLY_EXPECTED_MS = 120_000;

export function formatProgressBar(ratio: number, width = 20): string {
  const clamped = Math.max(0, Math.min(1, ratio));
  const filled = Math.min(width, Math.round(clamped * width));
  const hasHead = filled < width && filled > 0;
  const head = hasHead ? ">" : "";
  const spaces = Math.max(0, width - filled - (hasHead ? 1 : 0));
  return `[${"=".repeat(Math.max(0, filled - (hasHead ? 1 : 0)))}${head}${" ".repeat(spaces)}]`;
}

export function formatReplyProgressLine(
  personaName: string,
  elapsedMs: number,
  expectedMs = DEFAULT_REPLY_EXPECTED_MS,
): string {
  const safeExpected = Math.max(1, expectedMs);
  const percent = Math.min(99, Math.round((elapsedMs / safeExpected) * 100));
  const bar = formatProgressBar(elapsedMs / safeExpected);
  return `${personaName}正在回复... ${bar} ${percent}%`;
}
