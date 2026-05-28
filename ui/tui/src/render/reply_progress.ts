export function formatProgressBar(ratio: number, width = 20): string {
  const clamped = Math.max(0, Math.min(1, ratio));
  const filled = Math.min(width, Math.round(clamped * width));
  const hasHead = filled < width && filled > 0;
  const head = hasHead ? ">" : "";
  const spaces = Math.max(0, width - filled - (hasHead ? 1 : 0));
  return `[${"=".repeat(Math.max(0, filled - (hasHead ? 1 : 0)))}${head}${" ".repeat(spaces)}]`;
}

export function formatReplyProgressLine(personaName: string, percent: number): string {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  const bar = formatProgressBar(clamped / 100);
  return `${personaName}正在回复... ${bar} ${clamped}%`;
}
