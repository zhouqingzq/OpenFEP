export const THINKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const;

export function thinkingFrame(index: number): string {
  const frames = THINKING_FRAMES;
  const safe = Number.isFinite(index) ? Math.abs(Math.trunc(index)) : 0;
  return frames[safe % frames.length] ?? frames[0];
}

export function formatThinkingLine(frameIndex: number, label = "thinking"): string {
  return `${label} ${thinkingFrame(frameIndex)}  (assistant is replying...)`;
}

export interface ThinkingIndicatorHooks {
  render(line: string): void;
  clear(): void;
}

export interface ThinkingIndicatorOptions {
  label?: string;
  formatLine?: (elapsedMs: number, frameIndex: number) => string;
}

export class ThinkingIndicator {
  private timer: ReturnType<typeof setInterval> | null = null;
  private frameIndex = 0;
  private active = false;
  private timeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private startedAt = 0;
  private formatLine: (elapsedMs: number, frameIndex: number) => string = (elapsedMs, frameIndex) =>
    formatThinkingLine(frameIndex, "thinking");

  constructor(
    private readonly hooks: ThinkingIndicatorHooks,
    private readonly intervalMs = 100,
    private readonly timeoutMs = 300_000,
    private readonly onTimeout?: () => void,
  ) {}

  start(options?: ThinkingIndicatorOptions | string): void {
    const resolved =
      typeof options === "string"
        ? { label: options }
        : options ?? {};
    this.stop(false);
    this.active = true;
    this.frameIndex = 0;
    this.startedAt = Date.now();
    this.formatLine =
      resolved.formatLine ??
      ((elapsedMs, frameIndex) => formatThinkingLine(frameIndex, resolved.label ?? "thinking"));
    this.hooks.render(this.formatLine(0, 0));
    if (this.intervalMs > 0) {
      this.timer = setInterval(() => {
        if (!this.active) {
          return;
        }
        this.frameIndex += 1;
        this.hooks.render(this.formatLine(Date.now() - this.startedAt, this.frameIndex));
      }, this.intervalMs);
    }
    if (this.timeoutMs > 0) {
      this.timeoutTimer = setTimeout(() => {
        if (!this.active) {
          return;
        }
        this.stop(true);
        this.onTimeout?.();
      }, this.timeoutMs);
    }
  }

  stop(clearLine = true): void {
    this.active = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.timeoutTimer) {
      clearTimeout(this.timeoutTimer);
      this.timeoutTimer = null;
    }
    if (clearLine) {
      this.hooks.clear();
    }
  }

  isActive(): boolean {
    return this.active;
  }

  touch(): void {
    if (!this.active) {
      return;
    }
    this.hooks.render(this.formatLine(Date.now() - this.startedAt, this.frameIndex));
  }
}
