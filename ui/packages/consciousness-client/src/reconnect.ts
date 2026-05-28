/** Exponential backoff for WebSocket reconnect (cap 30s by default). */

export interface BackoffOptions {
  initialMs?: number;
  maxMs?: number;
  multiplier?: number;
}

export class ReconnectBackoff {
  private readonly initialMs: number;
  private readonly maxMs: number;
  private readonly multiplier: number;
  private attempt = 0;

  constructor(options: BackoffOptions = {}) {
    this.initialMs = options.initialMs ?? 500;
    this.maxMs = options.maxMs ?? 30_000;
    this.multiplier = options.multiplier ?? 2;
  }

  reset(): void {
    this.attempt = 0;
  }

  nextDelayMs(): number {
    const raw = this.initialMs * this.multiplier ** this.attempt;
    this.attempt += 1;
    return Math.min(this.maxMs, Math.round(raw));
  }

  get attempts(): number {
    return this.attempt;
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
