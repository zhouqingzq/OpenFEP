import * as readline from "node:readline";
import type { Interface } from "node:readline";

export interface ReplOutput {
  log(message: string): void;
  error(message: string): void;
}

type ReadlineWithLine = Interface & { line?: string };

export function createReplOutput(custom?: Pick<Console, "log" | "error">): ReplOutput {
  if (custom) {
    return {
      log: (message) => custom.log(message),
      error: (message) => custom.error(message),
    };
  }
  return {
    log: (message) => process.stderr.write(`${message}\n`),
    error: (message) => process.stderr.write(`${message}\n`),
  };
}

export function restoreReplPrompt(rl: Interface | null, prompt = "> "): void {
  if (!rl) {
    return;
  }
  const typed = String((rl as ReadlineWithLine).line ?? "");
  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(`${prompt}${typed}`);
}

export function emitReplMessage(
  rl: Interface | null,
  write: () => void,
  options?: { prompt?: string },
): void {
  if (!rl) {
    write();
    return;
  }
  const prompt = options?.prompt ?? "> ";
  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  write();
  restoreReplPrompt(rl, prompt);
}

export function emitReplInlineStatus(
  rl: Interface | null,
  line: string,
  _options?: { prompt?: string },
): void {
  if (!rl) {
    process.stderr.write(`${line}\n`);
    return;
  }
  // Keep spinner/status on stderr only; touching stdout clears the readline prompt and flickers.
  process.stderr.write(`\r\x1b[2K${line}`);
}

export function clearReplInlineStatus(rl: Interface | null, _options?: { prompt?: string }): void {
  if (!rl) {
    return;
  }
  process.stderr.write("\r\x1b[2K");
}

export async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

export function waitForAbortSignal(signal: AbortSignal, timeoutMs: number, label: string): Promise<void> {
  if (signal.aborted) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}
