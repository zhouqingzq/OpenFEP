import * as readline from "node:readline";
import type { Interface } from "node:readline";

export const REPL_PROMPT = "> ";

export interface StdinLineSource {
  readline: Interface;
  prompt: string;
  lines(): AsyncIterable<string>;
}

export function createStdinLineSource(prompt = REPL_PROMPT): StdinLineSource {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
  });
  return {
    readline: rl,
    prompt,
    lines: async function* lines() {
      try {
        while (true) {
          const line = await new Promise<string>((resolve) => {
            rl.question(prompt, resolve);
          });
          yield line;
        }
      } finally {
        rl.close();
      }
    },
  };
}
