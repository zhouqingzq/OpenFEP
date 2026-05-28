import { describe, expect, it } from "vitest";

import { formatThinkingLine, thinkingFrame, THINKING_FRAMES, ThinkingIndicator } from "../src/render/thinking_indicator.js";

describe("thinking indicator", () => {
  it("cycles spinner frames", () => {
    expect(THINKING_FRAMES).toHaveLength(10);
    expect(thinkingFrame(0)).toBe("⠋");
    expect(thinkingFrame(1)).toBe("⠙");
    expect(thinkingFrame(10)).toBe("⠋");
  });

  it("formats thinking line", () => {
    expect(formatThinkingLine(2)).toBe("thinking ⠹  (assistant is replying...)");
  });
});

describe("ThinkingIndicator", () => {
  it("renders and clears on stop", async () => {
    const { ThinkingIndicator } = await import("../src/render/thinking_indicator.js");
    const lines: string[] = [];
    let cleared = 0;
    const indicator = new ThinkingIndicator(
      {
        render: (line) => lines.push(line),
        clear: () => {
          cleared += 1;
        },
      },
      20,
      0,
    );
    indicator.start();
    await new Promise((resolve) => setTimeout(resolve, 45));
    indicator.stop();
    expect(lines.length).toBeGreaterThan(1);
    expect(lines[0]).toContain("thinking");
    expect(cleared).toBe(1);
  });

  it("renders once when animation interval is disabled", () => {
    const lines: string[] = [];
    const indicator = new ThinkingIndicator(
      {
        render: (line) => lines.push(line),
        clear: () => undefined,
      },
      0,
      0,
    );
    indicator.start();
    indicator.stop();
    expect(lines).toEqual(["thinking ⠋  (assistant is replying...)"]);
  });
});
