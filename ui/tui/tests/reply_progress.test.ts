import { describe, expect, it } from "vitest";

import { formatProgressBar, formatReplyProgressLine } from "../src/render/reply_progress.js";

describe("reply progress", () => {
  it("renders a bounded progress bar", () => {
    expect(formatProgressBar(0)).toBe("[                    ]");
    expect(formatProgressBar(0.5, 10)).toBe("[====>    ]");
    expect(formatProgressBar(1, 8)).toBe("[========]");
  });

  it("formats persona reply line with percent", () => {
    const line = formatReplyProgressLine("胡桃", 60_000, 120_000);
    expect(line).toContain("胡桃正在回复...");
    expect(line).toContain("50%");
  });
});
