import { describe, expect, it } from "vitest";

import { formatProgressBar, formatReplyProgressLine } from "../src/render/reply_progress.js";

describe("reply progress", () => {
  it("renders a bounded progress bar", () => {
    expect(formatProgressBar(0)).toBe("[                    ]");
    expect(formatProgressBar(0.5, 10)).toBe("[====>    ]");
    expect(formatProgressBar(1, 8)).toBe("[========]");
  });

  it("formats persona reply line with stage percent", () => {
    const line = formatReplyProgressLine("胡桃", 42);
    expect(line).toContain("胡桃正在回复...");
    expect(line).toContain("42%");
  });

  it("clamps percent to 0-100", () => {
    expect(formatReplyProgressLine("胡桃", 150)).toContain("100%");
    expect(formatReplyProgressLine("胡桃", -5)).toContain("0%");
  });
});
