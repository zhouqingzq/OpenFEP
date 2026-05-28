import { describe, expect, it } from "vitest";

import {
  ansiColorForRole,
  displayWidth,
  formatTranscriptBlock,
  formatTranscriptLine,
  resolveTranscriptLabelWidth,
  roleLabel,
  transcriptRoleFromEvent,
} from "../src/render/transcript.js";

describe("transcript rendering", () => {
  it("maps chat tail events to roles", () => {
    expect(transcriptRoleFromEvent("assistant_message")).toBe("assistant");
    expect(transcriptRoleFromEvent("proactive_turn")).toBe("proactive");
    expect(transcriptRoleFromEvent("user_message")).toBe("user");
  });

  it("formats labeled transcript lines", () => {
    const plain = formatTranscriptLine({ role: "assistant", text: "你好" }, { color: false });
    expect(plain).toBe("assistant  | 你好");
    const named = formatTranscriptLine(
      { role: "assistant", text: "你好" },
      { color: false, labels: { assistant: "胡桃", user: "周青" } },
    );
    expect(named).toContain("胡桃");
    expect(named).toContain("| 你好");
    expect(roleLabel("suppression")).toBe("suppression");
  });

  it("aligns pipe column for mixed-width speaker labels", () => {
    const labels = { user: "zq", assistant: "胡桃" };
    const labelWidth = resolveTranscriptLabelWidth(labels);
    expect(displayWidth("zq")).toBe(2);
    expect(displayWidth("胡桃")).toBe(4);
    const block = formatTranscriptBlock(
      [
        { role: "user", text: "汪汪" },
        { role: "assistant", text: "乖狗狗" },
      ],
      { color: false, labels, labelWidth },
    );
    const lines = block.split("\n");
    const pipeColumn = (line: string) => displayWidth(line.slice(0, line.indexOf("|")));
    expect(pipeColumn(lines[0]!)).toBe(labelWidth);
    expect(pipeColumn(lines[0]!)).toBe(pipeColumn(lines[1]!));
  });

  it("applies ansi colors when enabled", () => {
    const colored = formatTranscriptLine({ role: "audit", text: "runner tick" });
    expect(colored).toContain(ansiColorForRole("audit"));
    expect(colored).toContain("runner tick");
  });

  it("renders multi-line blocks", () => {
    const block = formatTranscriptBlock(
      [
        { role: "user", text: "hi" },
        { role: "proactive", text: "follow-up" },
      ],
      { color: false },
    );
    expect(block.split("\n")).toHaveLength(2);
    expect(block).toContain("proactive");
  });
});
