import { describe, expect, it } from "vitest";

import {
  ansiColorForRole,
  formatTranscriptBlock,
  formatTranscriptLine,
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
