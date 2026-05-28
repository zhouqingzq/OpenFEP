import { describe, expect, it } from "vitest";

import { commandHelpText, parseReplLine } from "../src/commands.js";

describe("parseReplLine", () => {
  it("parses slash commands", () => {
    expect(parseReplLine("/status")).toEqual({ kind: "command", command: "status" });
    expect(parseReplLine("/snapshot")).toEqual({ kind: "command", command: "snapshot" });
    expect(parseReplLine("/debug")).toEqual({ kind: "command", command: "debug" });
    expect(parseReplLine("/start-runner")).toEqual({ kind: "command", command: "start-runner" });
    expect(parseReplLine("/quit")).toEqual({ kind: "command", command: "quit" });
  });

  it("treats unknown slash input as help", () => {
    expect(parseReplLine("/unknown")).toEqual({ kind: "command", command: "help", text: "/unknown" });
  });

  it("passes plain text through as user messages", () => {
    expect(parseReplLine("你好")).toEqual({ kind: "message", text: "你好" });
  });

  it("ignores empty lines", () => {
    expect(parseReplLine("   ")).toEqual({ kind: "empty" });
  });
});

describe("commandHelpText", () => {
  it("documents required operator commands", () => {
    const help = commandHelpText();
    expect(help).toContain("/status");
    expect(help).toContain("/start-runner");
    expect(help).toContain("/snapshot");
    expect(help).toContain("/debug");
    expect(help).toContain("/quit");
  });
});
