export type CommandName = "status" | "snapshot" | "debug" | "quit" | "help";

export interface ParsedInput {
  kind: "command" | "message" | "empty";
  command?: CommandName;
  text?: string;
}

const COMMAND_NAMES: ReadonlySet<string> = new Set(["status", "snapshot", "debug", "quit", "help"]);

export function parseReplLine(line: string): ParsedInput {
  const trimmed = line.trim();
  if (!trimmed) {
    return { kind: "empty" };
  }
  if (trimmed.startsWith("/")) {
    const token = trimmed.slice(1).split(/\s+/)[0]?.toLowerCase() ?? "";
    if (COMMAND_NAMES.has(token)) {
      return { kind: "command", command: token as CommandName };
    }
    return { kind: "command", command: "help", text: trimmed };
  }
  return { kind: "message", text: trimmed };
}

export function commandHelpText(): string {
  return [
    "Commands:",
    "  /status    runner health + connection summary",
    "  /snapshot  print latest gateway snapshot JSON",
    "  /debug     copy-friendly mind debug block",
    "  /quit      exit",
    "  /help      show this help",
  ].join("\n");
}
