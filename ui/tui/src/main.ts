#!/usr/bin/env node
import { runRepl } from "./repl.js";

export interface CliArgs {
  personaId: string;
  sessionId: string;
  gatewayUrl: string;
  userName?: string;
  authToken?: string;
  noColor: boolean;
  verboseAudit: boolean;
}

export function parseCliArgs(argv: string[]): CliArgs {
  const args = [...argv];
  let personaId = "胡桃";
  let sessionId = "tui_demo";
  let gatewayUrl = process.env.SEGMENTS_CONSCIOUSNESS_GATEWAY_URL?.trim() || "http://127.0.0.1:8765";
  let userName = process.env.SEGMENTS_USER_NAME?.trim() || undefined;
  let authToken = process.env.M16_AUTH_TOKEN?.trim() || undefined;
  let noColor = false;
  let verboseAudit = false;

  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (token === "--persona" && args[index + 1]) {
      personaId = args[++index];
      continue;
    }
    if (token === "--session" && args[index + 1]) {
      sessionId = args[++index];
      continue;
    }
    if (token === "--user-name" && args[index + 1]) {
      userName = args[++index];
      continue;
    }
    if ((token === "--gateway" || token === "--base-url") && args[index + 1]) {
      gatewayUrl = args[++index];
      continue;
    }
    if (token === "--auth-token" && args[index + 1]) {
      authToken = args[++index];
      continue;
    }
    if (token === "--no-color") {
      noColor = true;
      continue;
    }
    if (token === "--verbose-audit") {
      verboseAudit = true;
    }
  }

  return { personaId, sessionId, gatewayUrl, userName, authToken, noColor, verboseAudit };
}

async function main(): Promise<void> {
  const options = parseCliArgs(process.argv.slice(2));
  await runRepl({
    personaId: options.personaId,
    sessionId: options.sessionId,
    gatewayUrl: options.gatewayUrl,
    userName: options.userName,
    authToken: options.authToken,
    color: !options.noColor,
    verboseAudit: options.verboseAudit,
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
