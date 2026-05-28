import { describe, expect, it } from "vitest";

import { withTimeout } from "../src/repl_io.js";

describe("withTimeout", () => {
  it("resolves when the promise completes in time", async () => {
    await expect(withTimeout(Promise.resolve("ok"), 100, "test")).resolves.toBe("ok");
  });

  it("rejects when the promise is too slow", async () => {
    await expect(
      withTimeout(new Promise((resolve) => setTimeout(resolve, 50)), 5, "slow op"),
    ).rejects.toThrow("slow op timed out");
  });
});
