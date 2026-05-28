import { describe, expect, it } from "vitest";

import { badgeLabel, connectionBadge } from "../src/connection/status.js";

describe("connectionBadge", () => {
  it("reports live when subscribed", () => {
    expect(connectionBadge({ wsOpen: true, subscribed: true, reconnecting: false })).toBe("live");
    expect(badgeLabel("live")).toBe("在线");
  });

  it("reports resyncing during reconnect", () => {
    expect(connectionBadge({ wsOpen: false, subscribed: false, reconnecting: true })).toBe("resyncing");
  });

  it("reports connecting when socket open but not subscribed", () => {
    expect(connectionBadge({ wsOpen: true, subscribed: false, reconnecting: false })).toBe("connecting");
  });

  it("reports offline when explicitly offline", () => {
    expect(
      connectionBadge({ wsOpen: false, subscribed: false, reconnecting: false, explicitOffline: true }),
    ).toBe("offline");
  });
});
