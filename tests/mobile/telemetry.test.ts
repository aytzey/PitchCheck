import { describe, expect, it } from "vitest";
import { clearRuntimeEvents, getRuntimeEvents, logRuntimeEvent } from "@/mobile/src/telemetry";

describe("mobile telemetry", () => {
  it("stores and clears runtime events", () => {
    clearRuntimeEvents();
    logRuntimeEvent({
      at: new Date().toISOString(),
      level: "info",
      event: "score_start",
      requestId: "pc_test",
      details: { runtime: "pitchserver" },
    });

    const events = getRuntimeEvents();
    expect(events.length).toBe(1);
    expect(events[0].event).toBe("score_start");

    clearRuntimeEvents();
    expect(getRuntimeEvents()).toEqual([]);
  });
});
