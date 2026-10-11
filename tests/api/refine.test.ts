import { expect, it, vi } from "vitest";
import { POST } from "@/app/api/refine/route";

it("rejects oversized chunked refinement bodies before calling the provider", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  try {
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(70_000));
        controller.enqueue(new Uint8Array(70_000));
        controller.close();
      },
    });
    const request = new Request("http://localhost/api/refine", {
      method: "POST", body, duplex: "half",
    } as RequestInit);
    expect((await POST(request)).status).toBe(413);
    expect(fetch).not.toHaveBeenCalled();
  } finally {
    vi.unstubAllGlobals();
  }
});

it("releases a stalled request body after the read deadline", async () => {
  vi.useFakeTimers();
  try {
    const request = new Request("http://localhost/api/refine", {
      method: "POST", body: new ReadableStream(), duplex: "half",
    } as RequestInit);
    const response = POST(request);
    await vi.advanceTimersByTimeAsync(15_000);
    expect((await response).status).toBe(408);
  } finally {
    vi.useRealTimers();
  }
});
