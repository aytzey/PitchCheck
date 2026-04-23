export type RuntimeEvent = {
  at: string;
  level: "info" | "warn" | "error";
  event: string;
  requestId?: string;
  details?: Record<string, unknown>;
};

const events: RuntimeEvent[] = [];
const MAX_EVENTS = 200;

export function logRuntimeEvent(event: RuntimeEvent): void {
  events.push(event);
  if (events.length > MAX_EVENTS) events.shift();

  const line = `[mobile-runtime] ${event.event}`;
  if (event.level === "error") {
    console.error(line, { requestId: event.requestId, ...event.details });
  } else if (event.level === "warn") {
    console.warn(line, { requestId: event.requestId, ...event.details });
  } else {
    console.info(line, { requestId: event.requestId, ...event.details });
  }
}

export function getRuntimeEvents(): RuntimeEvent[] {
  return [...events];
}

export function clearRuntimeEvents(): void {
  events.length = 0;
}
