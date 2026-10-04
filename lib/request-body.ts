const configuredLimit = Number.parseInt(process.env.PITCHCHECK_MAX_REQUEST_BODY_BYTES ?? "", 10);
const MAX_REQUEST_BODY_BYTES = Number.isFinite(configuredLimit) ? Math.max(1024, configuredLimit) : 128 * 1024;
const REQUEST_BODY_TOO_LARGE = Symbol("REQUEST_BODY_TOO_LARGE");

function bodyIsTooLarge(request: Request): boolean {
  const rawLength = request.headers.get("content-length");
  if (!rawLength) return false;
  const length = Number.parseInt(rawLength, 10);
  return Number.isFinite(length) && length > MAX_REQUEST_BODY_BYTES;
}

async function readRequestTextWithLimit(
  request: Request,
  maxBytes: number,
): Promise<string | typeof REQUEST_BODY_TOO_LARGE> {
  if (!request.body) {
    return "";
  }

  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  const chunks: string[] = [];
  let totalBytes = 0;
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error("Body read timed out")), 15_000);
  });

  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), deadline]);
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > maxBytes) {
        await reader.cancel();
        return REQUEST_BODY_TOO_LARGE;
      }
      chunks.push(decoder.decode(value, { stream: true }));
    }
    chunks.push(decoder.decode());
    return chunks.join("");
  } finally {
    clearTimeout(timer!);
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function readJsonBody(
  request: Request,
): Promise<
  | { ok: true; body: unknown }
  | { ok: false; status: number; error: string }
> {
  if (bodyIsTooLarge(request)) {
    return {
      ok: false,
      status: 413,
      error: `Request body must be at most ${MAX_REQUEST_BODY_BYTES} bytes.`,
    };
  }

  let rawBody: string | typeof REQUEST_BODY_TOO_LARGE;
  try {
    rawBody = await readRequestTextWithLimit(request, MAX_REQUEST_BODY_BYTES);
  } catch {
    return { ok: false, status: 408, error: "Request body timed out or disconnected." };
  }
  if (rawBody === REQUEST_BODY_TOO_LARGE) {
    return {
      ok: false,
      status: 413,
      error: `Request body must be at most ${MAX_REQUEST_BODY_BYTES} bytes.`,
    };
  }

  try {
    return { ok: true, body: JSON.parse(rawBody) };
  } catch {
    return { ok: true, body: null };
  }
}
