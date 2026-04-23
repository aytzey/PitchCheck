import * as SecureStore from "expo-secure-store";
import { Platform } from "./types";

const KEY = "pitchcheck-mobile-pending-score";

export type PendingScoreDraft = {
  message: string;
  persona: string;
  platform: Platform;
  queuedAt: string;
};

function isPlatform(value: unknown): value is Platform {
  return (
    value === "email" ||
    value === "linkedin" ||
    value === "cold-call-script" ||
    value === "landing-page" ||
    value === "ad-copy" ||
    value === "general"
  );
}

export async function loadPendingDraft(): Promise<PendingScoreDraft | null> {
  const raw = await SecureStore.getItemAsync(KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<PendingScoreDraft>;
    if (
      typeof parsed.message === "string" &&
      typeof parsed.persona === "string" &&
      isPlatform(parsed.platform)
    ) {
      return {
        message: parsed.message,
        persona: parsed.persona,
        platform: parsed.platform,
        queuedAt: typeof parsed.queuedAt === "string" ? parsed.queuedAt : new Date().toISOString(),
      };
    }
    return null;
  } catch {
    return null;
  }
}

export async function savePendingDraft(draft: PendingScoreDraft): Promise<void> {
  await SecureStore.setItemAsync(KEY, JSON.stringify(draft));
}

export async function clearPendingDraft(): Promise<void> {
  await SecureStore.deleteItemAsync(KEY);
}
