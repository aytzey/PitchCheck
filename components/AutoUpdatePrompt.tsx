"use client";

import { useEffect, useState } from "react";
import { isDesktopRuntime } from "@/lib/desktop-runtime";

interface UpdateInfo {
  version: string;
  body?: string;
  date?: string;
  downloadAndInstall: (
    onEvent?: (event: { event: string; data?: unknown }) => void,
  ) => Promise<void>;
}

export default function AutoUpdatePrompt() {
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!isDesktopRuntime()) return;

    let cancelled = false;
    async function checkForUpdate() {
      try {
        const { check } = await import("@tauri-apps/plugin-updater");
        const nextUpdate = await check();
        if (!cancelled && nextUpdate) {
          setUpdate(nextUpdate as UpdateInfo);
        }
      } catch (error) {
        if (!cancelled) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    }

    checkForUpdate();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!update || dismissed) return null;

  async function installUpdate() {
    if (!update) return;
    setInstalling(true);
    setMessage("Downloading update...");
    try {
      await update.downloadAndInstall((event) => {
        if (event.event === "Started") setMessage("Starting download...");
        if (event.event === "Progress") setMessage("Downloading update...");
        if (event.event === "Finished") setMessage("Installing update...");
      });
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
      setInstalling(false);
    }
  }

  return (
    <section className="mb-6 rounded-lg border border-[var(--color-pitch)] bg-[var(--color-pitch-faint)] p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[var(--color-pitch)]">
            Update available
          </p>
          <h2 className="text-base font-bold text-[var(--color-ink)]">
            PitchCheck {update.version}
          </h2>
          {message && (
            <p className="mt-1 text-sm text-[var(--color-muted)]">{message}</p>
          )}
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={installUpdate}
            disabled={installing}
            className="rounded-lg bg-[var(--color-pitch)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-pitch-light)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {installing ? "Installing..." : "Install and restart"}
          </button>
          <button
            type="button"
            onClick={() => setDismissed(true)}
            disabled={installing}
            className="rounded-lg border border-[var(--color-line)] px-4 py-2 text-sm font-semibold text-[var(--color-ink)] transition-colors hover:border-[var(--color-pitch)] hover:text-[var(--color-pitch)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Later
          </button>
        </div>
      </div>
    </section>
  );
}
