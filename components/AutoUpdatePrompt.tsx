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
    <section className="pc-update-overlay" role="dialog" aria-modal="true" aria-label="Update available">
      <div className="pc-update-card">
        <i className="pc-corner tl" />
        <i className="pc-corner tr" />
        <i className="pc-corner bl" />
        <i className="pc-corner br" />
        <div className="pc-update-head">
          <span className="led warn pulse" />
          <span className="label">Update available</span>
        </div>
        <h2>PitchCheck {update.version} is ready to install.</h2>
        <p className="mono">
          Signed desktop update. The app will restart after installation.
        </p>
        {update.body && (
          <div className="pc-update-notes">
            <span className="label">Release notes</span>
            <p>{update.body.slice(0, 280)}</p>
          </div>
        )}
        {message && <p className="pc-update-message mono">{message}</p>}
        <div className="pc-update-actions">
          <button type="button" className="pc-button ghost" onClick={() => setDismissed(true)} disabled={installing}>
            Later
          </button>
          <button type="button" className="pc-button primary" onClick={installUpdate} disabled={installing}>
            {installing ? "Installing..." : "Install and restart"}
          </button>
        </div>
      </div>
    </section>
  );
}
