"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import {
  connectDesktopRuntime,
  disconnectDesktopRuntime,
  getDesktopRuntimeStatus,
  isDesktopRuntime,
  type DesktopRuntimeStatus,
} from "@/lib/desktop-runtime";

interface Props {
  onStatusChange?: (status: DesktopRuntimeStatus | null) => void;
}

const DEFAULT_IMAGE = "ghcr.io/aytzey/pitchcheck-tribe:latest";

function subscribeDesktopRuntime() {
  return () => undefined;
}

function formatPrice(value?: number) {
  return typeof value === "number" ? `$${value.toFixed(3)}/hr` : "n/a";
}

export default function DesktopConnectionPanel({ onStatusChange }: Props) {
  const desktop = useSyncExternalStore(
    subscribeDesktopRuntime,
    isDesktopRuntime,
    () => false,
  );
  const [status, setStatus] = useState<DesktopRuntimeStatus | null>(null);
  const [vastApiKey, setVastApiKey] = useState("");
  const [image, setImage] = useState(DEFAULT_IMAGE);
  const [minGpuRamGb, setMinGpuRamGb] = useState(16);
  const [maxHourlyPrice, setMaxHourlyPrice] = useState(0.45);
  const [preferInterruptible, setPreferInterruptible] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!desktop) return;

    getDesktopRuntimeStatus()
      .then((nextStatus) => {
        setStatus(nextStatus);
        setImage(nextStatus.image || DEFAULT_IMAGE);
        onStatusChange?.(nextStatus);
      })
      .catch((error) => {
        const msg = error instanceof Error ? error.message : String(error);
        setMessage(msg);
        onStatusChange?.(null);
      });
  }, [desktop, onStatusChange]);

  const sourceLabel = useMemo(() => {
    if (!status?.connected) return "Disconnected";
    return status.mode === "local" ? "Local GPU" : "Vast.ai";
  }, [status]);

  if (!desktop) return null;

  async function refreshStatus() {
    const nextStatus = await getDesktopRuntimeStatus();
    setStatus(nextStatus);
    onStatusChange?.(nextStatus);
  }

  async function connect() {
    setBusy(true);
    setMessage("Preparing runtime...");
    try {
      const nextStatus = await connectDesktopRuntime({
        vastApiKey,
        image,
        minGpuRamGb,
        maxHourlyPrice,
        preferInterruptible,
      });
      setStatus(nextStatus);
      onStatusChange?.(nextStatus);
      setMessage("Runtime connected.");
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessage(msg);
      await refreshStatus().catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }

  async function disconnect() {
    setBusy(true);
    setMessage("Disconnecting runtime...");
    try {
      const nextStatus = await disconnectDesktopRuntime(vastApiKey);
      setStatus(nextStatus);
      onStatusChange?.(nextStatus);
      setMessage("Runtime disconnected.");
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessage(msg);
      await refreshStatus().catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }

  const gpu = status?.local_gpu;

  return (
    <section className="panel mb-6 p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[var(--color-muted)]">
            Desktop runtime
          </p>
          <h2 className="mt-1 text-lg font-bold text-[var(--color-ink)]">
            {sourceLabel}
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-[var(--color-muted)]">
            Connect uses a local NVIDIA GPU when Docker can reach it. If not,
            Vast.ai rents the cheapest verified GPU offer that matches the
            limits below.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={connect}
            disabled={busy || status?.connected}
            className="rounded-lg bg-[var(--color-pitch)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-pitch-light)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Connect
          </button>
          <button
            type="button"
            onClick={disconnect}
            disabled={busy || !status?.connected}
            className="rounded-lg border border-[var(--color-line)] px-4 py-2 text-sm font-semibold text-[var(--color-ink)] transition-colors hover:border-[var(--color-danger)] hover:text-[var(--color-danger)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Disconnect
          </button>
        </div>
      </div>

      <div className="mt-5 grid gap-3 lg:grid-cols-[1.2fr_0.8fr_0.8fr_0.8fr]">
        <label className="block">
          <span className="text-xs font-semibold text-[var(--color-muted)]">
            Tribe Docker image
          </span>
          <input
            value={image}
            onChange={(event) => setImage(event.target.value)}
            className="mt-1 w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-ink)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)]"
          />
        </label>
        <label className="block">
          <span className="text-xs font-semibold text-[var(--color-muted)]">
            Min VRAM
          </span>
          <input
            type="number"
            min={8}
            value={minGpuRamGb}
            onChange={(event) => setMinGpuRamGb(Number(event.target.value))}
            className="mt-1 w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-ink)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)]"
          />
        </label>
        <label className="block">
          <span className="text-xs font-semibold text-[var(--color-muted)]">
            Max hourly
          </span>
          <input
            type="number"
            min={0.05}
            step={0.01}
            value={maxHourlyPrice}
            onChange={(event) => setMaxHourlyPrice(Number(event.target.value))}
            className="mt-1 w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-ink)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)]"
          />
        </label>
        <label className="block">
          <span className="text-xs font-semibold text-[var(--color-muted)]">
            Vast API key
          </span>
          <input
            type="password"
            value={vastApiKey}
            onChange={(event) => setVastApiKey(event.target.value)}
            placeholder="Only used in memory"
            className="mt-1 w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-ink)] placeholder:text-[var(--color-faint)] focus:border-[var(--color-pitch)] focus:outline-none focus:ring-2 focus:ring-[var(--color-pitch-faint)]"
          />
        </label>
      </div>

      <label className="mt-3 flex items-center gap-2 text-sm text-[var(--color-muted)]">
        <input
          type="checkbox"
          checked={preferInterruptible}
          onChange={(event) => setPreferInterruptible(event.target.checked)}
          className="h-4 w-4 rounded border-[var(--color-line)]"
        />
        Prefer interruptible Vast instances for lower cost.
      </label>

      <div className="mt-5 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border border-[var(--color-line)] p-3">
          <p className="text-xs text-[var(--color-muted)]">Local GPU</p>
          <p className="mt-1 font-semibold text-[var(--color-ink)]">
            {gpu?.available ? gpu.name || gpu.vendor : "Not detected"}
          </p>
        </div>
        <div className="rounded-lg border border-[var(--color-line)] p-3">
          <p className="text-xs text-[var(--color-muted)]">Service</p>
          <p className="mt-1 break-all font-semibold text-[var(--color-ink)]">
            {status?.service_url || "n/a"}
          </p>
        </div>
        <div className="rounded-lg border border-[var(--color-line)] p-3">
          <p className="text-xs text-[var(--color-muted)]">Vast offer</p>
          <p className="mt-1 font-semibold text-[var(--color-ink)]">
            {status?.offer
              ? `${status.offer.gpu_name || "GPU"} ${formatPrice(status.offer.dph_total)}`
              : "n/a"}
          </p>
        </div>
        <div className="rounded-lg border border-[var(--color-line)] p-3">
          <p className="text-xs text-[var(--color-muted)]">Instance</p>
          <p className="mt-1 font-semibold text-[var(--color-ink)]">
            {status?.vast_instance_id || status?.container_id || "n/a"}
          </p>
        </div>
      </div>

      {(message || status?.last_error) && (
        <p className="mt-4 text-sm text-[var(--color-muted)]" role="status">
          {message || status?.last_error}
        </p>
      )}
    </section>
  );
}
