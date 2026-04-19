"use client";

import { useEffect, useMemo, useState } from "react";
import {
  getSetupStatus,
  isDesktopRuntime,
  runSetupStep,
  type SetupStatus,
  type SetupStep,
} from "@/lib/desktop-runtime";

interface Props {
  image?: string;
}

function statusClass(status: SetupStep["status"]) {
  switch (status) {
    case "ok":
      return "border-[var(--color-success)] bg-[var(--color-success-faint)] text-[var(--color-success)]";
    case "missing":
      return "border-[var(--color-danger)] bg-red-50 text-[var(--color-danger)]";
    case "blocked":
      return "border-[var(--color-warning)] bg-[var(--color-warning-faint)] text-[var(--color-warning)]";
    case "warning":
      return "border-[var(--color-warning)] bg-[var(--color-warning-faint)] text-[var(--color-warning)]";
    case "action":
      return "border-[var(--color-pitch)] bg-[var(--color-pitch-faint)] text-[var(--color-pitch)]";
    default:
      return "border-[var(--color-line)] bg-[var(--color-surface)] text-[var(--color-muted)]";
  }
}

function statusLabel(status: SetupStep["status"]) {
  return {
    ok: "Ready",
    missing: "Install",
    blocked: "Blocked",
    optional: "Optional",
    warning: "Check",
    action: "Run",
  }[status];
}

export default function SetupWizard({ image }: Props) {
  const [desktop, setDesktop] = useState(false);
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const isDesktop = isDesktopRuntime();
    setDesktop(isDesktop);
    if (!isDesktop) return;

    getSetupStatus(image)
      .then((next) => {
        setStatus(next);
        setOpen(!next.ready_for_local);
      })
      .catch((error) => {
        setMessage(error instanceof Error ? error.message : String(error));
        setOpen(true);
      });
  }, [image]);

  const progress = useMemo(() => {
    if (!status?.steps.length) return 0;
    const ready = status.steps.filter((step) =>
      ["ok", "optional", "action"].includes(step.status),
    ).length;
    return Math.round((ready / status.steps.length) * 100);
  }, [status]);

  if (!desktop) return null;

  async function refresh() {
    const next = await getSetupStatus(image);
    setStatus(next);
  }

  async function runStep(step: SetupStep) {
    setBusyKey(step.key);
    setMessage(null);
    try {
      const result = await runSetupStep(step.key, image);
      setMessage(result.message);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <section className="panel mb-6 p-5">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-4 text-left"
      >
        <span>
          <span className="text-xs font-semibold uppercase tracking-[0.08em] text-[var(--color-muted)]">
            First-run installer
          </span>
          <span className="mt-1 block text-lg font-bold text-[var(--color-ink)]">
            {status?.ready_for_local
              ? "Local runtime is ready"
              : "Complete the runtime setup"}
          </span>
        </span>
        <span className="rounded-lg border border-[var(--color-line)] px-3 py-1 text-sm font-semibold text-[var(--color-ink)]">
          {progress}%
        </span>
      </button>

      {open && (
        <div className="mt-5">
          <p className="max-w-3xl text-sm text-[var(--color-muted)]">
            The installer checks Docker, local NVIDIA support, the TRIBE image,
            Vast.ai fallback, and signed release updates. Run each action in
            order, then refresh checks.
          </p>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {status?.steps.map((step) => (
              <div
                key={step.key}
                className="rounded-lg border border-[var(--color-line)] p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold text-[var(--color-ink)]">
                      {step.title}
                    </h3>
                    <p className="mt-1 text-sm text-[var(--color-muted)]">
                      {step.detail}
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-lg border px-2 py-1 text-xs font-semibold ${statusClass(step.status)}`}
                  >
                    {statusLabel(step.status)}
                  </span>
                </div>

                {step.action_label && (
                  <button
                    type="button"
                    onClick={() => runStep(step)}
                    disabled={busyKey !== null}
                    className="mt-4 rounded-lg border border-[var(--color-line)] px-3 py-2 text-sm font-semibold text-[var(--color-ink)] transition-colors hover:border-[var(--color-pitch)] hover:text-[var(--color-pitch)] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {busyKey === step.key ? "Working..." : step.action_label}
                  </button>
                )}
              </div>
            ))}
          </div>

          {message && (
            <p className="mt-4 text-sm text-[var(--color-muted)]" role="status">
              {message}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
