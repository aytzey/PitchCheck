"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import AutoUpdatePrompt from "@/components/AutoUpdatePrompt";
import {
  changePitchServerCredentials,
  connectDesktopRuntime,
  disconnectDesktopRuntime,
  getDesktopAppConfig,
  getDesktopRuntimeStatus,
  getSetupStatus,
  isDesktopRuntime,
  refinePitchOnDesktop,
  runSetupStep,
  saveDesktopAppConfig,
  scorePitchOnDesktop,
  type DesktopAppConfig,
  type DesktopRuntimeStatus,
  type OfferSummary,
  type SetupStatus,
  type SetupStep,
} from "@/lib/desktop-runtime";
import { type FmriOutput, type PitchScoreReport, type Platform } from "@/shared/types";

type Route = "workspace" | "runtime" | "setup" | "settings";
type RuntimeKind = "local" | "vast" | "pitchserver";
type RuntimeState =
  | "not-configured"
  | "ready"
  | "connecting"
  | "deploying"
  | "connected"
  | "scoring"
  | "failed"
  | "disconnected";

type Audience = {
  name: string;
  role: string;
  relationship: string;
  context: string;
  hue: number;
};

type Medium = {
  id: Platform;
  label: string;
  hint: string;
};

type RefinedPitch = {
  before: string;
  after: string;
  model: string;
  applied: string[];
};

type RankRow = {
  label: string;
  score: number;
  note: string;
  tone: Tone;
};

const DEFAULT_IMAGE = "ghcr.io/aytzey/pitchcheck-tribe:latest";
const DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6";
const ROUTES: Route[] = ["workspace", "runtime", "setup", "settings"];

const SAMPLE_PITCH = `Hey Jordan - saw your post about the Athena migration going live last week, congrats. I know you mentioned the team was stretched thin on the observability side afterwards.

I built a small tool that auto-generates Grafana dashboards from your existing OpenTelemetry traces in about 10 minutes. A few teams at Ramp and Linear are using it in prod. No dashboard templating, no YAML.

Would you be open to a 15-min call next Tuesday or Wednesday? Happy to screen-share it against one of your services - if it's not useful you've lost nothing but the meeting slot.`;

const MEDIUMS: Medium[] = [
  { id: "email", label: "Email", hint: "Cold or warm outreach, follow-up, intro." },
  { id: "linkedin", label: "LinkedIn", hint: "Short, professional, sender photo visible." },
  { id: "cold-call-script", label: "Cold call", hint: "Opening script, 30-second ear-grab." },
  { id: "landing-page", label: "Landing page", hint: "Hero copy, subhead, CTA. Scannable." },
  { id: "ad-copy", label: "Ad copy", hint: "Headline-first, tight, platform-constrained." },
  { id: "general", label: "Other", hint: "Speech, essay, note, anything else." },
];

const DEFAULT_AUDIENCE: Audience = {
  name: "Jordan Park",
  role: "Staff Engineer",
  relationship: "Warm - saw her talk at QCon",
  context:
    "Posts about platform engineering. Pragmatic, dry sense of humor. Dislikes generic outreach. Responds to specific, evidence-heavy asks.",
  hue: 155,
};

const DEPLOY_STEPS = [
  "Searching offers",
  "Creating instance",
  "Booting container",
  "Health check",
  "Ready",
];

const PITCHSERVER_STEPS = [
  "Updating image",
  "Starting container",
  "Opening SSH tunnel",
  "Health check",
  "Ready",
];

const BRAIN_REGIONS: Record<
  string,
  {
    label: string;
    role: string;
    points: Array<[number, number]>;
    color: "ok" | "warn" | "err";
    inverted?: boolean;
  }
> = {
  emotional_engagement: {
    label: "MPFC",
    role: "Affective value",
    points: [
      [100, 40],
      [100, 58],
    ],
    color: "ok",
  },
  personal_relevance: {
    label: "mPFC/PCC",
    role: "Self-value fit",
    points: [[100, 175]],
    color: "ok",
  },
  social_proof_potential: {
    label: "TPJ/dmPFC",
    role: "Social cognition",
    points: [
      [42, 150],
      [158, 150],
    ],
    color: "warn",
  },
  memorability: {
    label: "HPC",
    role: "Encoding",
    points: [
      [68, 140],
      [132, 140],
    ],
    color: "ok",
  },
  attention_capture: {
    label: "AI/dACC",
    role: "Early salience",
    points: [
      [80, 78],
      [120, 78],
    ],
    color: "ok",
  },
  cognitive_friction: {
    label: "dlPFC",
    role: "Cognitive load",
    points: [
      [58, 62],
      [142, 62],
    ],
    color: "err",
    inverted: true,
  },
};

const BRAIN_OUTLINE_PATH =
  "M 100 20 C 128 20 152 30 168 50 C 184 70 190 95 186 120 C 184 140 180 160 172 180 C 164 200 148 216 126 222 C 116 224 108 224 100 224 C 92 224 84 224 74 222 C 52 216 36 200 28 180 C 20 160 16 140 14 120 C 10 95 16 70 32 50 C 48 30 72 20 100 20 Z";

const SULCI_LINES = [
  "M 40 80 Q 60 95 70 120 Q 75 145 65 170",
  "M 160 80 Q 140 95 130 120 Q 125 145 135 170",
  "M 50 55 Q 70 65 85 85",
  "M 150 55 Q 130 65 115 85",
  "M 35 130 Q 60 140 80 135",
  "M 165 130 Q 140 140 120 135",
  "M 55 195 Q 80 205 100 200 Q 120 205 145 195",
  "M 100 60 Q 90 90 90 130",
  "M 100 60 Q 110 90 110 130",
];

export default function DesktopWorkbench() {
  const [route, setRoute] = useState<Route>(() =>
    typeof window === "undefined" ? "workspace" : routeFromHash(window.location.hash),
  );
  const [desktopMode, setDesktopMode] = useState(false);
  const [desktopStatus, setDesktopStatus] = useState<DesktopRuntimeStatus | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [runtimeKind, setRuntimeKind] = useState<RuntimeKind>("local");
  const [busyAction, setBusyAction] = useState<"connect" | "disconnect" | null>(null);
  const [deployStep, setDeployStep] = useState(0);
  const [runtimeMessage, setRuntimeMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<PitchScoreReport | null>(null);
  const [scoring, setScoring] = useState(false);
  const [refining, setRefining] = useState(false);
  const [refinedPitch, setRefinedPitch] = useState<RefinedPitch | null>(null);
  const [message, setMessage] = useState(SAMPLE_PITCH);
  const [audience, setAudience] = useState<Audience>(DEFAULT_AUDIENCE);
  const [medium, setMedium] = useState<Medium>(MEDIUMS[0]);
  const [vastApiKey, setVastApiKey] = useState("");
  const [pitchServerSshPassword, setPitchServerSshPassword] = useState("");
  const [pitchServerUsername, setPitchServerUsername] = useState("pitchserver");
  const [pitchServerPassword, setPitchServerPassword] = useState("");
  const [pitchServerNewUsername, setPitchServerNewUsername] = useState("pitchserver");
  const [pitchServerNewPassword, setPitchServerNewPassword] = useState("");
  const [openRouterApiKey, setOpenRouterApiKey] = useState("");
  const [openRouterModel, setOpenRouterModel] = useState(DEFAULT_OPENROUTER_MODEL);
  const [openRouterRefinerModel, setOpenRouterRefinerModel] = useState(DEFAULT_OPENROUTER_MODEL);
  const [configPath, setConfigPath] = useState<string | undefined>();
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [image, setImage] = useState(DEFAULT_IMAGE);
  const [minGpuRamGb, setMinGpuRamGb] = useState(16);
  const [maxHourlyPrice, setMaxHourlyPrice] = useState(0.45);
  const [preferInterruptible, setPreferInterruptible] = useState(true);

  const refreshDesktop = useCallback(async () => {
    if (!isDesktopRuntime()) return;
    const status = await getDesktopRuntimeStatus();
    setDesktopStatus(status);
    if (status.connected && status.image) {
      setImage(status.image);
    }
    setRuntimeKind(runtimeKindFromMode(status.mode));
  }, []);

  const refreshSetup = useCallback(async () => {
    if (!isDesktopRuntime()) return;
    const status = await getSetupStatus(image);
    setSetupStatus(status);
  }, [image]);

  useEffect(() => {
    const desktop = isDesktopRuntime();
    setDesktopMode(desktop);
    if (!desktop) return;

    getDesktopAppConfig()
      .then((config) => {
        applyDesktopConfig(config, {
          setVastApiKey,
          setOpenRouterApiKey,
          setOpenRouterModel,
          setOpenRouterRefinerModel,
          setImage,
          setMinGpuRamGb,
          setMaxHourlyPrice,
          setPreferInterruptible,
          setConfigPath,
        });
      })
      .catch((caught: unknown) => {
        setRuntimeMessage(readError(caught));
      });
    refreshDesktop().catch((caught: unknown) => {
      setRuntimeMessage(readError(caught));
    });
    refreshSetup().catch((caught: unknown) => {
      setRuntimeMessage(readError(caught));
    });
  }, [refreshDesktop, refreshSetup]);

  useEffect(() => {
    const syncRoute = () => setRoute(routeFromHash(window.location.hash));
    syncRoute();
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  const setAppRoute = useCallback((nextRoute: Route) => {
    setRoute(nextRoute);
    if (typeof window === "undefined") return;
    const nextHash = `#${nextRoute}`;
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, "", nextHash);
    }
  }, []);

  useEffect(() => {
    if (busyAction !== "connect" || runtimeKind === "local") {
      setDeployStep(0);
      return;
    }

    setDeployStep(0);
    const id = window.setInterval(() => {
      setDeployStep((current) => Math.min(current + 1, DEPLOY_STEPS.length - 1));
    }, 1300);
    return () => window.clearInterval(id);
  }, [busyAction, runtimeKind]);

  useEffect(() => {
    if (!desktopMode || busyAction !== "connect") return;

    let cancelled = false;
    const syncConnectingStatus = async () => {
      try {
        const status = await getDesktopRuntimeStatus();
        if (cancelled) return;
        setDesktopStatus(status);
        setRuntimeKind(runtimeKindFromMode(status.mode));
        if (status.connected) {
          setBusyAction(null);
          setRuntimeMessage("Runtime connected.");
          setAppRoute("workspace");
          await refreshSetup().catch(() => undefined);
        }
      } catch (caught) {
        if (!cancelled) setRuntimeMessage(readError(caught));
      }
    };

    void syncConnectingStatus();
    const id = window.setInterval(() => void syncConnectingStatus(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [busyAction, desktopMode, refreshSetup, setAppRoute]);

  const runtimeState = useMemo<RuntimeState>(() => {
    if (scoring) return "scoring";
    if (desktopStatus?.connected) return "connected";
    if (busyAction === "connect") return runtimeKind === "local" ? "connecting" : "deploying";
    if (error || desktopStatus?.last_error) return "failed";
    if (desktopMode && setupStatus && !setupStatus.ready_for_local && !setupStatus.ready_for_cloud && !desktopStatus?.connected) {
      return "not-configured";
    }
    if (!desktopMode) return "connected";
    if (desktopStatus) return "disconnected";
    return "ready";
  }, [busyAction, desktopMode, desktopStatus, error, runtimeKind, scoring, setupStatus]);

  const persona = useMemo(() => formatPersona(audience), [audience]);
  const canScore =
    runtimeState === "connected" &&
    message.trim().length >= 10 &&
    persona.trim().length >= 5;

  const handleConnect = useCallback(async () => {
    if (!desktopMode) {
      setRuntimeMessage("Web mode uses the local Next.js API.");
      return;
    }

    if (runtimeKind === "pitchserver" && !pitchServerSshPassword.trim()) {
      const next = "PitchServer SSH password is required for this runtime.";
      setError(next);
      setRuntimeMessage(next);
      return;
    }
    if (runtimeKind === "pitchserver" && (!pitchServerUsername.trim() || !pitchServerPassword.trim())) {
      const next = "PitchServer username and password are required for this runtime.";
      setError(next);
      setRuntimeMessage(next);
      return;
    }

    setBusyAction("connect");
    setError(null);
    setRuntimeMessage(
      runtimeKind === "vast"
        ? "Searching for the cheapest viable Vast.ai GPU..."
        : runtimeKind === "pitchserver"
        ? "Updating PitchServer and opening the SSH tunnel..."
        : "Starting local GPU runtime...",
    );
    try {
      const status = await connectDesktopRuntime({
        runtimeKind,
        vastApiKey,
        pitchServerSshPassword,
        pitchServerUsername,
        pitchServerPassword,
        openRouterApiKey,
        openRouterModel,
        openRouterRefinerModel,
        image: runtimeKind === "pitchserver" ? DEFAULT_IMAGE : image,
        minGpuRamGb,
        maxHourlyPrice,
        preferInterruptible,
      });
      setDesktopStatus(status);
      setRuntimeKind(runtimeKindFromMode(status.mode));
      setRuntimeMessage("Runtime connected.");
      await refreshSetup().catch(() => undefined);
      setAppRoute("workspace");
    } catch (caught) {
      const next = readError(caught);
      setError(next);
      setRuntimeMessage(next);
      await refreshDesktop().catch(() => undefined);
    } finally {
      setBusyAction(null);
    }
  }, [
    desktopMode,
    image,
    maxHourlyPrice,
    minGpuRamGb,
    openRouterApiKey,
    openRouterModel,
    openRouterRefinerModel,
    pitchServerPassword,
    pitchServerSshPassword,
    pitchServerUsername,
    preferInterruptible,
    refreshDesktop,
    refreshSetup,
    runtimeKind,
    setAppRoute,
    vastApiKey,
  ]);

  const handleDisconnect = useCallback(async () => {
    if (!desktopMode) return;
    setBusyAction("disconnect");
    setRuntimeMessage("Disconnecting runtime...");
    try {
      const status = await disconnectDesktopRuntime(vastApiKey);
      setDesktopStatus(status);
      setReport(null);
      setRuntimeMessage("Runtime disconnected.");
    } catch (caught) {
      const next = readError(caught);
      setError(next);
      setRuntimeMessage(next);
      await refreshDesktop().catch(() => undefined);
    } finally {
      setBusyAction(null);
    }
  }, [desktopMode, refreshDesktop, vastApiKey]);

  const handleChangePitchServerCredentials = useCallback(async () => {
    if (!desktopMode) return;
    if (desktopStatus?.mode !== "pitchserver" || !desktopStatus.connected) {
      setRuntimeMessage("Connect PitchServer before changing its login.");
      setAppRoute("runtime");
      return;
    }
    const currentPassword = pitchServerPassword;
    const newUsername = (pitchServerNewUsername || pitchServerUsername).trim();
    const newPassword = pitchServerNewPassword;
    if (!currentPassword.trim() || !newUsername || !newPassword.trim()) {
      setRuntimeMessage("Current password, new username, and new password are required.");
      return;
    }

    setBusyAction("connect");
    setError(null);
    setRuntimeMessage("Updating PitchServer login...");
    try {
      const result = await changePitchServerCredentials({
        currentPassword,
        newUsername,
        newPassword,
      });
      if (!result.ok) throw new Error(result.error || "PitchServer login update failed.");
      setPitchServerUsername(result.username || newUsername);
      setPitchServerPassword(newPassword);
      setPitchServerNewUsername(result.username || newUsername);
      setPitchServerNewPassword("");
      setRuntimeMessage("PitchServer login updated.");
    } catch (caught) {
      const next = readError(caught);
      setError(next);
      setRuntimeMessage(next);
    } finally {
      setBusyAction(null);
    }
  }, [
    desktopMode,
    desktopStatus?.connected,
    desktopStatus?.mode,
    pitchServerNewPassword,
    pitchServerNewUsername,
    pitchServerPassword,
    pitchServerUsername,
    setAppRoute,
  ]);

  const handleScore = useCallback(async (messageOverride?: string) => {
    const nextMessage = (typeof messageOverride === "string" ? messageOverride : message).trim();
    const nextCanScore =
      runtimeState === "connected" &&
      nextMessage.length >= 10 &&
      persona.trim().length >= 5;

    if (!nextCanScore) {
      if (runtimeState !== "connected") {
        setError("Connect a runtime before scoring.");
        setAppRoute("runtime");
      }
      return;
    }

    setScoring(true);
    setRefinedPitch(null);
    setError(null);
    try {
      if (desktopMode) {
        const data = await scorePitchOnDesktop({
          message: nextMessage,
          persona,
          platform: medium.id,
          openRouterModel,
        });
        if (!data.report) throw new Error(data.error || "Scoring failed.");
        setReport(data.report as PitchScoreReport);
      } else {
        const res = await fetch("/api/score", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: nextMessage,
            persona,
            platform: medium.id,
            openRouterModel,
          }),
        });
        const data = (await res.json()) as { report?: PitchScoreReport; error?: string };
        if (!res.ok || !data.report) throw new Error(data.error || "Scoring failed.");
        setReport(data.report);
      }
    } catch (caught) {
      setError(readError(caught));
    } finally {
      setScoring(false);
    }
  }, [desktopMode, medium.id, message, openRouterModel, persona, runtimeState, setAppRoute]);

  const handleRefine = useCallback(async () => {
    if (!report) return;
    const source = message.trim();
    if (source.length < 10) {
      setError("Write a message before refining.");
      return;
    }
    if (desktopMode && !openRouterApiKey.trim()) {
      setError("Add your OpenRouter API key in Settings, then save runtime.env.");
      setAppRoute("settings");
      return;
    }

    const refineBrief = extractRefineBrief(report);
    setRefining(true);
    setError(null);
    try {
      if (desktopMode) {
        const data = await refinePitchOnDesktop({
          message: source,
          persona,
          platform: medium.id,
          suggestions: refineBrief,
        });
        if (!data.refinedMessage) throw new Error(data.error || "Refine failed.");
        setRefinedPitch({
          before: source,
          after: data.refinedMessage.trim(),
          model: data.model || openRouterRefinerModel || openRouterModel || DEFAULT_OPENROUTER_MODEL,
          applied: refineBrief,
        });
      } else {
        setRefinedPitch({
          before: source,
          after: buildPreviewRewrite(source, refineBrief),
          model: "Web preview rewrite",
          applied: refineBrief,
        });
      }
    } catch (caught) {
      setError(readError(caught));
    } finally {
      setRefining(false);
    }
  }, [
    desktopMode,
    medium.id,
    message,
    openRouterApiKey,
    openRouterModel,
    openRouterRefinerModel,
    persona,
    report,
    setAppRoute,
  ]);

  const handleAcceptRefine = useCallback(() => {
    if (!refinedPitch) return;
    const nextMessage = refinedPitch.after;
    setMessage(nextMessage);
    setRefinedPitch(null);
    void handleScore(nextMessage);
  }, [handleScore, refinedPitch]);

  const handleAcceptRefineForEditing = useCallback(() => {
    if (!refinedPitch) return;
    setMessage(refinedPitch.after);
    setRefinedPitch(null);
  }, [refinedPitch]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        void handleScore();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleScore]);

  return (
    <main className="pc-shell grid-bg">
      <TopBar desktopMode={desktopMode} state={runtimeState} runtimeKind={runtimeKind} route={route} onRoute={setAppRoute} />
      <AutoUpdatePrompt />
      {route === "workspace" && (
        <WorkspaceView
          state={runtimeState}
          runtimeKind={runtimeKind}
          medium={medium}
          setMedium={setMedium}
          audience={audience}
          setAudience={setAudience}
          message={message}
          setMessage={(nextMessage) => {
            setMessage(nextMessage);
            setRefinedPitch(null);
          }}
          report={report}
          error={error}
          openRouterModel={openRouterModel}
          scoring={scoring}
          refining={refining}
          refinedPitch={refinedPitch}
          canScore={canScore}
          onScore={handleScore}
          onRefine={handleRefine}
          onAcceptRefine={handleAcceptRefine}
          onAcceptRefineForEditing={handleAcceptRefineForEditing}
          onDiscardRefine={() => setRefinedPitch(null)}
          onClear={() => {
            setMessage("");
            setReport(null);
            setError(null);
            setRefinedPitch(null);
          }}
          onConnect={handleConnect}
        />
      )}
      {route === "runtime" && (
        <RuntimeView
          desktopMode={desktopMode}
          state={runtimeState}
          runtimeKind={runtimeKind}
          setRuntimeKind={setRuntimeKind}
          status={desktopStatus}
          vastApiKey={vastApiKey}
          setVastApiKey={setVastApiKey}
          pitchServerSshPassword={pitchServerSshPassword}
          setPitchServerSshPassword={setPitchServerSshPassword}
          pitchServerUsername={pitchServerUsername}
          setPitchServerUsername={setPitchServerUsername}
          pitchServerPassword={pitchServerPassword}
          setPitchServerPassword={setPitchServerPassword}
          pitchServerNewUsername={pitchServerNewUsername}
          setPitchServerNewUsername={setPitchServerNewUsername}
          pitchServerNewPassword={pitchServerNewPassword}
          setPitchServerNewPassword={setPitchServerNewPassword}
          openRouterApiKey={openRouterApiKey}
          setOpenRouterApiKey={setOpenRouterApiKey}
          openRouterModel={openRouterModel}
          setOpenRouterModel={setOpenRouterModel}
          openRouterRefinerModel={openRouterRefinerModel}
          setOpenRouterRefinerModel={setOpenRouterRefinerModel}
          image={image}
          setImage={setImage}
          minGpuRamGb={minGpuRamGb}
          setMinGpuRamGb={setMinGpuRamGb}
          maxHourlyPrice={maxHourlyPrice}
          setMaxHourlyPrice={setMaxHourlyPrice}
          preferInterruptible={preferInterruptible}
          setPreferInterruptible={setPreferInterruptible}
          deployStep={deployStep}
          busy={busyAction !== null}
          message={runtimeMessage}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          onChangePitchServerCredentials={handleChangePitchServerCredentials}
          onRefresh={() => {
            void refreshDesktop();
            void refreshSetup();
          }}
        />
      )}
      {route === "setup" && (
        <SetupView
          desktopMode={desktopMode}
          setupStatus={setupStatus}
          image={image}
          onRefresh={() => void refreshSetup()}
          onRunStep={async (step) => {
            await runSetupStep(step.key, image);
            await refreshSetup();
          }}
          onComplete={() => setAppRoute("workspace")}
        />
      )}
      {route === "settings" && (
        <SettingsView
          vastApiKey={vastApiKey}
          setVastApiKey={setVastApiKey}
          openRouterApiKey={openRouterApiKey}
          setOpenRouterApiKey={setOpenRouterApiKey}
          openRouterModel={openRouterModel}
          setOpenRouterModel={setOpenRouterModel}
          openRouterRefinerModel={openRouterRefinerModel}
          setOpenRouterRefinerModel={setOpenRouterRefinerModel}
          image={image}
          setImage={setImage}
          minGpuRamGb={minGpuRamGb}
          setMinGpuRamGb={setMinGpuRamGb}
          maxHourlyPrice={maxHourlyPrice}
          setMaxHourlyPrice={setMaxHourlyPrice}
          preferInterruptible={preferInterruptible}
          setPreferInterruptible={setPreferInterruptible}
          configPath={configPath}
          settingsMessage={settingsMessage}
          onSave={async () => {
            setSettingsMessage("Saving runtime.env...");
            try {
              const saved = await saveDesktopAppConfig({
                vastApiKey,
                openRouterApiKey,
                openRouterModel,
                openRouterRefinerModel,
                image,
                minGpuRamGb,
                maxHourlyPrice,
                preferInterruptible,
              });
              applyDesktopConfig(saved, {
                setVastApiKey,
                setOpenRouterApiKey,
                setOpenRouterModel,
                setOpenRouterRefinerModel,
                setImage,
                setMinGpuRamGb,
                setMaxHourlyPrice,
                setPreferInterruptible,
                setConfigPath,
              });
              setSettingsMessage("Saved machine-local runtime.env.");
            } catch (caught) {
              setSettingsMessage(readError(caught));
            }
          }}
        />
      )}
      <StatusStrip
        desktopMode={desktopMode}
        state={runtimeState}
        runtimeKind={runtimeKind}
        report={report}
        cost={desktopStatus?.offer?.dph_total}
        model={openRouterModel || DEFAULT_OPENROUTER_MODEL}
      />
    </main>
  );
}

function TopBar({
  desktopMode,
  state,
  runtimeKind,
  route,
  onRoute,
}: {
  desktopMode: boolean;
  state: RuntimeState;
  runtimeKind: RuntimeKind;
  route: Route;
  onRoute: (route: Route) => void;
}) {
  return (
    <div className="pc-topbar">
      <div className="pc-brand">
        <Logo />
        <strong>PitchCheck</strong>
        <span className="mono">v0.1.4</span>
      </div>
      <nav className="pc-tabs nodrag" aria-label="Primary">
        {ROUTES.map((item) => (
          <button key={item} className={route === item ? "active" : ""} onClick={() => onRoute(item)} type="button">
            {capitalize(item)}
          </button>
        ))}
      </nav>
      <div className="pc-topbar-spacer" />
      <RuntimeBadge desktopMode={desktopMode} state={state} runtimeKind={runtimeKind} />
    </div>
  );
}

function WorkspaceView({
  state,
  runtimeKind,
  medium,
  setMedium,
  audience,
  setAudience,
  message,
  setMessage,
  report,
  error,
  openRouterModel,
  scoring,
  refining,
  refinedPitch,
  canScore,
  onScore,
  onRefine,
  onAcceptRefine,
  onAcceptRefineForEditing,
  onDiscardRefine,
  onClear,
  onConnect,
}: {
  state: RuntimeState;
  runtimeKind: RuntimeKind;
  medium: Medium;
  setMedium: (medium: Medium) => void;
  audience: Audience;
  setAudience: (audience: Audience) => void;
  message: string;
  setMessage: (message: string) => void;
  report: PitchScoreReport | null;
  error: string | null;
  openRouterModel: string;
  scoring: boolean;
  refining: boolean;
  refinedPitch: RefinedPitch | null;
  canScore: boolean;
  onScore: () => void;
  onRefine: () => void;
  onAcceptRefine: () => void;
  onAcceptRefineForEditing: () => void;
  onDiscardRefine: () => void;
  onClear: () => void;
  onConnect: () => void;
}) {
  const tokens = Math.ceil(message.length / 4.2);
  const needsConnect = state === "ready" || state === "disconnected" || state === "failed";
  const needsSetup = state === "not-configured";

  return (
    <section className="pc-workspace">
      <div className="pc-editor">
        <PlatformStrip medium={medium} setMedium={setMedium} />
        <AudienceStrip audience={audience} setAudience={setAudience} />

        <div className="pc-strip">
          <span className="label">03 / Your message</span>
          <span className="mono pc-count">
            {message.length} chars . ~{tokens} tokens
          </span>
          <button className="pc-link-button" onClick={onClear} type="button">
            Clear
          </button>
        </div>

        {refinedPitch ? (
          <DiffView refinedPitch={refinedPitch} />
        ) : (
          <div className="pc-text-well">
            <div className="pc-gutter" aria-hidden="true">
              {Array.from({ length: 24 }).map((_, index) => (
                <span key={index}>{String(index + 1).padStart(2, "0")}</span>
              ))}
            </div>
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder={`Write your ${medium.label.toLowerCase()}...\n\nTip: score against the recipient profile above. Specific context changes the result.`}
            />
          </div>
        )}

        <div className="pc-actionbar">
          {refinedPitch ? (
            <span className="pc-keyhint">
              <span>Review the candidate rewrite before re-score</span>
            </span>
          ) : (
            <span className="pc-keyhint">
              <kbd>Ctrl</kbd>
              <kbd>Enter</kbd>
              <span>to score</span>
            </span>
          )}
          <div className="pc-actionbar-spacer" />
          {refinedPitch && (
            <>
              <Button variant="ghost" onClick={onDiscardRefine}>
                Discard
              </Button>
              <Button variant="secondary" disabled={scoring} onClick={onAcceptRefineForEditing} icon={<Icon name="check" />}>
                Accept & continue editing
              </Button>
              <Button variant="primary" loading={scoring} disabled={scoring} onClick={onAcceptRefine} icon={<Icon name="spark" />}>
                {scoring ? "Re-evaluating..." : "Accept & re-evaluate"}
              </Button>
            </>
          )}
          {!refinedPitch && needsSetup && <span className="pc-muted">Runtime setup is incomplete.</span>}
          {!refinedPitch && needsConnect && <span className="pc-muted">Runtime offline.</span>}
          {!refinedPitch && (needsSetup || needsConnect) && (
            <Button variant="secondary" onClick={onConnect} icon={<Icon name="bolt" />}>
              Connect runtime
            </Button>
          )}
          {!refinedPitch && (state === "connecting" || state === "deploying") && (
            <Button variant="secondary" loading disabled>
              {state === "deploying" ? "Deploying..." : "Connecting..."}
            </Button>
          )}
          {!refinedPitch && (state === "connected" || state === "scoring") && (
            <Button variant="primary" loading={scoring} disabled={!canScore || scoring || refining} onClick={onScore} icon={<Icon name="spark" />}>
              {scoring ? "Scoring..." : "Score message"}
            </Button>
          )}
        </div>
      </div>

      <aside className="pc-result">
        <div className="pc-strip">
          <span className="label">04 / Calibration</span>
          <StatusPill tone={state === "failed" ? "err" : state === "scoring" ? "warn" : report ? "ok" : "off"} pulse={state === "scoring"}>
            {state === "scoring" ? "Analyzing" : report ? "Ready" : state === "failed" ? "Error" : "Idle"}
          </StatusPill>
        </div>
        <div className="pc-result-scroll">
          {state === "scoring" && <ScoringState />}
          {state !== "scoring" && error && <ErrorState error={error} />}
          {state !== "scoring" && !error && report && (
            <ResultView
              report={report}
              runtimeKind={runtimeKind}
              message={message}
              audience={audience}
              openRouterModel={openRouterModel}
              refining={refining}
              refinedPitch={refinedPitch}
              onRefine={onRefine}
            />
          )}
          {state !== "scoring" && !error && !report && <EmptyState state={state} />}
        </div>
      </aside>
    </section>
  );
}

function DiffView({ refinedPitch }: { refinedPitch: RefinedPitch }) {
  return (
    <div className="pc-diff-view">
      <div className="pc-diff-pane">
        <div className="pc-diff-head">
          <span className="label">Before</span>
          <span className="mono">{Math.ceil(refinedPitch.before.length / 4.2)} tokens</span>
        </div>
        <p>{refinedPitch.before}</p>
      </div>
      <div className="pc-diff-pane after">
        <div className="pc-diff-head">
          <span className="label">After . refined</span>
          <span className="mono">{refinedPitch.model}</span>
        </div>
        <p>{refinedPitch.after}</p>
      </div>
    </div>
  );
}

function PlatformStrip({ medium, setMedium }: { medium: Medium; setMedium: (medium: Medium) => void }) {
  return (
    <div className="pc-strip pc-platform-strip">
      <span className="label">01 / Medium</span>
      <div className="pc-strip-sep" />
      <div className="pc-medium-list">
        {MEDIUMS.map((item) => (
          <button key={item.id} className={item.id === medium.id ? "active" : ""} type="button" onClick={() => setMedium(item)}>
            <PlatformGlyph id={item.id} active={item.id === medium.id} />
            {item.label}
          </button>
        ))}
      </div>
      <span className="mono pc-strip-hint">{medium.hint}</span>
    </div>
  );
}

function AudienceStrip({
  audience,
  setAudience,
}: {
  audience: Audience;
  setAudience: (audience: Audience) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const initials = initialsFor(audience.name);

  function setField<K extends keyof Audience>(key: K, value: Audience[K]) {
    setAudience({ ...audience, [key]: value });
  }

  return (
    <div className="pc-audience">
      <div className="pc-audience-row">
        <span className="label">02 / Recipient</span>
        <div className="pc-strip-sep" />
        <div className="pc-avatar" style={{ "--avatar-hue": String(audience.hue) } as React.CSSProperties}>
          {initials}
        </div>
        <InlineInput value={audience.name} onChange={(value) => setField("name", value)} placeholder="Recipient name" strong />
        <span className="pc-dot">.</span>
        <InlineInput value={audience.role} onChange={(value) => setField("role", value)} placeholder="role / relationship" />
        <button className="pc-context-toggle" onClick={() => setExpanded((value) => !value)} type="button">
          {expanded ? "Less" : "Add context"}
        </button>
      </div>
      {expanded && (
        <div className="pc-context-grid">
          <label>
            <span className="label">Relationship / channel</span>
            <textarea value={audience.relationship} onChange={(event) => setField("relationship", event.target.value)} />
          </label>
          <label>
            <span className="label">What you know about them</span>
            <textarea value={audience.context} onChange={(event) => setField("context", event.target.value)} />
          </label>
        </div>
      )}
    </div>
  );
}

function RuntimeView({
  desktopMode,
  state,
  runtimeKind,
  setRuntimeKind,
  status,
  vastApiKey,
  setVastApiKey,
  pitchServerSshPassword,
  setPitchServerSshPassword,
  pitchServerUsername,
  setPitchServerUsername,
  pitchServerPassword,
  setPitchServerPassword,
  pitchServerNewUsername,
  setPitchServerNewUsername,
  pitchServerNewPassword,
  setPitchServerNewPassword,
  openRouterApiKey,
  setOpenRouterApiKey,
  openRouterModel,
  setOpenRouterModel,
  openRouterRefinerModel,
  setOpenRouterRefinerModel,
  image,
  setImage,
  minGpuRamGb,
  setMinGpuRamGb,
  maxHourlyPrice,
  setMaxHourlyPrice,
  preferInterruptible,
  setPreferInterruptible,
  deployStep,
  busy,
  message,
  onConnect,
  onDisconnect,
  onChangePitchServerCredentials,
  onRefresh,
}: {
  desktopMode: boolean;
  state: RuntimeState;
  runtimeKind: RuntimeKind;
  setRuntimeKind: (kind: RuntimeKind) => void;
  status: DesktopRuntimeStatus | null;
  vastApiKey: string;
  setVastApiKey: (key: string) => void;
  pitchServerSshPassword: string;
  setPitchServerSshPassword: (key: string) => void;
  pitchServerUsername: string;
  setPitchServerUsername: (value: string) => void;
  pitchServerPassword: string;
  setPitchServerPassword: (value: string) => void;
  pitchServerNewUsername: string;
  setPitchServerNewUsername: (value: string) => void;
  pitchServerNewPassword: string;
  setPitchServerNewPassword: (value: string) => void;
  openRouterApiKey: string;
  setOpenRouterApiKey: (key: string) => void;
  openRouterModel: string;
  setOpenRouterModel: (model: string) => void;
  openRouterRefinerModel: string;
  setOpenRouterRefinerModel: (model: string) => void;
  image: string;
  setImage: (image: string) => void;
  minGpuRamGb: number;
  setMinGpuRamGb: (value: number) => void;
  maxHourlyPrice: number;
  setMaxHourlyPrice: (value: number) => void;
  preferInterruptible: boolean;
  setPreferInterruptible: (value: boolean) => void;
  deployStep: number;
  busy: boolean;
  message: string | null;
  onConnect: () => void;
  onDisconnect: () => void;
  onChangePitchServerCredentials: () => void;
  onRefresh: () => void;
}) {
  return (
    <section className="pc-runtime">
      <div className="pc-runtime-main">
        <div className="pc-strip">
          <span className="label">03 / Runtime selection</span>
          <button className="pc-link-button" type="button" onClick={onRefresh}>
            Refresh
          </button>
        </div>
        <div className="pc-runtime-cards">
          <RuntimeCard
            desktopMode={desktopMode}
            kind="local"
            selected={runtimeKind === "local"}
            state={state}
            status={status}
            onSelect={() => setRuntimeKind("local")}
            onConnect={onConnect}
            onDisconnect={onDisconnect}
            busy={busy}
          />
          <RuntimeCard
            desktopMode={desktopMode}
            kind="vast"
            selected={runtimeKind === "vast"}
            state={state}
            status={status}
            onSelect={() => setRuntimeKind("vast")}
            onConnect={onConnect}
            onDisconnect={onDisconnect}
            busy={busy}
          />
          <RuntimeCard
            desktopMode={desktopMode}
            kind="pitchserver"
            selected={runtimeKind === "pitchserver"}
            state={state}
            status={status}
            onSelect={() => setRuntimeKind("pitchserver")}
            onConnect={onConnect}
            onDisconnect={onDisconnect}
            busy={busy}
          />
          <RuntimeConfig
            desktopMode={desktopMode}
            runtimeKind={runtimeKind}
            vastApiKey={vastApiKey}
            setVastApiKey={setVastApiKey}
            pitchServerSshPassword={pitchServerSshPassword}
            setPitchServerSshPassword={setPitchServerSshPassword}
            pitchServerUsername={pitchServerUsername}
            setPitchServerUsername={setPitchServerUsername}
            pitchServerPassword={pitchServerPassword}
            setPitchServerPassword={setPitchServerPassword}
            pitchServerNewUsername={pitchServerNewUsername}
            setPitchServerNewUsername={setPitchServerNewUsername}
            pitchServerNewPassword={pitchServerNewPassword}
            setPitchServerNewPassword={setPitchServerNewPassword}
            openRouterApiKey={openRouterApiKey}
            setOpenRouterApiKey={setOpenRouterApiKey}
            openRouterModel={openRouterModel}
            setOpenRouterModel={setOpenRouterModel}
            openRouterRefinerModel={openRouterRefinerModel}
            setOpenRouterRefinerModel={setOpenRouterRefinerModel}
            image={image}
            setImage={setImage}
            minGpuRamGb={minGpuRamGb}
            setMinGpuRamGb={setMinGpuRamGb}
            maxHourlyPrice={maxHourlyPrice}
            setMaxHourlyPrice={setMaxHourlyPrice}
            preferInterruptible={preferInterruptible}
            setPreferInterruptible={setPreferInterruptible}
            status={status}
            onChangePitchServerCredentials={onChangePitchServerCredentials}
            busy={busy}
          />
          {message && <div className="pc-logline mono">{message}</div>}
        </div>
      </div>
      <aside className="pc-runtime-side">
        <div className="pc-strip">
          <span className="label">04 / {runtimeKind === "local" ? "Diagnostics" : "Deployment"}</span>
          <StatusPill tone={state === "connected" ? "ok" : state === "deploying" || state === "connecting" ? "warn" : "off"} pulse={state === "deploying" || state === "connecting"}>
            {state === "connected" ? "Healthy" : state === "deploying" ? "Deploying" : state === "connecting" ? "Connecting" : "Idle"}
          </StatusPill>
        </div>
        {runtimeKind === "local" ? <LocalDiagnostics state={state} status={status} /> : <DeployTimeline runtimeKind={runtimeKind} state={state} step={deployStep} offer={status?.offer} />}
      </aside>
    </section>
  );
}

function SetupView({
  desktopMode,
  setupStatus,
  onRefresh,
  onRunStep,
  onComplete,
}: {
  desktopMode: boolean;
  setupStatus: SetupStatus | null;
  image: string;
  onRefresh: () => void;
  onRunStep: (step: SetupStep) => Promise<void>;
  onComplete: () => void;
}) {
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const steps = setupStatus?.steps ?? [];
  const readyCount = steps.filter((step) => ["ok", "optional", "action"].includes(step.status)).length;
  const progress = steps.length ? Math.round((readyCount / steps.length) * 100) : desktopMode ? 0 : 100;

  async function run(step: SetupStep) {
    setBusyKey(step.key);
    setMessage(null);
    try {
      await onRunStep(step);
      setMessage(`${step.title} completed.`);
    } catch (caught) {
      setMessage(readError(caught));
    } finally {
      setBusyKey(null);
    }
  }

  const groups = groupSetupSteps(steps);

  return (
    <section className="pc-setup">
      <div className="pc-setup-inner">
        <div className="pc-setup-head">
          <span className="label">First-run calibration</span>
          <h1>Verify your environment.</h1>
          <p>
            PitchCheck checks Docker, local NVIDIA support, the model image, Vast.ai fallback, and signed release updates.
          </p>
        </div>

        <div className="pc-progress-head">
          <span className="label">Checks</span>
          <span className="mono tnum">{readyCount} / {steps.length || 6}</span>
        </div>
        <Rail value={progress} tone={progress >= 80 ? "ok" : "warn"} />

        {!desktopMode && (
          <div className="pc-check-group">
            <CheckRow
              title="Desktop runtime"
              detail="Open the Tauri desktop app to run installer checks."
              status="optional"
              last
            />
          </div>
        )}

        {Object.entries(groups).map(([title, items]) => (
          <div key={title}>
            <div className="label pc-group-label">{title}</div>
            <div className="pc-check-group">
              {items.map((step, index) => (
                <CheckRow
                  key={step.key}
                  title={step.title}
                  detail={step.detail}
                  status={step.status}
                  last={index === items.length - 1}
                  actionLabel={step.action_label}
                  busy={busyKey === step.key}
                  onAction={() => void run(step)}
                />
              ))}
            </div>
          </div>
        ))}

        <div className="pc-setup-actions">
          <Button variant="primary" onClick={onRefresh} icon={<Icon name="retry" />}>
            Run checks
          </Button>
          <Button variant="secondary" onClick={onComplete} icon={<Icon name="arrow" />}>
            Continue to workspace
          </Button>
          <span className="pc-actionbar-spacer" />
          <Button variant="ghost" onClick={onComplete}>Skip / use Vast.ai only</Button>
        </div>
        {message && <div className="pc-logline mono">{message}</div>}
      </div>
    </section>
  );
}

function SettingsView({
  vastApiKey,
  setVastApiKey,
  openRouterApiKey,
  setOpenRouterApiKey,
  openRouterModel,
  setOpenRouterModel,
  openRouterRefinerModel,
  setOpenRouterRefinerModel,
  image,
  setImage,
  minGpuRamGb,
  setMinGpuRamGb,
  maxHourlyPrice,
  setMaxHourlyPrice,
  preferInterruptible,
  setPreferInterruptible,
  configPath,
  settingsMessage,
  onSave,
}: {
  vastApiKey: string;
  setVastApiKey: (key: string) => void;
  openRouterApiKey: string;
  setOpenRouterApiKey: (key: string) => void;
  openRouterModel: string;
  setOpenRouterModel: (model: string) => void;
  openRouterRefinerModel: string;
  setOpenRouterRefinerModel: (model: string) => void;
  image: string;
  setImage: (image: string) => void;
  minGpuRamGb: number;
  setMinGpuRamGb: (value: number) => void;
  maxHourlyPrice: number;
  setMaxHourlyPrice: (value: number) => void;
  preferInterruptible: boolean;
  setPreferInterruptible: (value: boolean) => void;
  configPath?: string;
  settingsMessage: string | null;
  onSave: () => Promise<void>;
}) {
  return (
    <section className="pc-settings">
      <div className="pc-settings-inner">
        <div className="pc-strip">
          <span className="label">Machine env</span>
          <span className="mono pc-strip-hint">{configPath || "runtime.env will be created by the desktop app."}</span>
        </div>
        <div className="pc-settings-grid">
          <Field label="Vast.ai API key" hint="Saved on this PC and used for deploy/destroy.">
            <input type="password" value={vastApiKey} onChange={(event) => setVastApiKey(event.target.value)} placeholder="Stored in runtime.env" />
          </Field>
          <Field label="OpenRouter API key" hint="Saved on this PC and injected into the TRIBE service env.">
            <input type="password" value={openRouterApiKey} onChange={(event) => setOpenRouterApiKey(event.target.value)} placeholder="Stored in runtime.env" />
          </Field>
          <Field label="Evaluator model" hint="Used for score explanations and rewrite suggestions.">
            <input value={openRouterModel} onChange={(event) => setOpenRouterModel(event.target.value)} />
          </Field>
          <Field label="Refiner model" hint="Used only when generating an accepted rewrite candidate.">
            <input value={openRouterRefinerModel} onChange={(event) => setOpenRouterRefinerModel(event.target.value)} />
          </Field>
          <Field label="Runtime image" hint="Published from GitHub Actions.">
            <input value={image} onChange={(event) => setImage(event.target.value)} />
          </Field>
          <Field label="Minimum VRAM" hint="Cheapest offers below this are ignored.">
            <input type="number" min={8} value={minGpuRamGb} onChange={(event) => setMinGpuRamGb(Number(event.target.value))} />
          </Field>
          <Field label="Max hourly price" hint="Hard ceiling for Vast.ai search.">
            <input type="number" min={0.05} step={0.01} value={maxHourlyPrice} onChange={(event) => setMaxHourlyPrice(Number(event.target.value))} />
          </Field>
          <label className="pc-checkbox">
            <input type="checkbox" checked={preferInterruptible} onChange={(event) => setPreferInterruptible(event.target.checked)} />
            <span>
              <strong>Prefer interruptible instances</strong>
              <small>Lower cost, possible eviction.</small>
            </span>
          </label>
          <div className="pc-settings-save">
            <Button variant="primary" onClick={() => void onSave()} icon={<Icon name="check" />}>
              Save runtime.env
            </Button>
            {settingsMessage && <span className="mono">{settingsMessage}</span>}
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultView({
  report,
  runtimeKind,
  message,
  audience,
  openRouterModel,
  refining,
  refinedPitch,
  onRefine,
}: {
  report: PitchScoreReport;
  runtimeKind: RuntimeKind;
  message: string;
  audience: Audience;
  openRouterModel: string;
  refining: boolean;
  refinedPitch: RefinedPitch | null;
  onRefine: () => void;
}) {
  const score = Math.round(report.persuasion_score);
  const fmri = report.fmri_output ?? fallbackFmri(score);
  const signals = report.neural_signals.length ? report.neural_signals : fallbackSignals(score);
  const breakdown = report.breakdown.map((item) => ({
    label: item.label,
    value: Math.round(item.score),
    note: item.explanation,
  }));
  const suggestions = extractSuggestions(report);

  return (
    <div className="pc-result-content">
      <BrainPanel
        score={score}
        verdict={report.verdict}
        confidence={report.robustness?.confidence ?? confidenceFromScore(score)}
        fmri={fmri}
        signals={signals}
        runtime={runtimeKind}
        latencyMs={0}
        tokens={Math.ceil(message.length / 4.2)}
        model={openRouterModel || DEFAULT_OPENROUTER_MODEL}
      />
      <NeuralSignalsGrid signals={signals} />
      {(report.robustness || report.persuasion_evidence) && (
        <RobustnessPanel report={report} />
      )}
      <div>
        <HeaderLine title="Writing facets" right="LLM interpreted" />
        <div className="pc-facet-list">
          {breakdown.map((facet, index) => (
            <FacetRow key={facet.label} facet={facet} last={index === breakdown.length - 1} />
          ))}
        </div>
      </div>
      <VariantRankPanel score={score} refinedPitch={refinedPitch} />
      <div>
        <HeaderLine title="Suggestions" right={`${suggestions.length} fixes`} />
        <div className="pc-suggestions">
          {suggestions.length ? (
            suggestions.map((suggestion, index) => (
              <div key={index}>
                <span className="mono">{String(index + 1).padStart(2, "0")}</span>
                <p>{suggestion}</p>
              </div>
            ))
          ) : (
            <div>
              <span className="mono">00</span>
              <p>{report.narrative}</p>
            </div>
          )}
        </div>
      </div>
      <div className="pc-refine">
        <div className="pc-refine-head">
          <span className="label">Auto-refine</span>
          <Button variant="primary" loading={refining} disabled={refining} onClick={onRefine} icon={<Icon name="spark" />}>
            {refining ? "Refining..." : refinedPitch ? "Refine again" : "Refine & re-score"}
          </Button>
        </div>
        <p>
          Rewrite this for <strong>{audience.name || "recipient"}</strong> with the strongest suggestions, review the diff,
          then accept it to re-score and re-rank the variants.
        </p>
      </div>
    </div>
  );
}

function VariantRankPanel({ score, refinedPitch }: { score: number; refinedPitch: RefinedPitch | null }) {
  const projectedLift = refinedPitch ? Math.max(4, Math.min(14, refinedPitch.applied.length * 3)) : 0;
  const rows = [
    refinedPitch && {
      label: "Refined candidate",
      score: Math.min(98, score + projectedLift),
      note: "Projected until accepted",
      tone: "warn" as Tone,
    },
    {
      label: "Current scored draft",
      score,
      note: "Last TRIBE score",
      tone: toneFromScore(score),
    },
    {
      label: "Persona baseline",
      score: Math.max(24, score - 11),
      note: "Control copy estimate",
      tone: toneFromScore(score - 11),
    },
  ]
    .filter(isRankRow)
    .sort((left, right) => right.score - left.score);

  return (
    <div>
      <HeaderLine title="Variant re-rank" right={refinedPitch ? "candidate pending" : "current"} />
      <div className="pc-rank-list">
        {rows.map((row, index) => (
          <div className="pc-rank-row" key={row.label}>
            <span className="mono">#{index + 1}</span>
            <strong>{row.label}</strong>
            <small>{row.note}</small>
            <b className={`mono score-${row.tone}`}>{Math.round(row.score)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function isRankRow(row: RankRow | false | null): row is RankRow {
  return Boolean(row);
}

function BrainPanel({
  score,
  verdict,
  confidence,
  fmri,
  signals,
  runtime,
  latencyMs,
  tokens,
  model,
}: {
  score: number;
  verdict: string;
  confidence: number;
  fmri: FmriOutput;
  signals: PitchScoreReport["neural_signals"];
  runtime: RuntimeKind;
  latencyMs: number;
  tokens: number;
  model: string;
}) {
  const usesSyntheticTrace = fmri.temporal_trace_basis === "synthetic_word_order";
  const traceTitle = usesSyntheticTrace ? "Engagement trace" : "Engagement timeline";
  const traceUnit = usesSyntheticTrace ? "Mean predicted response / ordered segment" : "Mean predicted response / time segment";

  return (
    <div className="pc-brain-panel">
      <CornerMarks />
      <div className="pc-brain-head">
        <span className="led ok pulse" />
        <span className="mono">TRIBE . FMRI RESPONSE PREDICTION</span>
        <strong className="mono">{fmri.voxel_count.toLocaleString()} VOXELS . {fmri.segments} SEGMENTS</strong>
      </div>
      <div className="pc-brain-hero">
        <BrainRender signals={signals} />
        <div className="pc-score-stack">
          <ScoreHalo score={score} confidence={confidence} />
          <div className="pc-verdict">
            <span className="label">Verdict</span>
            <strong>{verdict}</strong>
          </div>
        </div>
      </div>
      <PeakMeanBar fmri={fmri} />
      <div className="pc-trace-block">
        <HeaderLine title={traceTitle} right={traceUnit} />
        <TemporalTrace trace={fmri.temporal_trace} peaks={fmri.temporal_peaks} />
      </div>
      <div className="pc-brain-meta mono">
        <span>MODEL {model}</span>
        <span>RUNTIME {runtime}</span>
        <span>{latencyMs || "-"}ms . {tokens} tok</span>
      </div>
    </div>
  );
}

function BrainRender({ signals }: { signals: PitchScoreReport["neural_signals"] }) {
  const dots = useMemo(() => {
    return Object.entries(BRAIN_REGIONS).flatMap(([key, region]) => {
      const signal = signals.find((item) => item.key === key);
      if (!signal) return [];
      return region.points.map(([x, y]) => ({ x, y, region, score: signal.score }));
    });
  }, [signals]);

  return (
    <div className="pc-brain-render">
      <svg viewBox="0 0 200 240" width="180" height="216">
        <defs>
          <radialGradient id="pc-bg-ok" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="oklch(0.72 0.18 155)" stopOpacity="0.95" />
            <stop offset="60%" stopColor="oklch(0.72 0.18 155)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="oklch(0.72 0.18 155)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="pc-bg-warn" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="oklch(0.78 0.16 75)" stopOpacity="0.95" />
            <stop offset="60%" stopColor="oklch(0.78 0.16 75)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="oklch(0.78 0.16 75)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="pc-bg-err" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="oklch(0.68 0.22 25)" stopOpacity="0.95" />
            <stop offset="60%" stopColor="oklch(0.68 0.22 25)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="oklch(0.68 0.22 25)" stopOpacity="0" />
          </radialGradient>
          <clipPath id="pc-brain-clip">
            <path d={BRAIN_OUTLINE_PATH} />
          </clipPath>
        </defs>
        <path d={BRAIN_OUTLINE_PATH} fill="oklch(0.22 0.01 260)" stroke="var(--line-strong)" strokeWidth="1.2" />
        <line x1="100" y1="24" x2="100" y2="222" stroke="var(--line)" strokeWidth="0.6" strokeDasharray="2 3" />
        <g clipPath="url(#pc-brain-clip)" opacity="0.35">
          {SULCI_LINES.map((line, index) => (
            <path key={index} d={line} fill="none" stroke="var(--line-strong)" strokeWidth="0.6" />
          ))}
        </g>
        <g clipPath="url(#pc-brain-clip)">
          {dots.map((dot, index) => {
            const tone = dot.region.color;
            const gradientId = tone === "ok" ? "pc-bg-ok" : tone === "warn" ? "pc-bg-warn" : "pc-bg-err";
            const intensity = Math.max(0.25, dot.score / 100);
            return (
              <g key={`${dot.x}-${dot.y}-${index}`}>
                <circle cx={dot.x} cy={dot.y} r={18 + dot.score * 0.18} fill={`url(#${gradientId})`} opacity={intensity}>
                  <animate attributeName="opacity" values={`${intensity};${intensity * 0.6};${intensity}`} dur={`${2.4 + (index % 3) * 0.4}s`} repeatCount="indefinite" />
                </circle>
                <circle cx={dot.x} cy={dot.y} r="2.4" fill={`var(--${tone})`} />
              </g>
            );
          })}
        </g>
        {Object.entries(BRAIN_REGIONS).map(([key, region]) => {
          if (!signals.some((signal) => signal.key === key)) return null;
          const [x, y] = region.points[0];
          const outward = x < 100 ? -1 : x > 100 ? 1 : 0;
          const labelX = x + outward * 28;
          const labelY = y < 70 ? y - 10 : y + 4;
          return (
            <g key={key}>
              <line x1={x} y1={y} x2={labelX} y2={labelY} stroke="var(--fg-faint)" strokeWidth="0.5" opacity="0.5" />
              <text
                x={labelX}
                y={labelY}
                fontSize="7"
                fill={`var(--${region.color})`}
                textAnchor={outward < 0 ? "end" : outward > 0 ? "start" : "middle"}
                fontFamily="var(--ff-mono)"
                fontWeight="600"
              >
                {region.label}
              </text>
            </g>
          );
        })}
      </svg>
      <span className="mono">A . TOP-DOWN . P</span>
    </div>
  );
}

function ScoreHalo({ score, confidence }: { score: number; confidence: number }) {
  const tone = score >= 75 ? "ok" : score >= 55 ? "warn" : "err";
  return (
    <div>
      <div className="label">Persuasion score</div>
      <div className="pc-score-line">
        <span className={`mono tnum score-${tone}`}>{score}</span>
        <small className="mono">/100</small>
        <em className="mono">P = {confidence.toFixed(2)}</em>
      </div>
      <div className="pc-score-rail">
        <span style={{ left: `calc(${score}% - 1px)`, background: `var(--${tone})` }} />
      </div>
    </div>
  );
}

function PeakMeanBar({ fmri }: { fmri: FmriOutput }) {
  const ratio = fmri.global_peak_abs / Math.max(fmri.global_mean_abs, 0.001);
  return (
    <div className="pc-peak-bar">
      <MetricCell k="mean" v={fmri.global_mean_abs.toFixed(3)} note="mean predicted response" />
      <MetricCell k="peak" v={fmri.global_peak_abs.toFixed(3)} note="max predicted response" accent />
      <MetricCell k="ratio" v={`${ratio.toFixed(1)}x`} note="peak / mean" />
    </div>
  );
}

function MetricCell({ k, v, note, accent }: { k: string; v: string; note: string; accent?: boolean }) {
  return (
    <div>
      <span className="mono">{k.toUpperCase()}</span>
      <strong className={accent ? "mono tnum accent" : "mono tnum"}>{v}</strong>
      <small className="mono">{note}</small>
    </div>
  );
}

function TemporalTrace({ trace, peaks }: { trace: number[]; peaks?: number[] }) {
  const max = Math.max(...trace, 0.001);
  const peakMax = Math.max(...(peaks ?? trace), 0.001);
  const peakIndex = trace.indexOf(Math.max(...trace));

  return (
    <div className="pc-temporal">
      <div className="pc-bars">
        {trace.map((value, index) => {
          const height = (value / max) * 100;
          const peakHeight = (((peaks ?? trace)[index] ?? value) / peakMax) * 100;
          const tone = index === peakIndex || index / Math.max(trace.length - 1, 1) < 0.22 ? "ok" : index / Math.max(trace.length - 1, 1) > 0.78 ? "warn" : "muted";
          return (
            <span key={index} className={`tone-${tone}`}>
              <i style={{ height: `${Math.max(0, peakHeight - height)}%`, bottom: `${height}%` }} />
              <b style={{ height: `${height}%` }} />
            </span>
          );
        })}
      </div>
      <div className="pc-temporal-labels mono">
        <span>OPENER</span>
        <span>BODY</span>
        <span>CLOSE</span>
      </div>
    </div>
  );
}

function NeuralSignalsGrid({ signals }: { signals: PitchScoreReport["neural_signals"] }) {
  return (
    <div>
      <HeaderLine title="Neural signals" right={`Derived from ${signals.length} region analogues`} />
      <div className="pc-signal-grid">
        {signals.map((signal) => (
          <SignalCell key={signal.key} signal={signal} />
        ))}
      </div>
    </div>
  );
}

function RobustnessPanel({ report }: { report: PitchScoreReport }) {
  const robustness = report.robustness;
  const evidence = report.persuasion_evidence;
  const strategies = (evidence?.detected_strategies ?? []).slice(0, 5);
  const missing = (evidence?.missing_elements ?? []).slice(0, 3);
  const neuroAxes = robustness?.neuro_axes ? Object.entries(robustness.neuro_axes) : [];
  const warnings = [...(robustness?.warnings ?? []), ...(evidence?.warnings ?? [])]
    .filter((item, index, all) => all.indexOf(item) === index)
    .slice(0, 3);

  return (
    <div>
      <HeaderLine
        title="Robustness calibration"
        right={robustness ? `${Math.round(robustness.confidence * 100)}% confidence` : "deterministic audit"}
      />
      <div className="pc-rank-list">
        {robustness && (
          <>
            <div className="pc-rank-row">
              <span className="mono">NEU</span>
              <strong>Neuro-axis evidence</strong>
              <small>TRIBE-predicted analogues + text support</small>
              <b className={`mono score-${toneFromScore(robustness.neural_score)}`}>{Math.round(robustness.neural_score)}</b>
            </div>
            {neuroAxes.slice(0, 5).map(([key, axis]) => (
              <div className="pc-rank-row" key={key}>
                <span className="mono">{key.slice(0, 3).toUpperCase()}</span>
                <strong>{axis.label}</strong>
                <small>{axis.caveat}</small>
                <b className={`mono score-${toneFromScore(axis.score)}`}>{Math.round(axis.score)}</b>
              </div>
            ))}
            <div className="pc-rank-row">
              <span className="mono">TXT</span>
              <strong>Text evidence</strong>
              <small>Proof, CTA, audience fit, clarity</small>
              <b className={`mono score-${toneFromScore(robustness.text_score)}`}>{Math.round(robustness.text_score)}</b>
            </div>
          </>
        )}
        {evidence && (
          <div className="pc-rank-row">
            <span className="mono">CUE</span>
            <strong>{strategies.length ? strategies.join(", ") : "No strong persuasion cue"}</strong>
            <small>{missing.length ? `Missing: ${missing.join(", ")}` : "No major deterministic gap"}</small>
            <b className={`mono score-${toneFromScore(evidence.overall_text_score)}`}>{Math.round(evidence.overall_text_score)}</b>
          </div>
        )}
        {warnings.map((warning) => (
          <div className="pc-rank-row" key={warning}>
            <span className="mono">!</span>
            <strong>{warning.replaceAll("_", " ")}</strong>
            <small>Guardrail note</small>
            <b className="mono score-warn">watch</b>
          </div>
        ))}
        {(robustness?.scientific_caveats ?? []).slice(0, 1).map((caveat) => (
          <div className="pc-rank-row" key={caveat}>
            <span className="mono">SCI</span>
            <strong>Scientific caveat</strong>
            <small>{caveat}</small>
            <b className="mono score-warn">note</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function SignalCell({ signal }: { signal: PitchScoreReport["neural_signals"][number] }) {
  const region = BRAIN_REGIONS[signal.key] ?? { label: "??", role: signal.label, color: "ok" as const, points: [] };
  const tone = region.inverted
    ? signal.score >= 60
      ? "err"
      : signal.score >= 40
        ? "warn"
        : "ok"
    : signal.score >= 70
      ? "ok"
      : signal.score >= 50
        ? "warn"
        : "err";

  return (
    <div className="pc-signal-cell">
      <div>
        <span className={`mono region-${region.color}`}>{region.label}</span>
        <strong className={`mono tnum score-${tone}`}>{Math.round(signal.score)}</strong>
        {region.inverted && <small className="mono">low good</small>}
      </div>
      <p>{signal.label}</p>
      <small>{region.role}</small>
      <div className="pc-microbar">
        <span style={{ width: `${signal.score}%`, background: `var(--${tone})` }} />
      </div>
    </div>
  );
}

function RuntimeCard({
  desktopMode,
  kind,
  selected,
  state,
  status,
  onSelect,
  onConnect,
  onDisconnect,
  busy,
}: {
  desktopMode: boolean;
  kind: RuntimeKind;
  selected: boolean;
  state: RuntimeState;
  status: DesktopRuntimeStatus | null;
  onSelect: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  busy: boolean;
}) {
  const local = kind === "local";
  const pitchServer = kind === "pitchserver";
  const active = selected && (state === "connected" || state === "scoring");
  const working = selected && (state === "connecting" || state === "deploying");
  const gpu = status?.local_gpu;
  const offer = status?.offer;
  const title = !desktopMode && local ? "Web API preview" : local ? "Local GPU" : pitchServer ? "PitchServer" : "Vast.ai fallback";

  return (
    <div className={selected ? "pc-runtime-card selected" : "pc-runtime-card"}>
      {selected && <CornerMarks />}
      <div className="pc-runtime-icon">
        <Icon name={local ? "chip" : "cloud"} color={active ? "var(--ok)" : "var(--fg-muted)"} size={20} />
      </div>
      <div className="pc-runtime-body">
        <div className="pc-runtime-title">
          <h2>{title}</h2>
          {active && <StatusPill tone="ok">Active</StatusPill>}
          {working && <StatusPill tone="warn" pulse>{local ? "Connecting" : pitchServer ? "Updating" : "Deploying"}</StatusPill>}
        </div>
        <p>
          {!desktopMode && local
            ? "Scores through the local Next.js API in web preview. Open the desktop app for Docker, GPU, and Vast.ai controls."
            : local
            ? "Runs the TRIBE service on your NVIDIA GPU via Docker. Zero egress, zero rental cost."
            : pitchServer
            ? "Uses the Machinity RTX5080 box over an SSH tunnel and keeps the TRIBE image fresh on the server."
            : "Rents the cheapest viable GPU and deploys the runtime image when local hardware is unavailable."}
        </p>
        <div className="pc-spec-grid">
          {local ? (
            <>
              <Spec k="DEVICE" v={!desktopMode ? "Web API" : gpu?.available ? gpu.name || gpu.vendor || "NVIDIA" : "Not detected"} />
              <Spec k="VRAM" v={gpu?.memory_mb ? `${Math.round(gpu.memory_mb / 1024)} GB` : "-"} />
              <Spec k="DOCKER" v={!desktopMode ? "desktop only" : status?.container_id ? "running" : "-"} />
            </>
          ) : pitchServer ? (
            <>
              <Spec k="SERVER" v="RTX 5080" />
              <Spec k="SSH" v=":2022 tunnel" />
              <Spec k="DOCKER" v={active ? "running" : "remote"} />
            </>
          ) : (
            <>
              <Spec k="OFFER" v={offer?.gpu_name || "-"} />
              <Spec k="VRAM" v={offer?.gpu_ram_gb ? `${offer.gpu_ram_gb} GB` : "-"} />
              <Spec k="COST" v={formatPrice(offer?.dph_total)} tone="warn" />
            </>
          )}
        </div>
        <div className="pc-runtime-actions">
          {!desktopMode && selected && <Button variant="secondary" disabled>Web preview</Button>}
          {desktopMode && !selected && <Button variant="secondary" onClick={onSelect}>Select</Button>}
          {desktopMode && selected && state !== "connected" && state !== "scoring" && (
            <Button variant="primary" onClick={onConnect} loading={busy} disabled={busy} icon={<Icon name="bolt" />}>
              {local || pitchServer ? "Connect" : "Deploy"}
            </Button>
          )}
          {desktopMode && selected && (state === "connected" || state === "scoring") && (
            <Button variant="secondary" onClick={onDisconnect} loading={busy} disabled={busy} icon={<Icon name="plug" />}>
              Disconnect
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function RuntimeConfig({
  desktopMode,
  runtimeKind,
  vastApiKey,
  setVastApiKey,
  pitchServerSshPassword,
  setPitchServerSshPassword,
  pitchServerUsername,
  setPitchServerUsername,
  pitchServerPassword,
  setPitchServerPassword,
  pitchServerNewUsername,
  setPitchServerNewUsername,
  pitchServerNewPassword,
  setPitchServerNewPassword,
  openRouterApiKey,
  setOpenRouterApiKey,
  openRouterModel,
  setOpenRouterModel,
  openRouterRefinerModel,
  setOpenRouterRefinerModel,
  image,
  setImage,
  minGpuRamGb,
  setMinGpuRamGb,
  maxHourlyPrice,
  setMaxHourlyPrice,
  preferInterruptible,
  setPreferInterruptible,
  status,
  onChangePitchServerCredentials,
  busy,
}: {
  desktopMode: boolean;
  runtimeKind: RuntimeKind;
  vastApiKey: string;
  setVastApiKey: (key: string) => void;
  pitchServerSshPassword: string;
  setPitchServerSshPassword: (key: string) => void;
  pitchServerUsername: string;
  setPitchServerUsername: (value: string) => void;
  pitchServerPassword: string;
  setPitchServerPassword: (value: string) => void;
  pitchServerNewUsername: string;
  setPitchServerNewUsername: (value: string) => void;
  pitchServerNewPassword: string;
  setPitchServerNewPassword: (value: string) => void;
  openRouterApiKey: string;
  setOpenRouterApiKey: (key: string) => void;
  openRouterModel: string;
  setOpenRouterModel: (model: string) => void;
  openRouterRefinerModel: string;
  setOpenRouterRefinerModel: (model: string) => void;
  image: string;
  setImage: (image: string) => void;
  minGpuRamGb: number;
  setMinGpuRamGb: (value: number) => void;
  maxHourlyPrice: number;
  setMaxHourlyPrice: (value: number) => void;
  preferInterruptible: boolean;
  setPreferInterruptible: (value: boolean) => void;
  status: DesktopRuntimeStatus | null;
  onChangePitchServerCredentials: () => void;
  busy: boolean;
}) {
  const pitchServerConnected = runtimeKind === "pitchserver" && status?.mode === "pitchserver" && status.connected;
  return (
    <div className="pc-config-panel">
      <Field
        label={runtimeKind === "pitchserver" ? "PitchServer image" : "Runtime image"}
        hint={runtimeKind === "pitchserver" ? "PitchServer pulls the published GHCR image; local image aliases are ignored." : undefined}
      >
        <input
          value={runtimeKind === "pitchserver" ? DEFAULT_IMAGE : image}
          onChange={(event) => setImage(event.target.value)}
          disabled={runtimeKind === "pitchserver"}
        />
      </Field>
      {runtimeKind === "pitchserver" ? (
        <>
          <Field label="PitchServer SSH password" hint="Used only for deployment/tunnel. It is never saved to runtime.env.">
            <input type="password" value={pitchServerSshPassword} onChange={(event) => setPitchServerSshPassword(event.target.value)} placeholder="Required for SSH tunnel" />
          </Field>
          <Field label="PitchServer username" hint="Seeds the server login on first setup; after that it must match the current server login.">
            <input value={pitchServerUsername} onChange={(event) => {
              setPitchServerUsername(event.target.value);
              if (!pitchServerConnected) setPitchServerNewUsername(event.target.value);
            }} placeholder="pitchserver" />
          </Field>
          <Field label="PitchServer password" hint="Used to login to the scoring API. It is not saved by the desktop app.">
            <input type="password" value={pitchServerPassword} onChange={(event) => setPitchServerPassword(event.target.value)} placeholder="Required for PitchServer login" />
          </Field>
          {pitchServerConnected && (
            <>
              <Field label="New PitchServer username" hint="Applied on the server after login.">
                <input value={pitchServerNewUsername} onChange={(event) => setPitchServerNewUsername(event.target.value)} placeholder={pitchServerUsername || "pitchserver"} />
              </Field>
              <Field label="New PitchServer password" hint="At least 8 characters. Existing sessions are revoked.">
                <input type="password" value={pitchServerNewPassword} onChange={(event) => setPitchServerNewPassword(event.target.value)} placeholder="New server password" />
              </Field>
              <Button variant="secondary" onClick={onChangePitchServerCredentials} disabled={busy} icon={<Icon name="check" />}>
                Change PitchServer login
              </Button>
            </>
          )}
        </>
      ) : (
        <Field label="Vast.ai API key" hint={desktopMode ? "Loaded from machine-local runtime.env when saved." : "Available in the desktop app."}>
          <input type="password" value={vastApiKey} onChange={(event) => setVastApiKey(event.target.value)} placeholder="Stored in runtime.env" />
        </Field>
      )}
      <Field label="OpenRouter API key" hint={desktopMode ? "Saved to this PC with Settings / Save." : "Available in the desktop app."}>
        <input type="password" value={openRouterApiKey} onChange={(event) => setOpenRouterApiKey(event.target.value)} placeholder="Saved in runtime.env" />
      </Field>
      <Field label="Evaluator model">
        <input value={openRouterModel} onChange={(event) => setOpenRouterModel(event.target.value)} />
      </Field>
      <Field label="Refiner model">
        <input value={openRouterRefinerModel} onChange={(event) => setOpenRouterRefinerModel(event.target.value)} />
      </Field>
      {runtimeKind !== "pitchserver" && (
        <>
          <Field label="Minimum VRAM">
            <input type="number" min={8} value={minGpuRamGb} onChange={(event) => setMinGpuRamGb(Number(event.target.value))} />
          </Field>
          <Field label="Max hourly">
            <input type="number" min={0.05} step={0.01} value={maxHourlyPrice} onChange={(event) => setMaxHourlyPrice(Number(event.target.value))} />
          </Field>
          <label className="pc-checkbox compact">
            <input type="checkbox" checked={preferInterruptible} onChange={(event) => setPreferInterruptible(event.target.checked)} />
            <span>Prefer interruptible for lower cost</span>
          </label>
        </>
      )}
    </div>
  );
}

function DeployTimeline({ runtimeKind, state, step, offer }: { runtimeKind: RuntimeKind; state: RuntimeState; step: number; offer?: OfferSummary }) {
  const steps = runtimeKind === "pitchserver" ? PITCHSERVER_STEPS : DEPLOY_STEPS;
  return (
    <div className="pc-deploy">
      {steps.map((label, index) => {
        const done = state === "connected" || index < step;
        const active = state === "deploying" && index === step;
        return <DeployStep key={label} index={index} label={label} done={done} active={active} last={index === steps.length - 1} />;
      })}
      <div className="pc-live-log mono">
        <p>{runtimeKind === "pitchserver" ? "$ ssh -p 2022 pitchserver update" : `$ search --verified --max-price ${formatPrice(offer?.dph_total) || "$0.450/hr"}`}</p>
        <p>{runtimeKind === "pitchserver" ? "using Machinity RTX5080 over local SSH tunnel" : offer?.gpu_name ? `selected ${offer.gpu_name} at ${formatPrice(offer.dph_total)}` : "waiting for offer selection"}</p>
        <p>{state === "connected" ? "health check passed" : "container logs streaming..."}</p>
      </div>
    </div>
  );
}

function DeployStep({
  index,
  label,
  done,
  active,
  last,
}: {
  index: number;
  label: string;
  done: boolean;
  active: boolean;
  last: boolean;
}) {
  return (
    <div className="pc-deploy-step">
      <div className="pc-deploy-rail">
        <span className={done ? "done" : active ? "active" : ""} />
        {!last && <i className={done ? "done" : ""} />}
      </div>
      <div>
        <span className="mono">{String(index + 1).padStart(2, "0")}</span>
        <strong>{label}</strong>
        <p>{done ? "complete" : active ? "in progress" : "pending"}</p>
      </div>
    </div>
  );
}

function LocalDiagnostics({ state, status }: { state: RuntimeState; status: DesktopRuntimeStatus | null }) {
  const gpu = status?.local_gpu;
  const on = state === "connected" || state === "scoring";
  return (
    <div className="pc-diagnostics">
      <Spec k="GPU" v={gpu?.available ? gpu.name || gpu.vendor || "available" : "not detected"} />
      <Spec k="Container" v={status?.container_id || "-"} />
      <Spec k="Service" v={status?.service_url || "-"} />
      <TelemetryStrip on={on} />
    </div>
  );
}

function TelemetryStrip({ on }: { on: boolean }) {
  const bars = Array.from({ length: 32 }).map((_, index) => {
    const seed = Math.sin(index * 2.3 + 0.7) * 0.5 + 0.5;
    return on ? 0.2 + seed * 0.8 : 0.06;
  });

  return (
    <div className="pc-telemetry">
      <HeaderLine title="Telemetry" right={on ? "live" : "offline"} />
      <div>
        {bars.map((bar, index) => (
          <span key={index} style={{ height: `${Math.max(2, bar * 100)}%` }} />
        ))}
      </div>
    </div>
  );
}

function EmptyState({ state }: { state: RuntimeState }) {
  const copy: Record<RuntimeState, { title: string; body: string }> = {
    "not-configured": { title: "Setup not complete", body: "Run setup checks before scoring." },
    ready: { title: "No runtime connected", body: "Connect a runtime to begin scoring." },
    connecting: { title: "Connecting runtime", body: "Local service should be ready shortly." },
    deploying: { title: "Deploying to Vast.ai", body: "Open Runtime for live progress." },
    connected: { title: "Ready to calibrate", body: "Pick a medium, tune recipient context, then score." },
    scoring: { title: "Analyzing pitch", body: "TRIBE response prediction in progress." },
    failed: { title: "Runtime error", body: "Open Runtime and check the action log." },
    disconnected: { title: "Runtime disconnected", body: "Reconnect to resume scoring." },
  };
  const item = copy[state];
  return (
    <div className="pc-empty">
      <CalibrationDial spinning={state === "connecting" || state === "deploying"} />
      <strong>{item.title}</strong>
      <p>{item.body}</p>
    </div>
  );
}

function ScoringState() {
  const [phase, setPhase] = useState(0);
  const phases = ["Tokenizing", "Embedding", "Scoring facets", "Aggregating"];

  useEffect(() => {
    const id = window.setInterval(() => setPhase((current) => (current + 1) % phases.length), 850);
    return () => window.clearInterval(id);
  }, [phases.length]);

  return (
    <div className="pc-empty">
      <CalibrationDial spinning />
      <strong>Analyzing pitch</strong>
      <p>{phases[phase]}...</p>
      <Rail indeterminate tone="ok" />
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <div className="pc-error">
      <Icon name="alert" color="var(--err)" size={18} />
      <div>
        <strong>Runtime not responding</strong>
        <span className="mono">RUNTIME_ERROR</span>
        <p>{error}</p>
      </div>
    </div>
  );
}

function CalibrationDial({ spinning }: { spinning?: boolean }) {
  return (
    <div className={spinning ? "pc-dial spinning" : "pc-dial"}>
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="48" stroke="var(--line-strong)" strokeWidth="1" fill="none" />
        <circle cx="60" cy="60" r="38" stroke="var(--line)" strokeWidth="1" fill="none" />
        {Array.from({ length: 24 }).map((_, index) => {
          const angle = (index * 15 * Math.PI) / 180;
          const strong = index % 6 === 0;
          const length = strong ? 8 : 4;
          const x1 = 60 + Math.cos(angle) * 48;
          const y1 = 60 + Math.sin(angle) * 48;
          const x2 = 60 + Math.cos(angle) * (48 - length);
          const y2 = 60 + Math.sin(angle) * (48 - length);
          return <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} stroke={strong ? "var(--fg-muted)" : "var(--fg-faint)"} />;
        })}
      </svg>
      <div>
        <span className="mono">CALIBRATION</span>
        <strong className="mono">-</strong>
      </div>
    </div>
  );
}

function StatusStrip({
  desktopMode,
  state,
  runtimeKind,
  report,
  cost,
  model,
}: {
  desktopMode: boolean;
  state: RuntimeState;
  runtimeKind: RuntimeKind;
  report: PitchScoreReport | null;
  cost?: number;
  model: string;
}) {
  return (
    <footer className="pc-status-strip">
      <StripItem k="RUNTIME" v={!desktopMode ? "web / api" : runtimeKind === "local" ? "local / gpu" : runtimeKind === "pitchserver" ? "pitchserver / ssh" : "vast.ai / remote"} />
      <StripItem k="MODEL" v={shorten(model, 34)} />
      <StripItem k="SCORE" v={report ? `${Math.round(report.persuasion_score)}/100` : "-"} />
      <StripItem k="COST" v={formatPrice(cost)} />
      <span className="pc-actionbar-spacer" />
      <StripItem k="STATE" v={state} />
      <StripItem k="TELEMETRY" v="off" />
    </footer>
  );
}

function RuntimeBadge({ desktopMode, state, runtimeKind }: { desktopMode: boolean; state: RuntimeState; runtimeKind: RuntimeKind }) {
  const map: Record<RuntimeState, { tone: Tone; label: string; pulse?: boolean }> = {
    "not-configured": { tone: "off", label: "NOT CONFIGURED" },
    ready: { tone: "off", label: "READY" },
    connecting: { tone: "warn", label: "CONNECTING", pulse: true },
    deploying: { tone: "warn", label: "DEPLOYING", pulse: true },
    connected: { tone: "ok", label: "CONNECTED" },
    scoring: { tone: "ok", label: "SCORING", pulse: true },
    failed: { tone: "err", label: "FAILED", pulse: true },
    disconnected: { tone: "off", label: "DISCONNECTED" },
  };
  const item = map[state];
  return (
    <div className="pc-runtime-badge nodrag">
      <span>
        <i className={`led ${item.tone} ${item.pulse ? "pulse" : ""}`} />
        <strong className={`mono tone-${item.tone}`}>{item.label}</strong>
      </span>
      <em className="mono">{!desktopMode ? "WEB-API" : runtimeKind === "local" ? "LOCAL-GPU" : runtimeKind === "pitchserver" ? "PITCHSERVER" : "VAST-AI"}</em>
    </div>
  );
}

type Tone = "ok" | "warn" | "err" | "info" | "off";

function StatusPill({ tone = "off", pulse, children }: { tone?: Tone; pulse?: boolean; children: React.ReactNode }) {
  return (
    <span className={`pc-pill tone-${tone}`}>
      <i className={`led ${tone} ${pulse ? "pulse" : ""}`} />
      {children}
    </span>
  );
}

function Button({
  children,
  variant = "secondary",
  loading,
  disabled,
  icon,
  onClick,
}: {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <button type="button" className={`pc-button ${variant}`} disabled={disabled || loading} onClick={onClick}>
      {loading ? <Spinner /> : icon}
      <span>{children}</span>
    </button>
  );
}

function Spinner() {
  return <span className="pc-spinner" />;
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="pc-field">
      <span className="label">{label}</span>
      {children}
      {hint && <small className="mono">{hint}</small>}
    </label>
  );
}

function InlineInput({
  value,
  onChange,
  placeholder,
  strong,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  strong?: boolean;
}) {
  return (
    <input
      className={strong ? "pc-inline-input strong" : "pc-inline-input"}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
    />
  );
}

function Spec({ k, v, tone }: { k: string; v: string; tone?: "warn" | "ok" | "err" }) {
  return (
    <div className="pc-spec">
      <span className="mono">{k}</span>
      <strong className={tone ? `mono tone-${tone}` : "mono"}>{v}</strong>
    </div>
  );
}

function CheckRow({
  title,
  detail,
  status,
  last,
  actionLabel,
  busy,
  onAction,
}: {
  title: string;
  detail: string;
  status: SetupStep["status"];
  last?: boolean;
  actionLabel?: string;
  busy?: boolean;
  onAction?: () => void;
}) {
  const tone = setupTone(status);
  return (
    <div className={last ? "pc-check-row last" : "pc-check-row"}>
      <span className={`pc-check-icon tone-${tone}`}>{status === "ok" ? <Icon name="check" /> : busy ? <Spinner /> : null}</span>
      <div>
        <strong>{title}</strong>
        <p className="mono">{detail}</p>
      </div>
      <span className={`mono pc-check-state tone-${tone}`}>{setupLabel(status)}</span>
      {actionLabel && (
        <Button variant="secondary" loading={busy} disabled={busy} onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
}

function HeaderLine({ title, right }: { title: string; right?: string }) {
  return (
    <div className="pc-header-line">
      <span className="label">{title}</span>
      {right && <span className="mono">{right}</span>}
    </div>
  );
}

function Rail({ value = 0, tone = "ok", indeterminate }: { value?: number; tone?: "ok" | "warn" | "err"; indeterminate?: boolean }) {
  return (
    <div className="pc-rail">
      {indeterminate ? <span className={`scan tone-${tone}`} /> : <span className={`tone-${tone}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />}
    </div>
  );
}

function CornerMarks() {
  return (
    <>
      <i className="pc-corner tl" />
      <i className="pc-corner tr" />
      <i className="pc-corner bl" />
      <i className="pc-corner br" />
    </>
  );
}

function Logo() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <circle cx="9" cy="9" r="7" stroke="var(--fg-muted)" strokeWidth="1" />
      <circle cx="9" cy="9" r="2.5" fill="var(--ok)" />
      <path d="M9 0.5 V4 M9 14 V17.5 M0.5 9 H4 M14 9 H17.5" stroke="var(--fg-muted)" strokeWidth="1" />
    </svg>
  );
}

function PlatformGlyph({ id, active }: { id: Platform; active: boolean }) {
  const color = active ? "var(--ok)" : "var(--fg-dim)";
  const props = { stroke: color, strokeWidth: 1.4, fill: "none", strokeLinecap: "round", strokeLinejoin: "round" } as const;
  switch (id) {
    case "email":
      return <svg width="12" height="12" viewBox="0 0 12 12"><rect {...props} x="1.5" y="2.5" width="9" height="7" rx="1" /><path {...props} d="M1.5 3.5L6 6.5l4.5-3" /></svg>;
    case "linkedin":
      return <svg width="12" height="12" viewBox="0 0 12 12"><rect {...props} x="1.5" y="1.5" width="9" height="9" rx="1" /><circle cx="3.5" cy="4" r="0.6" fill={color} /><path {...props} d="M3.5 5.5V8.5M5.5 5.5V8.5M5.5 7c0-.8.7-1.5 1.5-1.5S8.5 6.2 8.5 7v1.5" /></svg>;
    case "cold-call-script":
      return <svg width="12" height="12" viewBox="0 0 12 12"><path {...props} d="M2.5 3C2.5 2.5 2.8 2 3.5 2h1L5.5 4 4.5 5c.5 1.3 1.2 2 2.5 2.5L8 6.5 10 7.5v1c0 .7-.5 1-1 1C5.4 9.5 2.5 6.6 2.5 3z" /></svg>;
    case "landing-page":
      return <svg width="12" height="12" viewBox="0 0 12 12"><rect {...props} x="1.5" y="1.5" width="9" height="9" rx="1" /><path {...props} d="M1.5 4h9M3 6h4M3 7.5h5" /></svg>;
    case "ad-copy":
      return <svg width="12" height="12" viewBox="0 0 12 12"><path {...props} d="M2 4v4l5 2V2L2 4zM2 4H1.5v4H2" /><path {...props} d="M7 3.5c1.2 0 2 1.1 2 2.5s-.8 2.5-2 2.5" /></svg>;
    default:
      return <svg width="12" height="12" viewBox="0 0 12 12"><path {...props} d="M2 3.5a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1V7a1 1 0 0 1-1 1H5L3 10V8a1 1 0 0 1-1-1V3.5z" /></svg>;
  }
}

function Icon({ name, size = 14, color = "currentColor" }: { name: string; size?: number; color?: string }) {
  const props = { stroke: color, strokeWidth: 1.5, fill: "none", strokeLinecap: "round", strokeLinejoin: "round" } as const;
  switch (name) {
    case "spark":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M7 1.5v3M7 9.5v3M1.5 7h3M9.5 7h3M3.5 3.5l1.5 1.5M9 9l1.5 1.5M3.5 10.5L5 9M9 5l1.5-1.5" /></svg>;
    case "bolt":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M8 1L3 8h3l-1 5 5-7H7l1-5z" /></svg>;
    case "arrow":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M3 7h8M7.5 3.5L11 7l-3.5 3.5" /></svg>;
    case "retry":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M2 7a5 5 0 1 0 1.5-3.5L2 5M2 2v3h3" /></svg>;
    case "cloud":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M4 11h6a2.5 2.5 0 0 0 0-5 3 3 0 0 0-5.6-1A2.5 2.5 0 0 0 4 11z" /></svg>;
    case "alert":
      return <svg width={size} height={size} viewBox="0 0 14 14"><circle {...props} cx="7" cy="7" r="5.5" /><path {...props} d="M7 4v3.5M7 10v.01" /></svg>;
    case "check":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M3 7.5l2.5 2.5L11 4" /></svg>;
    case "chip":
      return <svg width={size} height={size} viewBox="0 0 14 14"><rect {...props} x="3" y="3" width="8" height="8" rx="1" /><path {...props} d="M1 5h2M1 9h2M11 5h2M11 9h2M5 1v2M9 1v2M5 11v2M9 11v2" /></svg>;
    case "plug":
      return <svg width={size} height={size} viewBox="0 0 14 14"><path {...props} d="M5 1v3M9 1v3M4 4h6v3a3 3 0 0 1-6 0V4zM7 10v3" /></svg>;
    default:
      return null;
  }
}

function StripItem({ k, v }: { k: string; v: string }) {
  return (
    <span>
      <b>{k}</b>
      <em>{v}</em>
    </span>
  );
}

function FacetRow({ facet, last }: { facet: { label: string; value: number; note: string }; last?: boolean }) {
  const tone = facet.value >= 75 ? "ok" : facet.value >= 60 ? "warn" : "err";
  return (
    <div className={last ? "pc-facet last" : "pc-facet"}>
      <div>
        <i className={`led ${tone}`} />
        <strong>{facet.label}</strong>
        <span className={`mono score-${tone}`}>{facet.value}</span>
      </div>
      <div className="pc-microbar">
        <span style={{ width: `${facet.value}%`, background: `var(--${tone})` }} />
      </div>
      <p>{facet.note}</p>
    </div>
  );
}

function readError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function routeFromHash(hash: string): Route {
  const route = hash.replace(/^#/, "").trim().toLowerCase();
  return ROUTES.includes(route as Route) ? (route as Route) : "workspace";
}

function runtimeKindFromMode(mode: DesktopRuntimeStatus["mode"]): RuntimeKind {
  return mode === "vast" || mode === "pitchserver" ? mode : "local";
}

function applyDesktopConfig(
  config: DesktopAppConfig,
  setters: {
    setVastApiKey: (value: string) => void;
    setOpenRouterApiKey: (value: string) => void;
    setOpenRouterModel: (value: string) => void;
    setOpenRouterRefinerModel: (value: string) => void;
    setImage: (value: string) => void;
    setMinGpuRamGb: (value: number) => void;
    setMaxHourlyPrice: (value: number) => void;
    setPreferInterruptible: (value: boolean) => void;
    setConfigPath: (value: string | undefined) => void;
  },
) {
  setters.setVastApiKey(config.vastApiKey || "");
  setters.setOpenRouterApiKey(config.openRouterApiKey || "");
  setters.setOpenRouterModel(config.openRouterModel || DEFAULT_OPENROUTER_MODEL);
  setters.setOpenRouterRefinerModel(config.openRouterRefinerModel || config.openRouterModel || DEFAULT_OPENROUTER_MODEL);
  setters.setImage(config.image || DEFAULT_IMAGE);
  setters.setMinGpuRamGb(config.minGpuRamGb ?? 16);
  setters.setMaxHourlyPrice(config.maxHourlyPrice ?? 0.45);
  setters.setPreferInterruptible(config.preferInterruptible ?? true);
  setters.setConfigPath(config.configPath);
}

function formatPersona(audience: Audience) {
  return [audience.name, audience.role, audience.relationship, audience.context].filter(Boolean).join(". ");
}

function initialsFor(name: string) {
  return (
    name
      .trim()
      .split(/\s+/)
      .map((part) => part[0] || "")
      .slice(0, 2)
      .join("")
      .toUpperCase() || "?"
  );
}

function formatPrice(value?: number) {
  return typeof value === "number" ? `$${value.toFixed(3)}/hr` : "-";
}

function shorten(value: string, max: number) {
  return value.length > max ? `${value.slice(0, max - 1)}...` : value;
}

function capitalize(value: string) {
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

function setupTone(status: SetupStep["status"]): Tone {
  if (status === "ok") return "ok";
  if (status === "missing") return "err";
  if (status === "blocked" || status === "warning") return "warn";
  return "off";
}

function setupLabel(status: SetupStep["status"]) {
  return {
    ok: "ready",
    missing: "install",
    blocked: "blocked",
    optional: "optional",
    warning: "check",
    action: "run",
  }[status];
}

function groupSetupSteps(steps: SetupStep[]) {
  const groups: Record<string, SetupStep[]> = {
    "Local runtime": [],
    Fallback: [],
    System: [],
  };

  for (const step of steps) {
    if (/vast/i.test(step.key) || /vast/i.test(step.title)) groups.Fallback.push(step);
    else if (/release|update/i.test(step.key) || /release|update/i.test(step.title)) groups.System.push(step);
    else groups["Local runtime"].push(step);
  }

  return Object.fromEntries(Object.entries(groups).filter(([, items]) => items.length)) as Record<string, SetupStep[]>;
}

function extractSuggestions(report: PitchScoreReport) {
  return [
    ...report.rewrite_suggestions.map((item) => `${item.title}: ${item.why || item.after}`),
    ...report.risks.map((risk) => `Risk: ${risk}`),
  ]
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 5);
}

function extractRefineBrief(report: PitchScoreReport) {
  const rewriteGuidance = report.rewrite_suggestions
    .map((item) => {
      const before = item.before ? ` Replace or improve "${item.before}"` : "";
      const after = item.after ? ` Target direction: "${item.after}"` : "";
      return `Rewrite suggestion - ${item.title}: ${item.why || "Apply this directly."}${before}${after}`;
    })
    .filter(Boolean);
  const weakFacets = report.breakdown
    .filter((item) => item.score < 78)
    .sort((left, right) => left.score - right.score)
    .slice(0, 3)
    .map((item) => `Weak facet - ${item.label} (${Math.round(item.score)}/100): ${item.explanation} Repair tactic: ${facetRepairTactic(item.key, item.label)}`);
  const weakSignals = report.neural_signals
    .filter((item) => (item.key === "cognitive_friction" ? item.score > 38 : item.score < 70))
    .sort((left, right) => {
      const leftWeakness = left.key === "cognitive_friction" ? left.score : 100 - left.score;
      const rightWeakness = right.key === "cognitive_friction" ? right.score : 100 - right.score;
      return rightWeakness - leftWeakness;
    })
    .slice(0, 3)
    .map((item) => `Weak neural signal - ${item.label} (${Math.round(item.score)}/100): ${neuralRepairTactic(item.key)}`);
  const evidence = report.persuasion_evidence;
  const missingElements = (evidence?.missing_elements ?? [])
    .slice(0, 3)
    .map((item) => `Missing persuasion element: ${item}. Add it without inventing unverifiable facts.`);
  const risks = report.risks.slice(0, 3).map((risk) => `Risk to fix: ${risk}`);

  return [
    ...rewriteGuidance,
    ...weakFacets,
    ...weakSignals,
    ...missingElements,
    ...risks,
  ]
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 12);
}

function facetRepairTactic(key: string, label: string) {
  const normalized = `${key} ${label}`.toLowerCase();
  if (/clarity|friction/.test(normalized)) {
    return "Shorten long sentences, remove vague words, make the value proposition instantly scannable, and keep one clear next step.";
  }
  if (/credibility|proof|trust/.test(normalized)) {
    return "Add concrete proof already supported by the draft: customer type, measurable outcome, benchmark, implementation detail, or a low-risk pilot proof path.";
  }
  if (/personal|fit|relevance/.test(normalized)) {
    return "Tie the opener and value claim to the recipient's role, current context, pain, and decision criteria.";
  }
  if (/urgency|attention/.test(normalized)) {
    return "Make the cost of waiting or timing reason explicit, then connect it to a specific CTA.";
  }
  if (/emotion|resonance|reward/.test(normalized)) {
    return "Translate the feature into an outcome the recipient wants, such as saved time, less risk, status, confidence, or reduced workload.";
  }
  return "Make the weak point more concrete, more persona-specific, and easier to act on.";
}

function neuralRepairTactic(key: string) {
  if (key === "cognitive_friction") {
    return "Reduce cognitive load: fewer clauses, simpler structure, one ask, and no generic jargon.";
  }
  if (key === "personal_relevance") {
    return "Increase self-relevance: mention the persona's situation and show why this matters to them now.";
  }
  if (key === "emotional_engagement") {
    return "Increase reward value: state a tangible win or avoided pain in human terms.";
  }
  if (key === "social_proof_potential") {
    return "Add credible social proof only if supported: peer teams, customer category, reference, benchmark, or proof path.";
  }
  if (key === "memorability") {
    return "Add a compact, specific phrase or contrast that is easy to remember.";
  }
  if (key === "attention_capture") {
    return "Strengthen the first line with a relevant trigger, specific observation, or sharp problem statement.";
  }
  return "Improve this signal with concrete, audience-specific evidence.";
}

function buildPreviewRewrite(source: string, suggestions: string[]) {
  const paragraphs = source
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
  const hasMetricSuggestion = suggestions.some((item) => /metric|proof|credibility|outcome/i.test(item));
  const hasCtaSuggestion = suggestions.some((item) => /cta|time|reply|friction|schedule/i.test(item));

  if (hasMetricSuggestion && paragraphs.length > 1 && !/\b\d+[%x]?\b/.test(paragraphs[1])) {
    paragraphs[1] = `${paragraphs[1].replace(/[.!?]$/, "")}, with the outcome framed as time saved and fewer manual dashboard reviews.`;
  }

  if (hasCtaSuggestion && paragraphs.length) {
    const lastIndex = paragraphs.length - 1;
    paragraphs[lastIndex] = paragraphs[lastIndex]
      .replace(/Would you be open to a 15-min call next Tuesday or Wednesday\?/i, "Could you do 15 minutes Tuesday 2pm ET or Wednesday 11am ET?")
      .replace(/Would you be open to/i, "Could you do");
  }

  return paragraphs.join("\n\n") || source;
}

function toneFromScore(score: number): Tone {
  if (score >= 75) return "ok";
  if (score >= 60) return "warn";
  return "err";
}

function confidenceFromScore(score: number) {
  return Math.max(0.55, Math.min(0.96, 0.62 + score / 300));
}

function fallbackFmri(score: number): FmriOutput {
  const base = Math.max(0.08, score / 500);
  const trace = [0.86, 1, 0.9, 0.74, 0.66, 0.71, 0.82, 0.88, 0.81, 0.69, 0.6, 0.56].map((value) => value * base);
  return {
    segments: 12,
    voxel_count: 60784,
    global_mean_abs: base,
    global_peak_abs: base * 7.8,
    temporal_trace: trace,
    temporal_peaks: trace.map((value, index) => value * (4.6 + (index % 3) * 0.3)),
    top_voxel_indices: [8421, 12044, 31207, 40918, 52114, 58002],
    top_voxel_values: [0.284, 0.241, 0.218, 0.204, 0.189, 0.176].map((value) => value * (score / 75)),
  };
}

function fallbackSignals(score: number): PitchScoreReport["neural_signals"] {
  return [
    ["emotional_engagement", "Affective value salience", score - 2],
    ["personal_relevance", "Self-value relevance", score + 4],
    ["social_proof_potential", "Social cognition / sharing", score - 9],
    ["memorability", "Encoding potential", score - 5],
    ["attention_capture", "Early attention salience", score + 1],
    ["cognitive_friction", "Cognitive friction", Math.max(18, 100 - score)],
  ].map(([key, label, value]) => ({
    key: String(key),
    label: String(label),
    score: Math.max(0, Math.min(100, Number(value))),
    direction: "neutral" as const,
  }));
}
