export type RuntimeMode = "disconnected" | "local" | "vast" | "pitchserver";

export interface LocalGpuInfo {
  available: boolean;
  vendor?: string;
  name?: string;
  memory_mb?: number;
  detail?: string;
}

export interface OfferSummary {
  id: number;
  gpu_name?: string;
  num_gpus?: number;
  gpu_ram_gb?: number;
  dph_total?: number;
  reliability?: number;
}

export interface DesktopRuntimeStatus {
  mode: RuntimeMode;
  connected: boolean;
  service_url?: string;
  local_gpu: LocalGpuInfo;
  container_id?: string;
  vast_instance_id?: number;
  offer?: OfferSummary;
  image: string;
  last_error?: string;
}

export interface DesktopRuntimeConfig {
  runtimeKind?: Exclude<RuntimeMode, "disconnected">;
  vastApiKey?: string;
  pitchServerSshPassword?: string;
  pitchServerUsername?: string;
  pitchServerPassword?: string;
  openRouterApiKey?: string;
  openRouterModel?: string;
  openRouterRefinerModel?: string;
  image?: string;
  minGpuRamGb?: number;
  maxHourlyPrice?: number;
  preferInterruptible?: boolean;
}

export interface DesktopAppConfig extends DesktopRuntimeConfig {
  configPath?: string;
}

export interface SetupStep {
  key: string;
  title: string;
  status: "ok" | "missing" | "blocked" | "optional" | "warning" | "action";
  detail: string;
  action_label?: string;
  url?: string;
}

export interface SetupStatus {
  platform: string;
  ready_for_local: boolean;
  ready_for_cloud: boolean;
  steps: SetupStep[];
}

export interface SetupActionResult {
  ok: boolean;
  message: string;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export function isDesktopRuntime(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.__TAURI_INTERNALS__ !== "undefined"
  );
}

async function invokeDesktop<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export async function getDesktopRuntimeStatus(): Promise<DesktopRuntimeStatus> {
  return invokeDesktop<DesktopRuntimeStatus>("get_runtime_status");
}

export async function connectDesktopRuntime(
  config: DesktopRuntimeConfig,
): Promise<DesktopRuntimeStatus> {
  return invokeDesktop<DesktopRuntimeStatus>("connect_runtime", { config });
}

export async function disconnectDesktopRuntime(
  vastApiKey?: string,
): Promise<DesktopRuntimeStatus> {
  return invokeDesktop<DesktopRuntimeStatus>("disconnect_runtime", {
    vastApiKey,
  });
}

export async function getSetupStatus(image?: string): Promise<SetupStatus> {
  return invokeDesktop<SetupStatus>("get_setup_status", { image });
}

export async function runSetupStep(
  key: string,
  image?: string,
): Promise<SetupActionResult> {
  return invokeDesktop<SetupActionResult>("run_setup_step", { key, image });
}

export async function getDesktopAppConfig(): Promise<DesktopAppConfig> {
  return invokeDesktop<DesktopAppConfig>("get_app_config");
}

export async function saveDesktopAppConfig(
  config: DesktopAppConfig,
): Promise<DesktopAppConfig> {
  return invokeDesktop<DesktopAppConfig>("save_app_config", { config });
}

export async function scorePitchOnDesktop(
  request: {
    message: string;
    persona: string;
    platform: string;
    openRouterModel?: string;
  },
): Promise<{ report?: unknown; error?: string }> {
  return invokeDesktop<{ report?: unknown; error?: string }>("score_pitch", {
    request,
  });
}

export async function changePitchServerCredentials(request: {
  currentPassword: string;
  newUsername: string;
  newPassword: string;
}): Promise<{ ok?: boolean; username?: string; error?: string }> {
  return invokeDesktop<{ ok?: boolean; username?: string; error?: string }>(
    "change_pitch_server_credentials",
    { request },
  );
}

export async function refinePitchOnDesktop(request: {
  message: string;
  persona: string;
  platform: string;
  suggestions: string[];
  clarificationAnswers?: Array<{
    id: string;
    question: string;
    answer: string;
  }>;
}): Promise<{
  refinedMessage?: string;
  model?: string;
  methodology?: string;
  needsClarification?: boolean;
  questions?: Array<
    | string
    | {
        id: string;
        label: string;
        question: string;
        why?: string;
      }
  >;
  safetyNotes?: string[];
  persuasionProfile?: unknown;
  baselineScore?: number;
  bestScore?: number;
  improvement?: number;
  selectedIndex?: number;
  error?: string;
}> {
  return invokeDesktop<{
    refinedMessage?: string;
    model?: string;
    methodology?: string;
    needsClarification?: boolean;
    questions?: Array<
      | string
      | {
          id: string;
          label: string;
          question: string;
          why?: string;
        }
    >;
    safetyNotes?: string[];
    persuasionProfile?: unknown;
    baselineScore?: number;
    bestScore?: number;
    improvement?: number;
    selectedIndex?: number;
    error?: string;
  }>("refine_pitch", { request });
}

export async function checkDesktopTribeHealth(): Promise<unknown> {
  return invokeDesktop<unknown>("check_runtime_health");
}
