import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform as NativePlatform,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import * as Clipboard from "expo-clipboard";
import Constants from "expo-constants";
import { SafeAreaView } from "react-native-safe-area-context";
import { isProbablyHttpUrl } from "./src/network";
import { defaultSettings, loadSettings, probeRuntime, saveSettings, scorePitch } from "./src/runtime";
import { clearPendingDraft, loadPendingDraft, savePendingDraft } from "./src/draft-queue";
import { getRuntimeEvents } from "./src/telemetry";
import { platformValues, RuntimeKind, RuntimeProbe, RuntimeSettings, type PitchScoreReport, type Platform, type TransportMode } from "./src/types";
import { theme } from "./src/theme";

type HistoryItem = { at: string; score: number; verdict: string };
type PendingScoreRequest = { message: string; persona: string; platform: Platform };

const platformLabels: Record<Platform, string> = {
  email: "Email",
  linkedin: "LinkedIn",
  "cold-call-script": "Cold Call",
  "landing-page": "Landing Page",
  "ad-copy": "Ad Copy",
  general: "General",
};

const transportLabels: Record<TransportMode, string> = {
  auto: "Auto",
  "next-api": "Next API",
  direct: "Direct",
};

export default function App() {
  const [settings, setSettings] = useState<RuntimeSettings>(defaultSettings);
  const [message, setMessage] = useState("");
  const [persona, setPersona] = useState("");
  const [platform, setPlatform] = useState<Platform>("email");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<RuntimeProbe | null>(null);
  const [probing, setProbing] = useState(false);
  const [report, setReport] = useState<PitchScoreReport | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [queuedRequest, setQueuedRequest] = useState<PendingScoreRequest | null>(null);
  const queuedRequestRef = useRef<PendingScoreRequest | null>(null);
  const strictTransportRequired = Boolean((Constants.expoConfig?.extra as { strictTransportRequired?: boolean } | undefined)?.strictTransportRequired);

  useEffect(() => {
    loadSettings()
      .then((loaded) => {
        const next = strictTransportRequired ? { ...loaded, strictTransportSecurity: true } : loaded;
        setSettings(next);
      })
      .catch(() => undefined);

    loadPendingDraft()
      .then((draft) => {
        if (!draft) return;
        setQueuedRequest({ message: draft.message, persona: draft.persona, platform: draft.platform });
      })
      .catch(() => undefined);
  }, [strictTransportRequired]);


  const runtimeLabel = settings.runtime === "pitchserver" ? "PitchServer" : "Vast AI";
  const activeRuntimeUrl = settings.runtime === "pitchserver" ? settings.pitchserverUrl : settings.vastUrl;
  const insecureHttpWarning =
    NativePlatform.OS === "ios" &&
    settings.strictTransportSecurity &&
    activeRuntimeUrl.startsWith("http://") &&
    !activeRuntimeUrl.includes("127.0.0.1") &&
    !activeRuntimeUrl.includes("localhost");

  const scoreColor = useMemo(() => {
    const score = report?.persuasion_score ?? 0;
    if (score >= 75) return theme.ok;
    if (score >= 50) return theme.warn;
    return theme.err;
  }, [report]);

  const canScore = message.trim().length >= 10 && persona.trim().length >= 5;

  async function patchSettings(patch: Partial<RuntimeSettings>) {
    const next = { ...settings, ...patch };
    if (strictTransportRequired) next.strictTransportSecurity = true;
    setSettings(next);
    setProbe(null);
    await saveSettings(next);
  }

  async function onScore(override?: PendingScoreRequest) {
    const request = override ?? { message: message.trim(), persona: persona.trim(), platform };

    if (request.message.length < 10 || request.persona.length < 5) {
      setError("Pitch must be at least 10 chars and persona at least 5 chars.");
      return;
    }

    if (loading && !override) {
      queuedRequestRef.current = request;
      setQueuedRequest(request);
      void savePendingDraft({ ...request, queuedAt: new Date().toISOString() });
      setError("Current scoring continues. Your latest draft is queued.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const next = await scorePitch(settings, request.message, request.persona, request.platform);
      setReport(next);
      setHistory((prev) => [{ at: next.scored_at, score: next.persuasion_score, verdict: next.verdict }, ...prev].slice(0, 5));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unexpected error.");
    } finally {
      setLoading(false);
      if (!override && queuedRequestRef.current) {
        const nextRequest = queuedRequestRef.current;
        queuedRequestRef.current = null;
        setQueuedRequest(null);
        void clearPendingDraft();
        void onScore(nextRequest);
      }
    }
  }

  async function runProbe() {
    setProbing(true);
    setError(null);
    try {
      setProbe(await probeRuntime(settings));
    } catch (caught) {
      setProbe(null);
      setError(caught instanceof Error ? caught.message : "Runtime check failed.");
    } finally {
      setProbing(false);
    }
  }


  async function exportTelemetry() {
    const events = getRuntimeEvents();
    await Clipboard.setStringAsync(JSON.stringify(events, null, 2));
    setError(`Telemetry copied (${events.length} events).`);
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" />
      <KeyboardAvoidingView style={styles.root} behavior={NativePlatform.OS === "ios" ? "padding" : undefined}>
        <ScrollView style={styles.root} contentContainerStyle={styles.content}>
          <LinearGradient colors={["#17191a", "#121311"]} style={styles.header}>
            <Text style={styles.kicker}>PITCHCHECK · MOBILE</Text>
            <Text style={styles.title}>Neural Persuasion Studio</Text>
            <Text style={styles.subtitle}>Premium iOS-first experience with production runtime controls for PitchServer and Vast AI.</Text>
          </LinearGradient>

          <View style={styles.card}>
            <View style={styles.inlineHead}>
              <Text style={styles.sectionTitle}>Runtime</Text>
              <Text style={styles.runtimePill}>{runtimeLabel}</Text>
            </View>

            <View style={styles.segmented}>
              <SegmentButton label="PitchServer" active={settings.runtime === "pitchserver"} onPress={() => void patchSettings({ runtime: "pitchserver" as RuntimeKind })} />
              <SegmentButton label="Vast AI" active={settings.runtime === "vast"} onPress={() => void patchSettings({ runtime: "vast" as RuntimeKind })} />
            </View>

            {settings.runtime === "pitchserver" ? (
              <Field label="PitchServer URL" value={settings.pitchserverUrl} placeholder="http://127.0.0.1:8090" onChange={(v) => void patchSettings({ pitchserverUrl: v })} />
            ) : (
              <>
                <Field label="Vast Runtime URL" value={settings.vastUrl} placeholder="https://your-vast-instance" onChange={(v) => void patchSettings({ vastUrl: v })} />
                <Field label="Vast API Key (optional)" value={settings.vastApiKey} placeholder="Bearer key" secure onChange={(v) => void patchSettings({ vastApiKey: v })} />
              </>
            )}

            {!isProbablyHttpUrl(activeRuntimeUrl) && activeRuntimeUrl.length > 0 && <Text style={styles.warningText}>Runtime URL should be a valid http(s) URL.</Text>}
            {insecureHttpWarning && <Text style={styles.warningText}>Strict mode is enabled: use HTTPS for non-local iOS production runtimes.</Text>}
            {strictTransportRequired && <Text style={styles.warningText}>Release profile enforces strict HTTPS policy.</Text>}

            <View style={styles.inlineToggleRow}>
              <Text style={styles.toggleLabel}>Strict HTTPS policy</Text>
              <Pressable
                style={[styles.togglePill, settings.strictTransportSecurity && styles.togglePillActive, strictTransportRequired && styles.toggleDisabled]}
                onPress={() => void patchSettings({ strictTransportSecurity: !settings.strictTransportSecurity })}
                disabled={strictTransportRequired}
              >
                <Text style={[styles.togglePillText, settings.strictTransportSecurity && styles.togglePillTextActive]}>{settings.strictTransportSecurity ? "ON" : "OFF"}</Text>
              </Pressable>
            </View>

            <Field label="OpenRouter Model" value={settings.openRouterModel} placeholder="anthropic/claude-sonnet-4.6" onChange={(v) => void patchSettings({ openRouterModel: v })} />

            <View style={styles.segmented}>
              {(Object.keys(transportLabels) as TransportMode[]).map((mode) => (
                <SegmentButton key={mode} label={transportLabels[mode]} active={settings.transportMode === mode} onPress={() => void patchSettings({ transportMode: mode })} />
              ))}
            </View>

            <Pressable style={styles.secondaryButton} onPress={() => void runProbe()} disabled={probing}>
              <Text style={styles.secondaryButtonText}>{probing ? "Checking runtime…" : "Check Runtime"}</Text>
            </Pressable>
            {probe && <Text style={[styles.probeText, { color: probe.ok ? theme.ok : theme.warn }]}>{probe.detail} {probe.endpointTried ? `(${probe.endpointTried})` : ""}</Text>}

            <Pressable style={styles.secondaryButton} onPress={() => void exportTelemetry()}>
              <Text style={styles.secondaryButtonText}>Export Runtime Logs</Text>
            </Pressable>
          </View>

          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Pitch Analyzer</Text>
            <Field label="Persona" value={persona} placeholder="Staff engineer, pragmatic, detail-oriented" multiline onChange={setPersona} />
            <Field label="Pitch" value={message} placeholder="Paste your pitch here..." multiline tall onChange={setMessage} />

            <View style={styles.platformRow}>
              {platformValues.map((item) => (
                <Pressable key={item} style={[styles.chip, platform === item && styles.chipActive]} onPress={() => setPlatform(item)}>
                  <Text style={[styles.chipText, platform === item && styles.chipTextActive]}>{platformLabels[item]}</Text>
                </Pressable>
              ))}
            </View>

            <Pressable style={styles.scoreButton} onPress={() => void onScore()} disabled={loading || !canScore}>
              <LinearGradient colors={["#44d59f", "#2ea77a"]} style={[styles.gradientBtn, (!canScore || loading) && { opacity: 0.55 }]}>
                {loading ? <ActivityIndicator color="#031a12" /> : <Text style={styles.scoreText}>Score My Pitch</Text>}
              </LinearGradient>
            </Pressable>
            {error ? <Text style={styles.error}>{error}</Text> : null}
            {queuedRequest ? <Text style={styles.warningText}>Queued draft will run after current request.</Text> : null}
          </View>

          {report && (
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Result</Text>
              <View style={styles.scoreRow}>
                <Text style={[styles.bigScore, { color: scoreColor }]}>{report.persuasion_score}</Text>
                <View style={{ flex: 1 }}>
                  <Text style={styles.verdict}>{report.verdict}</Text>
                  <Text style={styles.narrative}>{report.narrative}</Text>
                  <Text style={styles.timestamp}>{new Date(report.scored_at).toLocaleString()}</Text>
                </View>
              </View>

              <List title="Strengths" items={report.strengths} />
              <List title="Risks" items={report.risks} />

              <Pressable style={styles.secondaryButton} onPress={() => void Clipboard.setStringAsync(JSON.stringify(report, null, 2))}>
                <Text style={styles.secondaryButtonText}>Copy JSON</Text>
              </Pressable>
            </View>
          )}

          {history.length > 0 && (
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Recent Scores</Text>
              {history.map((item, index) => (
                <View key={`${item.at}-${index}`} style={styles.historyRow}>
                  <Text style={styles.historyScore}>{item.score}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.historyVerdict} numberOfLines={1}>{item.verdict}</Text>
                    <Text style={styles.timestamp}>{new Date(item.at).toLocaleString()}</Text>
                  </View>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function SegmentButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable style={[styles.segmentButton, active && styles.segmentButtonActive]} onPress={onPress}>
      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{label}</Text>
    </Pressable>
  );
}

function Field({ label, value, placeholder, onChange, multiline, secure, tall }: { label: string; value: string; placeholder: string; onChange: (next: string) => void; multiline?: boolean; secure?: boolean; tall?: boolean }) {
  return (
    <View style={{ marginBottom: 12 }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        style={[styles.input, multiline && styles.multiline, tall && styles.tall]}
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor={theme.fgDim}
        multiline={multiline}
        secureTextEntry={secure}
      />
    </View>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <View style={{ marginTop: 12 }}>
      <Text style={styles.listTitle}>{title}</Text>
      {items.slice(0, 3).map((item, idx) => (
        <Text key={`${title}-${idx}`} style={styles.listItem}>{`• ${item}`}</Text>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.bgApp },
  root: { flex: 1 },
  content: { padding: 16, paddingBottom: 40, gap: 14 },
  header: { borderWidth: 1, borderColor: theme.line, borderRadius: 18, padding: 16 },
  kicker: { color: theme.fgDim, fontSize: 11, letterSpacing: 1.2, marginBottom: 8 },
  title: { color: theme.fg, fontSize: 27, fontWeight: "700", marginBottom: 6 },
  subtitle: { color: theme.fgMuted, fontSize: 13, lineHeight: 18 },
  card: { backgroundColor: theme.bgPanel, borderRadius: 16, borderWidth: 1, borderColor: theme.line, padding: 14 },
  inlineHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  runtimePill: { backgroundColor: theme.bgElevated, borderColor: theme.lineStrong, borderWidth: 1, color: theme.fgMuted, borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4, fontSize: 11 },
  sectionTitle: { color: theme.fg, fontWeight: "700", fontSize: 16, marginBottom: 10 },
  label: { color: theme.fgMuted, fontSize: 12, marginBottom: 6 },
  input: { borderColor: theme.lineStrong, borderWidth: 1, borderRadius: 12, backgroundColor: theme.bgElevated, color: theme.fg, paddingHorizontal: 12, paddingVertical: 10, fontSize: 14 },
  multiline: { minHeight: 82, textAlignVertical: "top" },
  tall: { minHeight: 140 },
  segmented: { flexDirection: "row", gap: 8, marginBottom: 10 },
  segmentButton: { flex: 1, borderColor: theme.lineStrong, borderWidth: 1, borderRadius: 11, paddingVertical: 9, alignItems: "center", backgroundColor: theme.bgElevated },
  segmentButtonActive: { backgroundColor: "rgba(106,231,178,0.16)", borderColor: "rgba(106,231,178,0.52)" },
  segmentText: { color: theme.fgMuted, fontSize: 12, fontWeight: "600" },
  segmentTextActive: { color: theme.ok },
  secondaryButton: { borderColor: theme.lineStrong, borderWidth: 1, borderRadius: 10, paddingVertical: 10, alignItems: "center", marginTop: 2 },
  secondaryButtonText: { color: theme.fgMuted, fontWeight: "600" },
  probeText: { fontSize: 12, marginTop: 8, lineHeight: 16 },
  warningText: { color: theme.warn, fontSize: 12, marginTop: 2, marginBottom: 8 },
  inlineToggleRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 12 },
  toggleLabel: { color: theme.fgMuted, fontSize: 12 },
  togglePill: { borderWidth: 1, borderColor: theme.lineStrong, borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4, backgroundColor: theme.bgElevated },
  togglePillActive: { borderColor: "rgba(106,231,178,0.52)", backgroundColor: "rgba(106,231,178,0.16)" },
  togglePillText: { color: theme.fgMuted, fontSize: 11, fontWeight: "700" },
  togglePillTextActive: { color: theme.ok },
  toggleDisabled: { opacity: 0.5 },
  platformRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginVertical: 10 },
  chip: { borderColor: theme.lineStrong, borderWidth: 1, borderRadius: 999, paddingHorizontal: 11, paddingVertical: 7, backgroundColor: theme.bgElevated },
  chipActive: { backgroundColor: "rgba(106,231,178,0.16)", borderColor: "rgba(106,231,178,0.52)" },
  chipText: { color: theme.fgMuted, fontSize: 12 },
  chipTextActive: { color: theme.ok },
  scoreButton: { marginTop: 6, borderRadius: 12, overflow: "hidden" },
  gradientBtn: { paddingVertical: 12, alignItems: "center" },
  scoreText: { color: "#031a12", fontSize: 14, fontWeight: "800" },
  error: { color: theme.err, fontSize: 12, marginTop: 8 },
  scoreRow: { flexDirection: "row", gap: 14, alignItems: "center" },
  bigScore: { fontSize: 44, fontWeight: "800", width: 72 },
  verdict: { color: theme.fg, fontSize: 16, fontWeight: "700", marginBottom: 4 },
  narrative: { color: theme.fgMuted, lineHeight: 18, fontSize: 13 },
  timestamp: { color: theme.fgDim, fontSize: 11, marginTop: 6 },
  listTitle: { color: theme.fg, fontWeight: "600", marginBottom: 4 },
  listItem: { color: theme.fgMuted, fontSize: 13, lineHeight: 18 },
  historyRow: { flexDirection: "row", gap: 10, alignItems: "center", paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: theme.line },
  historyScore: { width: 44, textAlign: "center", color: theme.ok, fontSize: 20, fontWeight: "800" },
  historyVerdict: { color: theme.fg, fontSize: 13, fontWeight: "600" },
});
