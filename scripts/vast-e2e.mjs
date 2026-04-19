const API_BASE = "https://console.vast.ai/api/v0";
const TRIBE_PORT = 8090;
const BOOTSTRAP_IMAGE =
  process.env.VAST_BOOTSTRAP_IMAGE ||
  "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel";

const apiKey = process.env.VAST_API_KEY?.trim();
if (!apiKey) {
  throw new Error("Set VAST_API_KEY before running this script.");
}

const maxHourlyPrice = Number(process.env.VAST_MAX_HOURLY_PRICE || "0.12");
const minGpuRamGb = Number(process.env.VAST_MIN_GPU_RAM_GB || "24");
const keepInstance = process.env.KEEP_VAST_INSTANCE === "1";
const existingInstanceId = process.env.VAST_INSTANCE_ID
  ? Number(process.env.VAST_INSTANCE_ID)
  : undefined;

function headers() {
  return {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
}

async function vastFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Vast ${path} failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body;
}

function parseOffers(value) {
  const raw = Array.isArray(value.offers)
    ? value.offers
    : value.offers && typeof value.offers === "object"
      ? Object.values(value.offers)
      : [];
  return raw
    .map((offer) => ({
      id: offer.id ?? offer.ask_contract_id,
      gpu_name: offer.gpu_name,
      num_gpus: offer.num_gpus,
      gpu_ram_gb:
        typeof offer.gpu_ram === "number" ? offer.gpu_ram / 1024 : undefined,
      dph_total: offer.dph_total,
      reliability: offer.reliability,
    }))
    .filter((offer) => offer.id && offer.dph_total <= maxHourlyPrice)
    .sort(
      (a, b) =>
        (a.dph_total ?? Number.POSITIVE_INFINITY) -
          (b.dph_total ?? Number.POSITIVE_INFINITY) ||
        (b.reliability ?? 0) - (a.reliability ?? 0),
    );
}

function bootstrapOnstart() {
  const script = `#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git ffmpeg build-essential libsndfile1 curl ca-certificates
python -m pip install --upgrade pip setuptools wheel
python -m pip install uv==0.11.2
git clone --depth 1 https://github.com/aytzey/PitchCheck.git /workspace/PitchCheck
cd /workspace/PitchCheck
python -m pip install -r tribe_service/requirements.txt
python -m pip install --upgrade git+https://github.com/huggingface/transformers.git@9914a3641f7aaaabb0bcdfcd73a54a1cfa70c3e7
python -m spacy download en_core_web_lg
python -m pip install --no-deps git+https://github.com/facebookresearch/tribev2.git@main
python tribe_service/patch_tribev2_whisperx.py || true
mkdir -p /models /models/huggingface/hub /models/.cache
export PYTHONUNBUFFERED=1
export TRIBE_MODEL_ID=facebook/tribev2
export TRIBE_DEVICE=cuda
export TRIBE_CACHE_DIR=/models
export TRIBE_ALLOW_MOCK=0
export HF_HOME=/models/huggingface
export HUGGINGFACE_HUB_CACHE=/models/huggingface/hub
export XDG_CACHE_HOME=/models/.cache
export TRIBE_TEXT_MODEL=NousResearch/Hermes-3-Llama-3.2-3B
exec uvicorn tribe_service.app:app --host 0.0.0.0 --port ${TRIBE_PORT}
`;
  return `cat >/root/pitchcheck_bootstrap.sh <<'PITCHCHECK_BOOTSTRAP'
${script}
PITCHCHECK_BOOTSTRAP
chmod +x /root/pitchcheck_bootstrap.sh
nohup /bin/bash /root/pitchcheck_bootstrap.sh > /tmp/pitchcheck-bootstrap.log 2>&1 &`;
}

async function createInstance(offer) {
  const env = {
    TRIBE_DEVICE: "cuda",
    TRIBE_ALLOW_MOCK: "0",
    TRIBE_CACHE_DIR: "/models",
    OPEN_BUTTON_PORT: String(TRIBE_PORT),
    [`-p ${TRIBE_PORT}:${TRIBE_PORT}`]: "1",
  };
  const body = {
    image: BOOTSTRAP_IMAGE,
    label: "pitchcheck-tribe-e2e",
    disk: 120,
    runtype: "ssh_direct",
    target_state: "running",
    cancel_unavail: true,
    env,
    onstart: bootstrapOnstart(),
  };
  if (offer.dph_total) {
    body.price = Math.ceil(offer.dph_total * 1.03 * 1000) / 1000;
  }
  const response = await vastFetch(`/asks/${offer.id}/`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
  const instanceId = response.new_contract ?? response.id;
  if (!instanceId) {
    throw new Error(`Vast did not return an instance id: ${JSON.stringify(response)}`);
  }
  return instanceId;
}

async function startInstance(instanceId) {
  const response = await fetch(`${API_BASE}/instances/${instanceId}/`, {
    method: "PUT",
    headers: headers(),
    body: JSON.stringify({ state: "running" }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Start failed: ${response.status} ${JSON.stringify(body)}`);
  }
  if (body.success === false) {
    throw new Error(`Start failed: ${body.msg || body.error || "unknown"}`);
  }
  return body;
}

async function destroyInstance(instanceId) {
  const response = await fetch(`${API_BASE}/instances/${instanceId}/`, {
    method: "DELETE",
    headers: headers(),
  });
  if (!response.ok && response.status !== 404) {
    const body = await response.text().catch(() => "");
    throw new Error(`Destroy failed: ${response.status} ${body}`);
  }
}

async function fetchInstance(instanceId) {
  const value = await vastFetch(`/instances/${instanceId}/`, { method: "GET" });
  if (Array.isArray(value.instances)) return value.instances[0];
  return value.instances || value;
}

function asPort(value) {
  const port = Number(value);
  return Number.isInteger(port) && port > 0 && port < 65536 ? port : undefined;
}

function findHost(value) {
  for (const key of [
    "public_ipaddr",
    "public_ip",
    "publicIp",
    "ipaddr",
    "ssh_host",
    "host",
  ]) {
    if (typeof value?.[key] === "string" && value[key] !== "0.0.0.0") {
      return value[key];
    }
  }
}

function findExternalPort(value, internalPort) {
  if (!value || typeof value !== "object") return undefined;
  if (Array.isArray(value)) {
    for (const child of value) {
      const found = findExternalPort(child, internalPort);
      if (found) return found;
    }
    return undefined;
  }
  const internal =
    asPort(value.internal_port) ||
    asPort(value.container_port) ||
    asPort(value.private_port);
  if (internal === internalPort) {
    for (const key of [
      "external_port",
      "public_port",
      "host_port",
      "mapped_port",
      "port",
      "HostPort",
    ]) {
      const port = asPort(value[key]);
      if (port && port !== internalPort) return port;
    }
  }
  for (const [key, child] of Object.entries(value)) {
    if (key.includes(String(internalPort))) {
      const mapped = portFromMappingValue(child, internalPort);
      if (mapped) return mapped;
    }
  }
  for (const child of Object.values(value)) {
    const nested = findExternalPort(child, internalPort);
    if (nested) return nested;
  }
}

function portFromMappingValue(value, internalPort) {
  const direct = asPort(value);
  if (direct && direct !== internalPort) return direct;
  if (!value || typeof value !== "object") return undefined;
  if (Array.isArray(value)) {
    for (const child of value) {
      const found = portFromMappingValue(child, internalPort);
      if (found) return found;
    }
    return undefined;
  }
  for (const key of [
    "HostPort",
    "host_port",
    "external_port",
    "public_port",
    "mapped_port",
  ]) {
    const port = asPort(value[key]);
    if (port && port !== internalPort) return port;
  }
  for (const child of Object.values(value)) {
    const found = portFromMappingValue(child, internalPort);
    if (found) return found;
  }
}

function serviceUrl(instance) {
  const host = findHost(instance);
  const port = findExternalPort(instance, TRIBE_PORT);
  return host && port ? `http://${host}:${port}` : undefined;
}

async function waitForService(instanceId) {
  const deadline = Date.now() + Number(process.env.VAST_BOOTSTRAP_TIMEOUT_MS || 45 * 60_000);
  let last = "waiting for port map";
  while (Date.now() < deadline) {
    const instance = await fetchInstance(instanceId);
    const url = serviceUrl(instance);
    if (url) {
      try {
        const response = await fetch(`${url}/health`, {
          signal: AbortSignal.timeout(10_000),
        });
        last = `${url}/health -> ${response.status}`;
        if (response.ok) return { url, health: await response.json() };
      } catch (error) {
        last = `${url}/health -> ${error.message}`;
      }
    }
    console.log(`[wait] ${last}`);
    await new Promise((resolve) => setTimeout(resolve, 15_000));
  }
  throw new Error(`Service did not become healthy: ${last}`);
}

async function score(url) {
  const response = await fetch(`${url}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(Number(process.env.VAST_SCORE_TIMEOUT_MS || 30 * 60_000)),
    body: JSON.stringify({
      message:
        "Our deployment platform cuts release coordination time by 80 percent for engineering leaders without changing their existing CI stack.",
      persona:
        "VP of Engineering at a 200 person B2B SaaS company, pragmatic, cost-aware, under pressure to ship faster.",
      platform: "email",
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Score failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body.report;
}

let instanceId;
const started = Date.now();
try {
  if (existingInstanceId) {
    instanceId = existingInstanceId;
    console.log("[instance-existing]", instanceId);
  } else {
    const offers = parseOffers(
      await vastFetch("/bundles/", {
        method: "POST",
        body: JSON.stringify({
          limit: 20,
          type: "bid",
          verified: { eq: true },
          rentable: { eq: true },
          rented: { eq: false },
          num_gpus: { gte: 1 },
          gpu_ram: { gte: minGpuRamGb * 1024 },
          reliability: { gte: 0.95 },
          direct_port_count: { gte: 1 },
        }),
      }),
    );
    if (!offers.length) throw new Error("No Vast offers matched the configured limits.");

    let selectedOffer;
    for (const offer of offers.slice(0, 8)) {
      console.log("[offer]", JSON.stringify(offer));
      try {
        instanceId = await createInstance(offer);
        console.log("[instance]", instanceId);
        await startInstance(instanceId);
        selectedOffer = offer;
        break;
      } catch (error) {
        console.log("[skip]", error.message);
        if (instanceId) {
          await destroyInstance(instanceId).catch((destroyError) =>
            console.log("[destroy-skip-failed]", destroyError.message),
          );
          instanceId = undefined;
        }
      }
    }
    if (!selectedOffer || !instanceId) {
      throw new Error("Could not start any candidate Vast offer.");
    }
  }

  const { url, health } = await waitForService(instanceId);
  console.log("[health]", JSON.stringify({ url, health }));
  const report = await score(url);
  console.log(
    "[score]",
    JSON.stringify({
      persuasion_score: report.persuasion_score,
      verdict: report.verdict,
      model_seconds: Math.round((Date.now() - started) / 1000),
      scored_at: report.scored_at,
    }),
  );
} finally {
  if (instanceId && !keepInstance) {
    console.log("[destroy]", instanceId);
    await destroyInstance(instanceId);
  } else if (instanceId) {
    console.log("[keep]", instanceId);
  }
}
