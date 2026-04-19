use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::Mutex,
    time::{Duration, Instant},
};
use tauri::{AppHandle, Manager};
use thiserror::Error;
use tokio::time::sleep;

const DEFAULT_IMAGE_FALLBACK: &str = "ghcr.io/aytzey/pitchcheck-tribe:latest";
const VAST_BOOTSTRAP_IMAGE: &str = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel";
const LOCAL_CONTAINER_NAME: &str = "pitchcheck-tribe-service";
const LOCAL_SERVICE_URL: &str = "http://127.0.0.1:8090";
const TRIBE_PORT: u16 = 8090;
const VAST_API_BASE: &str = "https://console.vast.ai/api/v0";

#[derive(Debug, Error)]
enum RuntimeError {
    #[error("{0}")]
    Message(String),
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("State lock failed")]
    StateLock,
    #[error("File operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
}

type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalGpuInfo {
    available: bool,
    vendor: Option<String>,
    name: Option<String>,
    memory_mb: Option<u64>,
    detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfferSummary {
    id: u64,
    gpu_name: Option<String>,
    num_gpus: Option<u64>,
    gpu_ram_gb: Option<f64>,
    dph_total: Option<f64>,
    reliability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatus {
    mode: String,
    connected: bool,
    service_url: Option<String>,
    local_gpu: LocalGpuInfo,
    container_id: Option<String>,
    vast_instance_id: Option<u64>,
    offer: Option<OfferSummary>,
    image: String,
    last_error: Option<String>,
}

impl Default for RuntimeStatus {
    fn default() -> Self {
        Self {
            mode: "disconnected".to_string(),
            connected: false,
            service_url: None,
            local_gpu: LocalGpuInfo::default(),
            container_id: None,
            vast_instance_id: None,
            offer: None,
            image: default_image(),
            last_error: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeConfig {
    vast_api_key: Option<String>,
    image: Option<String>,
    min_gpu_ram_gb: Option<u64>,
    max_hourly_price: Option<f64>,
    prefer_interruptible: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScoreRequest {
    message: String,
    persona: String,
    platform: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupStatus {
    platform: String,
    ready_for_local: bool,
    ready_for_cloud: bool,
    steps: Vec<SetupStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupStep {
    key: String,
    title: String,
    status: String,
    detail: String,
    action_label: Option<String>,
    url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupActionResult {
    ok: bool,
    message: String,
}

struct AppState {
    status: Mutex<RuntimeStatus>,
    client: Client,
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let status = load_status(app.handle()).unwrap_or_default();
            app.manage(AppState {
                status: Mutex::new(status),
                client: Client::new(),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime_status,
            get_setup_status,
            run_setup_step,
            connect_runtime,
            disconnect_runtime,
            check_runtime_health,
            score_pitch
        ])
        .run(tauri::generate_context!())
        .expect("error while running PitchCheck desktop app");
}

#[tauri::command]
async fn get_runtime_status(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<RuntimeStatus, String> {
    run_command(async move {
        let mut status = clone_status(&state)?;
        status.local_gpu = detect_local_gpu();
        if let Some(service_url) = status.service_url.clone() {
            status.connected = service_healthy(&state.client, &service_url).await;
            if !status.connected {
                status.last_error = Some("Configured service did not answer /health.".to_string());
            }
        }
        replace_status(&app, &state, status.clone())?;
        Ok(status)
    })
    .await
}

#[tauri::command]
async fn get_setup_status(
    state: tauri::State<'_, AppState>,
    image: Option<String>,
) -> Result<SetupStatus, String> {
    run_command(async move {
        let image = clean_image(image.as_deref());
        Ok(build_setup_status(&state.client, &image).await)
    })
    .await
}

#[tauri::command]
async fn run_setup_step(
    state: tauri::State<'_, AppState>,
    key: String,
    image: Option<String>,
) -> Result<SetupActionResult, String> {
    run_command(async move {
        let image = clean_image(image.as_deref());
        match key.as_str() {
            "docker_install" => {
                open_install_url(docker_install_url())?;
                Ok(SetupActionResult {
                    ok: true,
                    message: "Docker install guide opened.".to_string(),
                })
            }
            "nvidia_toolkit" => {
                open_install_url("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")?;
                Ok(SetupActionResult {
                    ok: true,
                    message: "NVIDIA Container Toolkit guide opened.".to_string(),
                })
            }
            "vast_account" => {
                open_install_url("https://cloud.vast.ai/account/")?;
                Ok(SetupActionResult {
                    ok: true,
                    message: "Vast.ai account page opened.".to_string(),
                })
            }
            "github_release" => {
                open_install_url("https://github.com/aytzey/PitchCheck/releases/latest")?;
                Ok(SetupActionResult {
                    ok: true,
                    message: "Latest GitHub release page opened.".to_string(),
                })
            }
            "pull_tribe_image" => {
                ensure_docker_available()?;
                let output = docker_output(&["pull", &image])?;
                Ok(SetupActionResult {
                    ok: true,
                    message: format!(
                        "TRIBE image pulled: {}",
                        output.lines().last().unwrap_or("done")
                    ),
                })
            }
            "recheck" => {
                let status = build_setup_status(&state.client, &image).await;
                Ok(SetupActionResult {
                    ok: status.ready_for_local || status.ready_for_cloud,
                    message: "Setup checks refreshed.".to_string(),
                })
            }
            _ => Err(RuntimeError::Message(format!("Unknown setup step: {key}"))),
        }
    })
    .await
}

#[tauri::command]
async fn connect_runtime(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    config: RuntimeConfig,
) -> Result<RuntimeStatus, String> {
    run_command(async move {
        let local_gpu = detect_local_gpu();
        let image = clean_image(config.image.as_deref());
        if local_gpu.available {
            match connect_local(&state.client, &image, local_gpu.clone()).await {
                Ok(status) => {
                    replace_status(&app, &state, status.clone())?;
                    return Ok(status);
                }
                Err(error) => {
                    if config
                        .vast_api_key
                        .as_deref()
                        .unwrap_or("")
                        .trim()
                        .is_empty()
                    {
                        return Err(error);
                    }
                }
            }
        }

        let api_key = config
            .vast_api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                RuntimeError::Message(
                    "No local GPU runtime was available and Vast API key is empty.".to_string(),
                )
            })?;
        let status = connect_vast(&state.client, api_key, &image, &config, local_gpu).await?;
        replace_status(&app, &state, status.clone())?;
        Ok(status)
    })
    .await
}

#[tauri::command]
async fn disconnect_runtime(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    vast_api_key: Option<String>,
) -> Result<RuntimeStatus, String> {
    run_command(async move {
        let current = clone_status(&state)?;
        match current.mode.as_str() {
            "local" => stop_local_container().await?,
            "vast" => {
                let instance_id = current.vast_instance_id.ok_or_else(|| {
                    RuntimeError::Message("No Vast instance id is stored.".to_string())
                })?;
                let api_key = vast_api_key
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        RuntimeError::Message(
                            "Vast API key is required to destroy the remote instance.".to_string(),
                        )
                    })?;
                destroy_vast_instance(&state.client, api_key, instance_id).await?;
            }
            _ => {}
        }

        let mut status = RuntimeStatus::default();
        status.local_gpu = detect_local_gpu();
        status.image = current.image;
        replace_status(&app, &state, status.clone())?;
        Ok(status)
    })
    .await
}

#[tauri::command]
async fn check_runtime_health(state: tauri::State<'_, AppState>) -> Result<Value, String> {
    run_command(async move {
        let status = clone_status(&state)?;
        let service_url = status
            .service_url
            .ok_or_else(|| RuntimeError::Message("Runtime is not connected.".to_string()))?;
        let response = state
            .client
            .get(format!("{service_url}/health"))
            .timeout(Duration::from_secs(20))
            .send()
            .await?;
        let http_status = response.status();
        let body = response.json::<Value>().await.unwrap_or_else(|_| json!({}));
        if !http_status.is_success() {
            return Err(RuntimeError::Message(format!(
                "Runtime health failed with status {http_status}: {body}"
            )));
        }
        Ok(body)
    })
    .await
}

#[tauri::command]
async fn score_pitch(
    state: tauri::State<'_, AppState>,
    request: ScoreRequest,
) -> Result<Value, String> {
    run_command(async move {
        let status = clone_status(&state)?;
        let service_url = status
            .service_url
            .ok_or_else(|| RuntimeError::Message("Runtime is not connected.".to_string()))?;
        let response = state
            .client
            .post(format!("{service_url}/score"))
            .json(&request)
            .timeout(Duration::from_secs(240))
            .send()
            .await?;
        let http_status = response.status();
        let body = response
            .json::<Value>()
            .await
            .unwrap_or_else(|_| json!({ "error": "Invalid response from runtime" }));
        if !http_status.is_success() {
            return Err(RuntimeError::Message(
                body.get("detail")
                    .or_else(|| body.get("error"))
                    .and_then(Value::as_str)
                    .unwrap_or("Scoring failed")
                    .to_string(),
            ));
        }
        Ok(body)
    })
    .await
}

async fn run_command<F, T>(future: F) -> Result<T, String>
where
    F: std::future::Future<Output = RuntimeResult<T>>,
{
    future.await.map_err(|error| error.to_string())
}

fn clone_status(state: &tauri::State<'_, AppState>) -> RuntimeResult<RuntimeStatus> {
    state
        .status
        .lock()
        .map(|status| status.clone())
        .map_err(|_| RuntimeError::StateLock)
}

fn replace_status(
    app: &AppHandle,
    state: &tauri::State<'_, AppState>,
    status: RuntimeStatus,
) -> RuntimeResult<()> {
    {
        let mut guard = state.status.lock().map_err(|_| RuntimeError::StateLock)?;
        *guard = status.clone();
    }
    save_status(app, &status)?;
    Ok(())
}

fn clean_image(image: Option<&str>) -> String {
    image
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(default_image)
}

fn default_image() -> String {
    std::env::var("PITCHCHECK_TRIBE_IMAGE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_IMAGE_FALLBACK.to_string())
}

fn state_path(app: &AppHandle) -> RuntimeResult<PathBuf> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| RuntimeError::Message(format!("Cannot resolve app data dir: {error}")))?;
    fs::create_dir_all(&dir)?;
    Ok(dir.join("runtime-state.json"))
}

fn load_status(app: &AppHandle) -> RuntimeResult<RuntimeStatus> {
    let path = state_path(app)?;
    if !path.exists() {
        return Ok(RuntimeStatus::default());
    }
    let mut status = serde_json::from_str::<RuntimeStatus>(&fs::read_to_string(path)?)?;
    status.connected = false;
    status.local_gpu = detect_local_gpu();
    Ok(status)
}

fn save_status(app: &AppHandle, status: &RuntimeStatus) -> RuntimeResult<()> {
    fs::write(state_path(app)?, serde_json::to_vec_pretty(status)?)?;
    Ok(())
}

fn detect_local_gpu() -> LocalGpuInfo {
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().find(|line| !line.trim().is_empty()) {
                let mut parts = line.split(',').map(str::trim);
                let name = parts.next().filter(|value| !value.is_empty());
                let memory_mb = parts.next().and_then(|value| value.parse::<u64>().ok());
                return LocalGpuInfo {
                    available: true,
                    vendor: Some("nvidia".to_string()),
                    name: name.map(str::to_string),
                    memory_mb,
                    detail: Some(stdout.trim().to_string()),
                };
            }
        }
    }

    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            return LocalGpuInfo {
                available: true,
                vendor: Some("amd".to_string()),
                name: Some("ROCm GPU".to_string()),
                memory_mb: None,
                detail: Some("rocminfo succeeded".to_string()),
            };
        }
    }

    LocalGpuInfo {
        available: false,
        vendor: None,
        name: None,
        memory_mb: None,
        detail: Some("No local NVIDIA or ROCm GPU command answered.".to_string()),
    }
}

async fn build_setup_status(client: &Client, image: &str) -> SetupStatus {
    let local_gpu = detect_local_gpu();
    let docker_cli = command_available("docker");
    let docker_daemon = if docker_cli {
        ensure_docker_available().is_ok()
    } else {
        false
    };
    let nvidia_runtime = if docker_daemon && local_gpu.vendor.as_deref() == Some("nvidia") {
        docker_output(&["info", "--format", "{{json .Runtimes}}"])
            .map(|value| value.contains("nvidia"))
            .unwrap_or(false)
    } else {
        false
    };
    let image_present = if docker_daemon {
        docker_output(&["image", "inspect", image]).is_ok()
    } else {
        false
    };
    let release_reachable = client
        .get("https://api.github.com/repos/aytzey/PitchCheck/releases/latest")
        .header("User-Agent", "PitchCheck-Setup")
        .timeout(Duration::from_secs(8))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false);

    let mut steps = vec![
        SetupStep {
            key: "docker_install".to_string(),
            title: "Docker runtime".to_string(),
            status: if docker_daemon { "ok" } else { "missing" }.to_string(),
            detail: if docker_daemon {
                docker_output(&["version", "--format", "Docker {{.Server.Version}}"])
                    .unwrap_or_else(|_| "Docker daemon is reachable.".to_string())
            } else if docker_cli {
                "Docker CLI exists, but the daemon is not reachable. Start Docker Desktop or the Docker service.".to_string()
            } else {
                "Docker is required to run the local TRIBE image from a fresh install.".to_string()
            },
            action_label: if docker_daemon {
                None
            } else {
                Some("Open Docker installer".to_string())
            },
            url: if docker_daemon {
                None
            } else {
                Some(docker_install_url().to_string())
            },
        },
        SetupStep {
            key: "nvidia_gpu".to_string(),
            title: "Local GPU".to_string(),
            status: if local_gpu.vendor.as_deref() == Some("nvidia") {
                "ok"
            } else {
                "optional"
            }
            .to_string(),
            detail: if local_gpu.vendor.as_deref() == Some("nvidia") {
                format!(
                    "{} detected with {} MB VRAM.",
                    local_gpu.name.clone().unwrap_or_else(|| "NVIDIA GPU".to_string()),
                    local_gpu.memory_mb.unwrap_or(0)
                )
            } else {
                "No supported local NVIDIA GPU was detected. The app can use Vast.ai fallback.".to_string()
            },
            action_label: None,
            url: None,
        },
        SetupStep {
            key: "nvidia_toolkit".to_string(),
            title: "NVIDIA Docker support".to_string(),
            status: if local_gpu.vendor.as_deref() == Some("nvidia") {
                if nvidia_runtime { "ok" } else { "missing" }
            } else {
                "optional"
            }
            .to_string(),
            detail: if local_gpu.vendor.as_deref() == Some("nvidia") {
                if nvidia_runtime {
                    "Docker can expose NVIDIA GPUs to containers.".to_string()
                } else {
                    "Install NVIDIA Container Toolkit so Docker can run the CUDA TRIBE image.".to_string()
                }
            } else {
                "Skipped unless this machine has a local NVIDIA GPU.".to_string()
            },
            action_label: if local_gpu.vendor.as_deref() == Some("nvidia") && !nvidia_runtime {
                Some("Open NVIDIA guide".to_string())
            } else {
                None
            },
            url: if local_gpu.vendor.as_deref() == Some("nvidia") && !nvidia_runtime {
                Some("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html".to_string())
            } else {
                None
            },
        },
        SetupStep {
            key: "pull_tribe_image".to_string(),
            title: "TRIBE image".to_string(),
            status: if image_present {
                "ok"
            } else if docker_daemon {
                "missing"
            } else {
                "blocked"
            }
            .to_string(),
            detail: if image_present {
                format!("{image} is already cached locally.")
            } else if docker_daemon {
                format!("Pull {image} before first local run for a faster connect.")
            } else {
                "Docker must be installed before pulling the TRIBE image.".to_string()
            },
            action_label: if docker_daemon && !image_present {
                Some("Pull image".to_string())
            } else {
                None
            },
            url: None,
        },
        SetupStep {
            key: "vast_account".to_string(),
            title: "Vast.ai fallback".to_string(),
            status: "optional".to_string(),
            detail: "Add a Vast API key in the runtime panel when no local GPU is available. The app will search verified offers and pick the cheapest match.".to_string(),
            action_label: Some("Open Vast account".to_string()),
            url: Some("https://cloud.vast.ai/account/".to_string()),
        },
        SetupStep {
            key: "github_release".to_string(),
            title: "Auto updates".to_string(),
            status: if release_reachable { "ok" } else { "warning" }.to_string(),
            detail: if release_reachable {
                "GitHub releases are reachable. The app checks for signed updates on startup.".to_string()
            } else {
                "Could not reach the latest release endpoint from this network.".to_string()
            },
            action_label: Some("Open releases".to_string()),
            url: Some("https://github.com/aytzey/PitchCheck/releases/latest".to_string()),
        },
    ];

    steps.push(SetupStep {
        key: "recheck".to_string(),
        title: "Final check".to_string(),
        status: "action".to_string(),
        detail: "Refresh all checks after installing prerequisites.".to_string(),
        action_label: Some("Run checks".to_string()),
        url: None,
    });

    SetupStatus {
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        ready_for_local: docker_daemon
            && local_gpu.vendor.as_deref() == Some("nvidia")
            && nvidia_runtime,
        ready_for_cloud: true,
        steps,
    }
}

fn command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn docker_install_url() -> &'static str {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        "https://www.docker.com/products/docker-desktop/"
    } else {
        "https://docs.docker.com/engine/install/"
    }
}

fn open_install_url(url: &str) -> RuntimeResult<()> {
    let status = if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", "start", "", url]).status()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(url).status()
    } else {
        Command::new("xdg-open").arg(url).status()
    }?;

    if status.success() {
        Ok(())
    } else {
        Err(RuntimeError::Message(format!("Failed to open {url}")))
    }
}

async fn connect_local(
    client: &Client,
    image: &str,
    local_gpu: LocalGpuInfo,
) -> RuntimeResult<RuntimeStatus> {
    if local_gpu.vendor.as_deref() != Some("nvidia") {
        return Err(RuntimeError::Message(
            "A local GPU was detected, but the current TRIBE image expects NVIDIA CUDA."
                .to_string(),
        ));
    }
    ensure_docker_available()?;
    let existing = docker_output(&[
        "ps",
        "-q",
        "--filter",
        &format!("name=^/{LOCAL_CONTAINER_NAME}$"),
    ])?;
    if existing.trim().is_empty() {
        let _ = Command::new("docker")
            .args(["rm", "-f", LOCAL_CONTAINER_NAME])
            .output();

        if docker_output(&["image", "inspect", image]).is_err() {
            docker_output(&["pull", image])?;
        }

        let container_id = docker_output(&[
            "run",
            "-d",
            "--rm",
            "--gpus",
            "all",
            "--name",
            LOCAL_CONTAINER_NAME,
            "-p",
            "127.0.0.1:8090:8090",
            "-e",
            "TRIBE_DEVICE=cuda",
            "-e",
            "TRIBE_ALLOW_MOCK=0",
            image,
        ])?;
        wait_for_health(client, LOCAL_SERVICE_URL, Duration::from_secs(900)).await?;
        Ok(RuntimeStatus {
            mode: "local".to_string(),
            connected: true,
            service_url: Some(LOCAL_SERVICE_URL.to_string()),
            local_gpu,
            container_id: Some(container_id.trim().to_string()),
            vast_instance_id: None,
            offer: None,
            image: image.to_string(),
            last_error: None,
        })
    } else {
        wait_for_health(client, LOCAL_SERVICE_URL, Duration::from_secs(120)).await?;
        Ok(RuntimeStatus {
            mode: "local".to_string(),
            connected: true,
            service_url: Some(LOCAL_SERVICE_URL.to_string()),
            local_gpu,
            container_id: Some(existing.trim().to_string()),
            vast_instance_id: None,
            offer: None,
            image: image.to_string(),
            last_error: None,
        })
    }
}

async fn stop_local_container() -> RuntimeResult<()> {
    let output = Command::new("docker")
        .args(["rm", "-f", LOCAL_CONTAINER_NAME])
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No such container") {
            Ok(())
        } else {
            Err(RuntimeError::Message(format!(
                "Failed to stop local container: {}",
                stderr.trim()
            )))
        }
    }
}

fn ensure_docker_available() -> RuntimeResult<()> {
    docker_output(&["version", "--format", "{{.Server.Version}}"]).map(|_| ())
}

fn docker_output(args: &[&str]) -> RuntimeResult<String> {
    let output = Command::new("docker").args(args).output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(RuntimeError::Message(format!(
            "Docker command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

async fn connect_vast(
    client: &Client,
    api_key: &str,
    image: &str,
    config: &RuntimeConfig,
    local_gpu: LocalGpuInfo,
) -> RuntimeResult<RuntimeStatus> {
    let mut offers = search_vast_offers(client, api_key, config).await?;
    if let Some(max_price) = config.max_hourly_price {
        offers.retain(|offer| offer.dph_total.unwrap_or(f64::MAX) <= max_price);
    }
    offers.sort_by(|left, right| {
        left.dph_total
            .unwrap_or(f64::MAX)
            .partial_cmp(&right.dph_total.unwrap_or(f64::MAX))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .reliability
                    .unwrap_or(0.0)
                    .partial_cmp(&left.reliability.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    if offers.is_empty() {
        return Err(RuntimeError::Message(
            "No Vast offers matched the configured GPU and price limits.".to_string(),
        ));
    }

    let vast_image = select_vast_image(client, image).await;
    let use_bootstrap = vast_image != image;
    let mut last_error: Option<String> = None;
    for offer in offers.iter().take(5) {
        match create_vast_instance(client, api_key, offer, &vast_image, config, use_bootstrap).await
        {
            Ok(instance_id) => {
                if let Err(error) = start_vast_instance(client, api_key, instance_id).await {
                    let _ = destroy_vast_instance(client, api_key, instance_id).await;
                    last_error = Some(error.to_string());
                    continue;
                }
                let mut status =
                    wait_for_vast_runtime(client, api_key, instance_id, &vast_image, offer)
                        .await
                        .unwrap_or_else(|error| RuntimeStatus {
                            mode: "vast".to_string(),
                            connected: false,
                            service_url: None,
                            local_gpu: local_gpu.clone(),
                            container_id: None,
                            vast_instance_id: Some(instance_id),
                            offer: Some(offer.clone()),
                            image: vast_image.clone(),
                            last_error: Some(error.to_string()),
                        });
                status.local_gpu = local_gpu;
                return Ok(status);
            }
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }
    }

    Err(RuntimeError::Message(format!(
        "Vast instance creation failed for all candidate offers. Last error: {}",
        last_error.unwrap_or_else(|| "unknown".to_string())
    )))
}

async fn search_vast_offers(
    client: &Client,
    api_key: &str,
    config: &RuntimeConfig,
) -> RuntimeResult<Vec<OfferSummary>> {
    let min_gpu_ram_mb = config.min_gpu_ram_gb.unwrap_or(16) * 1024;
    let offer_type = if config.prefer_interruptible.unwrap_or(true) {
        "bid"
    } else {
        "ondemand"
    };
    let body = json!({
        "limit": 40,
        "type": offer_type,
        "verified": { "eq": true },
        "rentable": { "eq": true },
        "rented": { "eq": false },
        "num_gpus": { "gte": 1 },
        "gpu_ram": { "gte": min_gpu_ram_mb },
        "reliability": { "gte": 0.95 },
        "direct_port_count": { "gte": 1 }
    });

    let response = client
        .post(format!("{VAST_API_BASE}/bundles/"))
        .bearer_auth(api_key)
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    let status = response.status();
    let value = response.json::<Value>().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(RuntimeError::Message(format!(
            "Vast offer search failed with status {status}: {value}"
        )));
    }

    Ok(parse_offers(&value))
}

fn parse_offers(value: &Value) -> Vec<OfferSummary> {
    match value.get("offers") {
        Some(Value::Array(items)) => items.iter().filter_map(parse_offer).collect(),
        Some(Value::Object(object)) => {
            if object.contains_key("id") || object.contains_key("ask_contract_id") {
                parse_offer(&Value::Object(object.clone()))
                    .into_iter()
                    .collect()
            } else {
                object.values().filter_map(parse_offer).collect()
            }
        }
        _ => vec![],
    }
}

fn parse_offer(value: &Value) -> Option<OfferSummary> {
    let id = value
        .get("id")
        .and_then(Value::as_u64)
        .or_else(|| value.get("ask_contract_id").and_then(Value::as_u64))?;
    Some(OfferSummary {
        id,
        gpu_name: value
            .get("gpu_name")
            .and_then(Value::as_str)
            .map(str::to_string),
        num_gpus: value.get("num_gpus").and_then(Value::as_u64),
        gpu_ram_gb: value
            .get("gpu_ram")
            .and_then(Value::as_f64)
            .map(|mb| mb / 1024.0),
        dph_total: value.get("dph_total").and_then(Value::as_f64),
        reliability: value.get("reliability").and_then(Value::as_f64),
    })
}

async fn select_vast_image(client: &Client, requested_image: &str) -> String {
    if requested_image == DEFAULT_IMAGE_FALLBACK && !default_ghcr_image_is_public(client).await {
        return VAST_BOOTSTRAP_IMAGE.to_string();
    }
    requested_image.to_string()
}

async fn default_ghcr_image_is_public(client: &Client) -> bool {
    client
        .get("https://ghcr.io/v2/aytzey/pitchcheck-tribe/manifests/latest")
        .header(
            "Accept",
            "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.index.v1+json",
        )
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn prebuilt_onstart_script() -> String {
    format!(
        "env >> /etc/environment; cd /app; nohup uvicorn tribe_service.app:app --host 0.0.0.0 --port {TRIBE_PORT} > /tmp/pitchcheck-tribe.log 2>&1 &"
    )
}

fn bootstrap_onstart_script() -> String {
    let script = format!(
        r#"#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git ffmpeg build-essential libsndfile1 curl ca-certificates
python -m pip install --upgrade pip setuptools wheel
python -m pip install uv==0.11.2
git clone --depth 1 https://github.com/aytzey/PitchCheck.git /workspace/PitchCheck || true
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
exec uvicorn tribe_service.app:app --host 0.0.0.0 --port {TRIBE_PORT}
"#
    );
    format!(
        "cat >/root/pitchcheck_bootstrap.sh <<'PITCHCHECK_BOOTSTRAP'\n{script}\nPITCHCHECK_BOOTSTRAP\nchmod +x /root/pitchcheck_bootstrap.sh\nnohup /bin/bash /root/pitchcheck_bootstrap.sh > /tmp/pitchcheck-bootstrap.log 2>&1 &"
    )
}

async fn create_vast_instance(
    client: &Client,
    api_key: &str,
    offer: &OfferSummary,
    image: &str,
    config: &RuntimeConfig,
    use_bootstrap: bool,
) -> RuntimeResult<u64> {
    let mut env = Map::new();
    env.insert("TRIBE_DEVICE".to_string(), json!("cuda"));
    env.insert("TRIBE_ALLOW_MOCK".to_string(), json!("0"));
    env.insert("TRIBE_CACHE_DIR".to_string(), json!("/models"));
    env.insert(
        "OPEN_BUTTON_PORT".to_string(),
        json!(TRIBE_PORT.to_string()),
    );
    env.insert(format!("-p {TRIBE_PORT}:{TRIBE_PORT}"), json!("1"));

    let onstart = if use_bootstrap {
        bootstrap_onstart_script()
    } else {
        prebuilt_onstart_script()
    };

    let mut body = json!({
        "image": image,
        "label": "pitchcheck-tribe",
        "disk": if use_bootstrap { 120 } else { 80 },
        "runtype": "ssh_direct",
        "target_state": "running",
        "cancel_unavail": true,
        "env": Value::Object(env),
        "onstart": onstart
    });

    if config.prefer_interruptible.unwrap_or(true) {
        if let Some(price) = offer.dph_total {
            body["price"] = json!((price * 1.03 * 1000.0).ceil() / 1000.0);
        }
    }

    let response = client
        .put(format!("{VAST_API_BASE}/asks/{}/", offer.id))
        .bearer_auth(api_key)
        .json(&body)
        .timeout(Duration::from_secs(45))
        .send()
        .await?;
    let status = response.status();
    let value = response.json::<Value>().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(RuntimeError::Message(format!(
            "Vast create instance failed with status {status}: {value}"
        )));
    }
    value
        .get("new_contract")
        .and_then(Value::as_u64)
        .or_else(|| value.get("id").and_then(Value::as_u64))
        .ok_or_else(|| {
            RuntimeError::Message(format!(
                "Vast create instance did not return an instance id: {value}"
            ))
        })
}

async fn wait_for_vast_runtime(
    client: &Client,
    api_key: &str,
    instance_id: u64,
    image: &str,
    offer: &OfferSummary,
) -> RuntimeResult<RuntimeStatus> {
    let deadline = Instant::now() + Duration::from_secs(900);

    loop {
        let instance = fetch_vast_instance(client, api_key, instance_id).await?;
        let iteration_error =
            if let Some(service_url) = service_url_from_instance(&instance, TRIBE_PORT) {
                match wait_for_health(client, &service_url, Duration::from_secs(15)).await {
                    Ok(()) => {
                        return Ok(RuntimeStatus {
                            mode: "vast".to_string(),
                            connected: true,
                            service_url: Some(service_url),
                            local_gpu: LocalGpuInfo::default(),
                            container_id: None,
                            vast_instance_id: Some(instance_id),
                            offer: Some(offer.clone()),
                            image: image.to_string(),
                            last_error: None,
                        });
                    }
                    Err(error) => error.to_string(),
                }
            } else {
                "Vast instance has no visible mapped service port yet.".to_string()
            };

        if Instant::now() >= deadline {
            return Err(RuntimeError::Message(format!(
                "Timed out waiting for Vast service. Last status: {iteration_error}"
            )));
        }
        sleep(Duration::from_secs(10)).await;
    }
}

async fn fetch_vast_instance(
    client: &Client,
    api_key: &str,
    instance_id: u64,
) -> RuntimeResult<Value> {
    let response = client
        .get(format!("{VAST_API_BASE}/instances/{instance_id}/"))
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    let status = response.status();
    let value = response.json::<Value>().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(RuntimeError::Message(format!(
            "Vast instance status failed with status {status}: {value}"
        )));
    }
    if let Some(instances) = value.get("instances") {
        if let Some(first) = instances.as_array().and_then(|items| items.first()) {
            return Ok(first.clone());
        }
        if !instances.is_null() {
            return Ok(instances.clone());
        }
    }
    Ok(value)
}

async fn destroy_vast_instance(
    client: &Client,
    api_key: &str,
    instance_id: u64,
) -> RuntimeResult<()> {
    let response = client
        .delete(format!("{VAST_API_BASE}/instances/{instance_id}/"))
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(());
    }
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(RuntimeError::Message(format!(
            "Vast destroy failed with status {status}: {body}"
        )));
    }
    Ok(())
}

async fn start_vast_instance(
    client: &Client,
    api_key: &str,
    instance_id: u64,
) -> RuntimeResult<()> {
    let response = client
        .put(format!("{VAST_API_BASE}/instances/{instance_id}/"))
        .bearer_auth(api_key)
        .json(&json!({ "state": "running" }))
        .timeout(Duration::from_secs(30))
        .send()
        .await?;
    let status = response.status();
    let value = response.json::<Value>().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(RuntimeError::Message(format!(
            "Vast start failed with status {status}: {value}"
        )));
    }
    if value.get("success").and_then(Value::as_bool) == Some(false) {
        return Err(RuntimeError::Message(format!(
            "Vast start failed: {}",
            value
                .get("msg")
                .or_else(|| value.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        )));
    }
    Ok(())
}

async fn wait_for_health(
    client: &Client,
    service_url: &str,
    timeout: Duration,
) -> RuntimeResult<()> {
    let deadline = Instant::now() + timeout;
    let mut last_error = String::new();
    while Instant::now() < deadline {
        match client
            .get(format!("{service_url}/health"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => last_error = format!("HTTP {}", response.status()),
            Err(error) => last_error = error.to_string(),
        }
        sleep(Duration::from_secs(5)).await;
    }
    Err(RuntimeError::Message(format!(
        "Runtime did not become healthy at {service_url}: {last_error}"
    )))
}

async fn service_healthy(client: &Client, service_url: &str) -> bool {
    client
        .get(format!("{service_url}/health"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn service_url_from_instance(instance: &Value, internal_port: u16) -> Option<String> {
    let host = find_host(instance)?;
    let port = find_external_port_for_internal(instance, internal_port)?;
    Some(format!("http://{host}:{port}"))
}

fn find_host(value: &Value) -> Option<String> {
    let keys = [
        "public_ipaddr",
        "public_ip",
        "publicIp",
        "ipaddr",
        "ssh_host",
        "host",
    ];
    for key in keys {
        if let Some(host) = value.get(key).and_then(Value::as_str) {
            if !host.is_empty() && host != "0.0.0.0" {
                return Some(host.to_string());
            }
        }
    }
    None
}

fn find_external_port_for_internal(value: &Value, internal_port: u16) -> Option<u16> {
    match value {
        Value::Object(object) => {
            if object
                .get("internal_port")
                .or_else(|| object.get("container_port"))
                .or_else(|| object.get("private_port"))
                .and_then(value_as_u16)
                == Some(internal_port)
            {
                for key in [
                    "external_port",
                    "public_port",
                    "host_port",
                    "mapped_port",
                    "port",
                ] {
                    if let Some(port) = object.get(key).and_then(value_as_u16) {
                        if port != internal_port {
                            return Some(port);
                        }
                    }
                }
            }

            for (key, child) in object {
                if key.contains(&internal_port.to_string()) {
                    if let Some(port) = port_from_mapping_value(child, internal_port) {
                        return Some(port);
                    }
                }
            }

            object
                .values()
                .find_map(|child| find_external_port_for_internal(child, internal_port))
        }
        Value::Array(items) => items
            .iter()
            .find_map(|child| find_external_port_for_internal(child, internal_port)),
        _ => None,
    }
}

fn port_from_mapping_value(value: &Value, internal_port: u16) -> Option<u16> {
    match value {
        Value::Number(_) | Value::String(_) => {
            value_as_u16(value).filter(|port| *port != internal_port)
        }
        Value::Array(items) => items
            .iter()
            .find_map(|item| port_from_mapping_value(item, internal_port)),
        Value::Object(object) => {
            for key in [
                "HostPort",
                "host_port",
                "external_port",
                "public_port",
                "mapped_port",
            ] {
                if let Some(port) = object.get(key).and_then(value_as_u16) {
                    if port != internal_port {
                        return Some(port);
                    }
                }
            }
            object
                .values()
                .find_map(|child| port_from_mapping_value(child, internal_port))
        }
        _ => None,
    }
}

fn value_as_u16(value: &Value) -> Option<u16> {
    value
        .as_u64()
        .and_then(|value| u16::try_from(value).ok())
        .or_else(|| value.as_str().and_then(|value| value.parse::<u16>().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_offers_accepts_array_response() {
        let value = json!({
            "offers": [
                {
                    "id": 11,
                    "gpu_name": "RTX 4090",
                    "num_gpus": 1,
                    "gpu_ram": 24576,
                    "dph_total": 0.31,
                    "reliability": 0.99
                }
            ]
        });

        let offers = parse_offers(&value);
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].id, 11);
        assert_eq!(offers[0].gpu_ram_gb, Some(24.0));
    }

    #[test]
    fn service_url_uses_docker_style_port_map() {
        let instance = json!({
            "public_ipaddr": "65.130.162.74",
            "ports": {
                "8090/tcp": [
                    { "HostIp": "0.0.0.0", "HostPort": "33526" }
                ]
            }
        });

        assert_eq!(
            service_url_from_instance(&instance, 8090),
            Some("http://65.130.162.74:33526".to_string())
        );
    }

    #[test]
    fn service_url_uses_explicit_port_records() {
        let instance = json!({
            "ssh_host": "203.0.113.5",
            "port_map": [
                { "internal_port": 8090, "external_port": 40123 }
            ]
        });

        assert_eq!(
            service_url_from_instance(&instance, 8090),
            Some("http://203.0.113.5:40123".to_string())
        );
    }
}
