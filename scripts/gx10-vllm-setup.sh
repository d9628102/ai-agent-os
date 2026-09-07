#!/usr/bin/env bash
#
# gx10-vllm-setup.sh
#
# Idempotent setup script for ASUS Ascent GX10 / NVIDIA DGX Spark (GB10,
# Grace Blackwell, ARM64/aarch64, SM_121). Covers:
#   Step 2: Docker Engine + nvidia-container-toolkit install & GPU verify
#   Step 2.5: Driver version gate (fails fast, BEFORE Step 3, if the driver
#             is too old for the NGC vLLM image — see docs/gx10-known-issues.md)
#   Step 3: GB10/DGX Spark compatibility check + vLLM container launch
#   Step 4: Startup-log wait + curl inference smoke test + summary
#
# Known issues hit on real GX10 hardware (driver upgrade, Secure Boot/MOK
# enrollment, HF_TOKEN rate limiting, RAM headroom) are documented in
# docs/gx10-known-issues.md — read it before running this on a new machine.
#
# MUST be run directly on the target machine (not in a container/VM) as a
# user with sudo privileges. Re-running is safe: every step checks current
# state before acting.
#
# Usage:
#   sudo ./gx10-vllm-setup.sh
#
# Common overrides (env vars), see CONFIG section below for the full list:
#   MODEL_HANDLE=Qwen/Qwen3-30B-A3B  HF_TOKEN=hf_xxx  ./gx10-vllm-setup.sh
#
set -euo pipefail

# ============================================================================
# CONFIG (override any of these via environment variables)
# ============================================================================
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(whoami)}}"

# vLLM image: NVIDIA's NGC catalog ships prebuilt images with GB10/SM_121
# kernels included. Upstream vllm/vllm-openai:latest does NOT (its bundled
# PyTorch has no sm_121 kernels compiled in), so it is only used as a last
# resort fallback with a loud warning.
NGC_VLLM_IMAGE="${NGC_VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
FALLBACK_VLLM_IMAGE="${FALLBACK_VLLM_IMAGE:-vllm/vllm-openai:cu130-nightly}"
NGC_API_KEY="${NGC_API_KEY:-}"

# Minimum driver version required by NGC_VLLM_IMAGE. GX10 factory image can
# ship with an older driver (e.g. 580.x) that is too old — see
# docs/gx10-known-issues.md #1/#2 for the upgrade + Secure Boot/MOK steps.
MIN_DRIVER_VERSION="${MIN_DRIVER_VERSION:-595.58}"
ALLOW_OLD_DRIVER="${ALLOW_OLD_DRIVER:-0}"

MODEL_HANDLE="${MODEL_HANDLE:-Qwen/Qwen3-30B-A3B}"
QUANTIZATION="${QUANTIZATION:-}"           # e.g. awq, fp8, nvfp4 — empty = let vLLM decide
# NOTE: --gpu-memory-utilization is a fraction of TOTAL device memory that
# vLLM reserves upfront at startup (not just what the weights need), and it
# holds that whole reservation for the process's lifetime. On a 121.63GB
# GB10, 0.90 reserves ~109.5GB even though Qwen3-30B-A3B's weights are only
# ~58.5GB — leaving almost no room for a second GPU workload (e.g. the
# BGE-M3 embedding server from scripts/gx10-rag-setup.sh). Default lowered
# to 0.75 (~91.2GB, still generous KV-cache headroom) to leave ~30GB free
# for RAG infra. See docs/gx10-known-issues.md #5.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-server}"
HOST_PORT="${HOST_PORT:-8000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/${TARGET_USER}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
# Secrets file sourced when HF_TOKEN isn't already in the environment. Lives
# OUTSIDE the repo so a token can never be committed. Create it with:
#   mkdir -p ~/.config/gx10-llm && chmod 700 ~/.config/gx10-llm
#   printf 'HF_TOKEN=hf_xxx\n' > ~/.config/gx10-llm/env && chmod 600 ~/.config/gx10-llm/env
HF_ENV_FILE="${HF_ENV_FILE:-/home/${TARGET_USER}/.config/gx10-llm/env}"

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"   # seconds to wait for model load
POLL_INTERVAL="${POLL_INTERVAL:-5}"          # seconds between log polls

# Compatibility gate overrides — only set these if you KNOW what you're doing.
ALLOW_NON_ARM64="${ALLOW_NON_ARM64:-0}"
ALLOW_NON_GB10="${ALLOW_NON_GB10:-0}"

GPU_SMOKE_TEST_IMAGE="${GPU_SMOKE_TEST_IMAGE:-nvidia/cuda:12.4.0-base-ubuntu22.04}"

# ============================================================================
# LOGGING HELPERS
# ============================================================================
_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[INFO  $(_ts)] $*"; }
warn() { echo "[WARN  $(_ts)] $*" >&2; }
die()  { echo "[ERROR $(_ts)] $*" >&2; exit 1; }
step() { echo; echo "==================================================================="; echo "  $*"; echo "==================================================================="; }

trap 'die "腳本在第 $LINENO 行意外中止(上一個指令回傳非 0)。請看上面的錯誤輸出。"' ERR

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "此腳本需要 root/sudo 權限才能安裝套件與設定 Docker。請用: sudo $0"
  fi
}

# Load HF_TOKEN from HF_ENV_FILE when it isn't already in the environment.
# The token value is NEVER echoed — only whether one was found, and a
# masked prefix so you can tell which token is in play.
load_hf_token() {
  if [[ -n "${HF_TOKEN}" ]]; then
    log "HF_TOKEN 由環境變數提供 (${HF_TOKEN:0:5}…,共 ${#HF_TOKEN} 字元)。"
    return 0
  fi

  if [[ -f "${HF_ENV_FILE}" ]]; then
    local perms
    perms="$(stat -c '%a' "${HF_ENV_FILE}" 2>/dev/null || echo '?')"
    if [[ "${perms}" != "600" && "${perms}" != "400" ]]; then
      warn "${HF_ENV_FILE} 的權限是 ${perms},建議收緊為 600: chmod 600 ${HF_ENV_FILE}"
    fi
    set -a
    # shellcheck disable=SC1090
    source "${HF_ENV_FILE}"
    set +a
    HF_TOKEN="${HF_TOKEN:-}"
    if [[ -n "${HF_TOKEN}" ]]; then
      log "已從 ${HF_ENV_FILE} 載入 HF_TOKEN (${HF_TOKEN:0:5}…,共 ${#HF_TOKEN} 字元)。"
      return 0
    fi
    warn "${HF_ENV_FILE} 存在,但裡面沒有設定 HF_TOKEN。"
  fi

  warn "未設定 HF_TOKEN(環境變數與 ${HF_ENV_FILE} 都沒有)。下載模型會以未驗證身分請求 HF Hub,可能被限速(見 docs/gx10-known-issues.md #3)。"
  return 0
}

# version_ge A B -> true (0) if version A >= version B, using natural
# version sort (handles "580.159.03" vs "595.58" correctly).
version_ge() {
  [[ "$1" == "$2" ]] && return 0
  local smaller
  smaller="$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)"
  [[ "${smaller}" != "$1" ]]
}

# ============================================================================
# STEP 0: PRECONDITIONS
# ============================================================================
check_preconditions() {
  step "Step 0/4: 前置檢查"

  require_root

  if ! command -v nvidia-smi &>/dev/null; then
    die "找不到 nvidia-smi。這台機器沒有安裝 NVIDIA 驅動,或你不是在 GX10 實體機上執行此腳本。請先確認驅動已安裝。"
  fi
  log "nvidia-smi 存在,驅動版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

  if ! command -v curl &>/dev/null; then
    die "找不到 curl,請先安裝: apt-get install -y curl"
  fi

  log "目標使用者(將加入 docker 群組): ${TARGET_USER}"
  load_hf_token
  log "前置檢查通過。"
}

# ============================================================================
# STEP 2.1-2.3: DOCKER ENGINE INSTALL (idempotent, official apt repo method)
# ============================================================================
install_docker_engine() {
  step "Step 2/4: 安裝 Docker Engine"

  if snap list docker &>/dev/null 2>&1; then
    die "偵測到 snap 版本的 docker 已安裝,這會與官方 apt 版本衝突。請先手動移除: sudo snap remove docker,再重新執行本腳本。"
  fi

  if command -v docker &>/dev/null && docker info &>/dev/null; then
    log "Docker 已安裝且可正常運作 ($(docker --version)),略過安裝步驟。"
  else
    log "安裝 Docker 官方 apt 套件庫與 GPG key..."
    apt-get update -y || die "apt-get update 失敗,請檢查網路連線或 apt 來源設定。"
    apt-get install -y ca-certificates curl gnupg || die "安裝 ca-certificates/curl/gnupg 失敗。"

    install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
      . /etc/os-release
      curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o /etc/apt/keyrings/docker.asc \
        || die "下載 Docker GPG key 失敗,請檢查網路是否能連到 download.docker.com。"
      chmod a+r /etc/apt/keyrings/docker.asc
    else
      log "Docker GPG key 已存在,略過下載。"
    fi

    if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
      . /etc/os-release
      ARCH="$(dpkg --print-architecture)"
      echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list
      log "已寫入 Docker apt 來源 (arch=${ARCH}, codename=${VERSION_CODENAME})。"
    else
      log "Docker apt 來源已存在,略過寫入。"
    fi

    apt-get update -y || die "寫入 Docker 來源後 apt-get update 失敗,請檢查 /etc/apt/sources.list.d/docker.list 內容。"
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
      || die "安裝 docker-ce 相關套件失敗。"

    systemctl enable --now docker || die "啟動/啟用 docker service 失敗。"
    log "Docker Engine 安裝完成: $(docker --version)"
  fi

  # 2.2: add user to docker group (idempotent)
  if id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx docker; then
    log "使用者 ${TARGET_USER} 已在 docker 群組中,略過。"
  else
    usermod -aG docker "${TARGET_USER}" || die "將 ${TARGET_USER} 加入 docker 群組失敗。"
    warn "已將 ${TARGET_USER} 加入 docker 群組。這在目前已登入的 shell 不會立即生效,請登出重新登入(或執行 'newgrp docker')後再用非 sudo 方式跑 docker 指令。"
  fi

  # 2.3: verify with hello-world
  log "驗證 Docker 安裝: docker run hello-world"
  if ! docker run --rm hello-world &>/tmp/docker-hello-world.log; then
    cat /tmp/docker-hello-world.log >&2
    die "docker run hello-world 失敗,請看上面輸出。常見原因:docker daemon 未啟動、網路無法連到 Docker Hub。"
  fi
  log "Docker hello-world 驗證成功。"
}

# ============================================================================
# STEP 2.4: NVIDIA CONTAINER TOOLKIT (idempotent)
# ============================================================================
install_nvidia_container_toolkit() {
  step "Step 2.4/4: 安裝 nvidia-container-toolkit 並驗證容器內 GPU 存取"

  if dpkg -l nvidia-container-toolkit &>/dev/null && dpkg -s nvidia-container-toolkit 2>/dev/null | grep -q "Status: install ok installed"; then
    log "nvidia-container-toolkit 已安裝,略過套件安裝步驟。"
  else
    log "新增 NVIDIA Container Toolkit apt 來源..."
    if [[ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]]; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        || die "下載/匯入 NVIDIA Container Toolkit GPG key 失敗。"
    else
      log "NVIDIA Container Toolkit GPG key 已存在,略過。"
    fi

    if [[ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]]; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
        || die "寫入 NVIDIA Container Toolkit apt 來源失敗。"
    else
      log "NVIDIA Container Toolkit apt 來源已存在,略過。"
    fi

    apt-get update -y || die "新增 NVIDIA Container Toolkit 來源後 apt-get update 失敗。"
    apt-get install -y nvidia-container-toolkit || die "安裝 nvidia-container-toolkit 失敗。"
    log "nvidia-container-toolkit 安裝完成。"
  fi

  log "設定 Docker runtime 使用 nvidia-container-toolkit..."
  nvidia-ctk runtime configure --runtime=docker || die "nvidia-ctk runtime configure 失敗。"
  systemctl restart docker || die "重新啟動 docker service 失敗。"

  log "驗證容器內可存取 GPU: docker run --rm --gpus all ${GPU_SMOKE_TEST_IMAGE} nvidia-smi"
  if ! docker run --rm --gpus all "${GPU_SMOKE_TEST_IMAGE}" nvidia-smi &> /tmp/docker-gpu-smoke-test.log; then
    cat /tmp/docker-gpu-smoke-test.log >&2
    die "容器內無法看到 GPU。請看上面輸出。常見原因:nvidia-container-toolkit 設定未生效(需要重開 docker daemon)、驅動與容器工具版本不相容。"
  fi
  log "容器內 GPU 驗證成功,內容如下:"
  cat /tmp/docker-gpu-smoke-test.log
}

# ============================================================================
# STEP 2.5: DRIVER VERSION CHECK (run BEFORE Step 3, not after container fails)
# ============================================================================
check_driver_version() {
  step "Step 2.5/4: 檢查驅動版本是否符合 NGC 官方 vLLM 映像需求"

  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
  log "目前驅動版本: ${driver_version}(NGC 映像 ${NGC_VLLM_IMAGE} 要求 ${MIN_DRIVER_VERSION}+)"

  if version_ge "${driver_version}" "${MIN_DRIVER_VERSION}"; then
    log "驅動版本符合需求,繼續 Step 3。"
    return 0
  fi

  warn "驅動版本 ${driver_version} 低於官方 NGC vLLM 映像要求的 ${MIN_DRIVER_VERSION}+。"
  warn "已知問題與解法(詳見 docs/gx10-known-issues.md):"
  warn "  1) 升級驅動: sudo apt update && sudo apt install nvidia-driver-595-open && sudo reboot"
  warn "  2) 若此機器啟用 Secure Boot,升級後重開機核心模組可能被拒絕載入:"
  warn "       modprobe: ERROR: could not insert 'nvidia': Key was rejected by service"
  warn "     解法: sudo mokutil --import /var/lib/shim-signed/mok/MOK.der"
  warn "     設定一次性密碼後 reboot,並在下次開機時於【實體螢幕】手動完成 MOK Manager 註冊:"
  warn "       Enroll MOK -> Continue -> Yes -> 輸入密碼 -> Reboot"
  warn "     此步驟無法透過 SSH 遠端完成,請安排有人能在現場或用 KVM 操作。"

  if [[ "${ALLOW_OLD_DRIVER}" == "1" ]]; then
    warn "ALLOW_OLD_DRIVER=1,依你的指示忽略此檢查繼續執行(NGC 官方映像仍可能啟動失敗,建議搭配 fallback 映像)。"
  else
    die "驅動版本不足,已在 Step 3 開始前中止,避免等到 container 啟動失敗才發現。請先完成上面兩步升級後再重跑本腳本;若確定要略過此檢查,設定環境變數 ALLOW_OLD_DRIVER=1 後重跑。"
  fi
}

# ============================================================================
# STEP 3.1: GB10 / DGX SPARK COMPATIBILITY CHECK
# ============================================================================
check_gb10_compatibility() {
  step "Step 3/4: GB10 / DGX Spark 相容性檢查"

  local arch
  arch="$(uname -m)"
  log "系統架構: ${arch}"
  if [[ "${arch}" != "aarch64" ]]; then
    if [[ "${ALLOW_NON_ARM64}" == "1" ]]; then
      warn "架構不是 aarch64,但 ALLOW_NON_ARM64=1,依你的指示繼續執行。"
    else
      die "系統架構是 '${arch}',不是 GX10/DGX Spark 預期的 aarch64。這台機器可能不是 GB10 平台。若你確定要強制繼續,請設定環境變數 ALLOW_NON_ARM64=1 後重跑。"
    fi
  fi

  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  log "偵測到的 GPU 名稱: ${gpu_name}"
  if ! echo "${gpu_name}" | grep -Eqi 'GB10|Grace ?Blackwell|Spark'; then
    if [[ "${ALLOW_NON_GB10}" == "1" ]]; then
      warn "GPU 名稱不像是 GB10/DGX Spark,但 ALLOW_NON_GB10=1,依你的指示繼續執行。"
    else
      die "GPU 名稱 '${gpu_name}' 不像是 GB10 / DGX Spark。已知問題:vLLM 官方 upstream 映像目前不含 sm_121 kernel(見 vllm-project/vllm issue #36821、#31128),用錯映像會在啟動時出現 'sm_121a not recognized' 或 CUDA kernel 找不到的錯誤。若你確定要強制繼續,請設定環境變數 ALLOW_NON_GB10=1 後重跑。"
    fi
  else
    log "確認為 GB10 / DGX Spark 平台,SM_121(Blackwell)架構。"
  fi

  local cuda_version
  cuda_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  log "驅動版本: ${cuda_version}"
}

# ============================================================================
# STEP 3.2: SELECT & PULL A COMPATIBLE vLLM IMAGE
# ============================================================================
select_vllm_image() {
  step "選擇相容的 vLLM 映像"

  if [[ -n "${NGC_API_KEY}" ]]; then
    log "偵測到 NGC_API_KEY,嘗試登入 nvcr.io..."
    if ! echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin &>/tmp/ngc-login.log; then
      cat /tmp/ngc-login.log >&2
      warn "登入 nvcr.io 失敗,將直接嘗試不需登入的公開拉取(可能因權限不足而失敗)。"
    else
      log "nvcr.io 登入成功。"
    fi
  else
    warn "未設定 NGC_API_KEY 環境變數。NVIDIA 官方 NGC 映像 (${NGC_VLLM_IMAGE}) 是目前對 GB10/SM_121 支援最完整的來源,若拉取失敗,很可能是因為需要先在 https://ngc.nvidia.com 申請 API key 並登入。"
  fi

  log "嘗試拉取官方 NGC 映像: ${NGC_VLLM_IMAGE}"
  if docker pull "${NGC_VLLM_IMAGE}" &>/tmp/vllm-pull.log; then
    VLLM_IMAGE="${NGC_VLLM_IMAGE}"
    log "成功拉取 NGC 官方映像,將使用: ${VLLM_IMAGE}"
  else
    warn "拉取 NGC 映像失敗,詳細輸出如下:"
    cat /tmp/vllm-pull.log >&2
    warn "改用 fallback 映像: ${FALLBACK_VLLM_IMAGE}(注意:upstream vLLM 對 sm_121 的支援仍在演進中,啟動失敗時請檢查是否出現 'sm_121a not recognized' 之類的錯誤訊息)。"
    if docker pull "${FALLBACK_VLLM_IMAGE}" &>/tmp/vllm-pull-fallback.log; then
      VLLM_IMAGE="${FALLBACK_VLLM_IMAGE}"
      log "成功拉取 fallback 映像,將使用: ${VLLM_IMAGE}"
    else
      cat /tmp/vllm-pull-fallback.log >&2
      die "官方 NGC 映像與 fallback 映像都拉取失敗。請檢查網路連線、NGC_API_KEY 是否正確,或手動指定 NGC_VLLM_IMAGE / FALLBACK_VLLM_IMAGE 環境變數為已知可用的 tag。"
    fi
  fi
  export VLLM_IMAGE
}

# ============================================================================
# STEP 3.3: LAUNCH vLLM CONTAINER (idempotent)
# ============================================================================
run_vllm_container() {
  step "啟動 vLLM 容器: ${MODEL_HANDLE}"

  mkdir -p "${HF_CACHE_DIR}" || die "建立 Hugging Face 快取目錄 ${HF_CACHE_DIR} 失敗。"
  chown -R "${TARGET_USER}:${TARGET_USER}" "${HF_CACHE_DIR}" 2>/dev/null || true

  if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
    local state
    state="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")"
    if [[ "${state}" == "true" ]]; then
      log "容器 '${CONTAINER_NAME}' 已在執行中,略過重新建立。若要換模型/參數,請先手動 'docker rm -f ${CONTAINER_NAME}' 再重跑本腳本。"
      SKIP_WAIT_FOR_STARTUP="${SKIP_WAIT_FOR_STARTUP:-0}"
      return 0
    else
      log "發現已存在但未在執行的同名容器,先移除: docker rm ${CONTAINER_NAME}"
      docker rm "${CONTAINER_NAME}" &>/dev/null || die "移除舊容器 ${CONTAINER_NAME} 失敗。"
    fi
  fi

  local quant_args=()
  if [[ -n "${QUANTIZATION}" ]]; then
    quant_args=(--quantization "${QUANTIZATION}")
    log "使用量化方式: ${QUANTIZATION}"
  else
    log "未指定量化方式,交由 vLLM 依模型 config 自動判斷(如需 AWQ/FP8/NVFP4,設定 QUANTIZATION 環境變數)。"
  fi

  local hf_token_args=()
  if [[ -n "${HF_TOKEN}" ]]; then
    hf_token_args=(-e "HF_TOKEN=${HF_TOKEN}")
  else
    warn "未設定 HF_TOKEN。若 ${MODEL_HANDLE} 是 gated model 或私有 repo,下載會失敗。"
  fi

  log "執行: docker run -d --name ${CONTAINER_NAME} --restart unless-stopped --gpus all -p ${HOST_PORT}:8000 -v ${HF_CACHE_DIR}:/root/.cache/huggingface ${VLLM_IMAGE} vllm serve ${MODEL_HANDLE} ..."

  VLLM_START_TS="$(date +%s)"
  export VLLM_START_TS

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus all \
    -p "${HOST_PORT}:8000" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    "${hf_token_args[@]}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL_HANDLE}" \
      --host 0.0.0.0 \
      --port 8000 \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      "${quant_args[@]}" \
    || die "docker run 啟動 vLLM 容器失敗。"

  log "容器已啟動 (--restart unless-stopped,重開機會自動啟動)。容器 ID: $(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}" | cut -c1-12)"
}

# ============================================================================
# STEP 3.4: WAIT FOR MODEL LOAD, WATCHING FOR ERRORS (no silent skip)
# ============================================================================
wait_for_vllm_ready() {
  step "Step 3.4/4: 等待模型載入完成(監看 docker logs)"

  if [[ "${SKIP_WAIT_FOR_STARTUP:-0}" == "1" ]]; then
    log "容器先前已在執行中,略過等待啟動流程。"
    return 0
  fi

  local start_ts="${VLLM_START_TS:-$(date +%s)}"
  local elapsed=0
  local ready=0

  log "開始輪詢 docker logs,逾時設定 ${STARTUP_TIMEOUT} 秒..."
  while (( elapsed < STARTUP_TIMEOUT )); do
    if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q true; then
      warn "容器提早結束,完整 log 如下:"
      docker logs "${CONTAINER_NAME}" >&2 || true
      die "vLLM 容器在載入模型過程中結束(crash)。請看上面完整 log 找出原因(常見:OOM、CUDA/架構不相容、模型下載失敗)。"
    fi

    local logs
    logs="$(docker logs "${CONTAINER_NAME}" 2>&1 || true)"

    if echo "${logs}" | grep -Eqi 'sm_121a? not recognized|CUDA error|CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|RuntimeError'; then
      warn "偵測到啟動錯誤訊息,完整 log 如下:"
      echo "${logs}" >&2
      die "vLLM 啟動過程中出現錯誤(OOM 或架構不相容等),請看上面完整 log。這代表目前的映像 (${VLLM_IMAGE}) 或參數與這台機器不相容。"
    fi

    if echo "${logs}" | grep -Eqi 'Uvicorn running on|Application startup complete|Started server process'; then
      ready=1
      break
    fi

    sleep "${POLL_INTERVAL}"
    elapsed=$(( $(date +%s) - start_ts ))
    log "仍在載入中... 已等待 ${elapsed} 秒(逾時上限 ${STARTUP_TIMEOUT} 秒)"
  done

  if [[ "${ready}" -ne 1 ]]; then
    warn "等待逾時,完整 log 如下:"
    docker logs "${CONTAINER_NAME}" >&2 || true
    die "在 ${STARTUP_TIMEOUT} 秒內未偵測到 vLLM 啟動完成訊息。請看上面 log,或提高 STARTUP_TIMEOUT 環境變數後重跑(大模型第一次下載會比較久)。"
  fi

  VLLM_LOAD_SECONDS=$(( $(date +%s) - start_ts ))
  export VLLM_LOAD_SECONDS
  log "vLLM 啟動完成,從啟動到可用共花了 ${VLLM_LOAD_SECONDS} 秒。"
}

# ============================================================================
# STEP 4: CURL SMOKE TEST + SUMMARY
# ============================================================================
run_inference_smoke_test() {
  step "Step 4/4: 驗證服務可用(curl 測試)"

  local url="http://localhost:${HOST_PORT}/v1/chat/completions"
  local payload
  payload=$(cat <<JSON
{
  "model": "${MODEL_HANDLE}",
  "messages": [{"role": "user", "content": "你好,請用一句話自我介紹。"}],
  "max_tokens": 128
}
JSON
)

  log "送出測試請求: POST ${url}"
  local t_start t_end response http_code
  t_start="$(date +%s.%N)"
  response="$(curl -sS -w '\n%{http_code}' -X POST "${url}" \
    -H 'Content-Type: application/json' \
    -d "${payload}" 2>/tmp/curl-test.log)" || {
      cat /tmp/curl-test.log >&2
      die "curl 測試請求失敗,無法連到 ${url}。請確認容器是否還在執行: docker ps --filter name=${CONTAINER_NAME}"
    }
  t_end="$(date +%s.%N)"

  http_code="$(echo "${response}" | tail -1)"
  local body
  body="$(echo "${response}" | sed '$d')"

  if [[ "${http_code}" != "200" ]]; then
    warn "回應內容: ${body}"
    die "curl 測試請求回傳 HTTP ${http_code}(預期 200)。請看上面回應內容找出原因。"
  fi

  local elapsed_s completion_tokens tokens_per_sec
  elapsed_s="$(echo "${t_end} ${t_start}" | awk '{printf "%.2f", $1 - $2}')"
  completion_tokens="$(echo "${body}" | grep -o '"completion_tokens":[0-9]*' | head -1 | grep -o '[0-9]*' || echo "")"
  if [[ -n "${completion_tokens}" ]] && awk "BEGIN{exit !(${elapsed_s} > 0)}"; then
    tokens_per_sec="$(echo "${completion_tokens} ${elapsed_s}" | awk '{printf "%.2f", $1 / $2}')"
  else
    tokens_per_sec="N/A"
  fi

  log "測試請求成功 (HTTP 200)。"
  log "回應內容: ${body}"

  VLLM_RESP_SECONDS="${elapsed_s}"
  VLLM_TOKENS_PER_SEC="${tokens_per_sec}"
  VLLM_COMPLETION_TOKENS="${completion_tokens:-N/A}"
  export VLLM_RESP_SECONDS VLLM_TOKENS_PER_SEC VLLM_COMPLETION_TOKENS
}

print_summary() {
  step "總結"

  local gpu_mem
  gpu_mem="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")"
  local restart_policy
  restart_policy="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "${CONTAINER_NAME}" 2>/dev/null || echo "N/A")"

  cat <<SUMMARY

模型從啟動到可用花了:   ${VLLM_LOAD_SECONDS:-N/A} 秒
測試請求回應時間:       ${VLLM_RESP_SECONDS:-N/A} 秒
測試請求輸出 tokens:    ${VLLM_COMPLETION_TOKENS:-N/A}
約略 tokens/秒:         ${VLLM_TOKENS_PER_SEC:-N/A}
GPU 記憶體使用量:       ${gpu_mem}
容器名稱:               ${CONTAINER_NAME}
對外 Port:              ${HOST_PORT}
重開機自動啟動:         ${restart_policy} $([[ "${restart_policy}" == "unless-stopped" ]] && echo "(是)" || echo "(否)")
使用的 vLLM 映像:       ${VLLM_IMAGE:-N/A}
模型:                   ${MODEL_HANDLE}

服務存取方式: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${HOST_PORT}/v1/chat/completions

SUMMARY
  log "全部步驟完成。"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
  check_preconditions
  install_docker_engine
  install_nvidia_container_toolkit
  check_driver_version
  check_gb10_compatibility
  select_vllm_image
  run_vllm_container
  wait_for_vllm_ready
  run_inference_smoke_test
  print_summary
}

main "$@"
