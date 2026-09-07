#!/usr/bin/env bash
#
# gx10-rag-setup.sh
#
# Step 4: RAG infrastructure on ASUS Ascent GX10 / NVIDIA DGX Spark (GB10).
# Assumes scripts/gx10-vllm-setup.sh has already been run successfully and
# Qwen/Qwen3-30B-A3B is serving on the main vLLM container.
#
#   Step 4.1: Qdrant (vector DB) — docker run + /healthz verification
#   Step 4.2: BGE-M3 embedding service via a SECOND vLLM container
#             (--runner pooling), reusing the same NGC image already validated
#             on this hardware — see docs/gx10-known-issues.md for why a
#             raw sentence-transformers install is avoided on GB10/ARM64.
#   Step 4.3: Test collection + ingestion + semantic search
#             (delegated to scripts/rag_smoke_test.py)
#   Step 4.4: Restart Qdrant and re-check the collection to prove
#             persistence survives a container restart.
#
# Idempotent: safe to re-run. Fails loudly (non-zero exit + clear message)
# instead of silently skipping errors.
#
# Usage:
#   sudo ./gx10-rag-setup.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# CONFIG (override via environment variables)
# ============================================================================
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(whoami)}}"

QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-/home/${TARGET_USER}/.qdrant/storage}"
QDRANT_READY_TIMEOUT="${QDRANT_READY_TIMEOUT:-60}"

# Reuse the same NGC vLLM image already validated for this GB10 machine by
# gx10-vllm-setup.sh (same rationale: it has GB10/SM_121 kernels baked in).
EMBED_VLLM_IMAGE="${EMBED_VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
EMBED_CONTAINER_NAME="${EMBED_CONTAINER_NAME:-vllm-embed}"
EMBED_MODEL_HANDLE="${EMBED_MODEL_HANDLE:-BAAI/bge-m3}"
EMBED_PORT="${EMBED_PORT:-8001}"
EMBED_VECTOR_SIZE="${EMBED_VECTOR_SIZE:-1024}"
# BGE-M3 is ~568M params (~1.1GB fp16) — a tiny fraction of what the main
# Qwen3-30B-A3B vLLM server needs. IMPORTANT: this is a fraction of TOTAL
# device memory (like the main server's GPU_MEM_UTIL), not of whatever is
# currently free — vLLM reserves that whole fraction upfront. On a
# 121.63GB GB10, 0.08 alone requests ~9.73GB, which can exceed what's
# actually left if the main server's own GPU_MEM_UTIL wasn't lowered first
# (see gx10-vllm-setup.sh's GPU_MEM_UTIL, default 0.75). Keep this small —
# see docs/gx10-known-issues.md #5.
EMBED_GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL:-0.03}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/${TARGET_USER}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
EMBED_STARTUP_TIMEOUT="${EMBED_STARTUP_TIMEOUT:-600}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

# Minimum free unified-memory headroom (GB, from `nvidia-smi --query-gpu=memory.free`)
# required before starting the embedding service, checked against what's
# left AFTER the main Qwen3 server is already running.
MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB:-10}"

COLLECTION_NAME="${COLLECTION_NAME:-psf_test_kb}"
TEST_DOCS_FILE="${TEST_DOCS_FILE:-}"
QUERY_TEXT="${QUERY_TEXT:-}"

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
    die "此腳本需要 root/sudo 權限才能操作 Docker。請用: sudo $0"
  fi
}

# ============================================================================
# PRECONDITIONS
# ============================================================================
check_preconditions() {
  step "Step 4/4: 前置檢查"

  require_root

  command -v docker &>/dev/null || die "找不到 docker,請先執行 scripts/gx10-vllm-setup.sh 完成 Step 2。"
  command -v nvidia-smi &>/dev/null || die "找不到 nvidia-smi,請確認驅動已安裝且在 GX10 實體機上執行。"
  command -v curl &>/dev/null || die "找不到 curl,請先安裝: apt-get install -y curl"
  command -v python3 &>/dev/null || die "找不到 python3(Step 4.3 的 rag_smoke_test.py 需要),請先安裝: apt-get install -y python3"

  if ! docker inspect vllm-server &>/dev/null || [[ "$(docker inspect -f '{{.State.Running}}' vllm-server 2>/dev/null)" != "true" ]]; then
    warn "找不到執行中的 'vllm-server' 容器。本腳本假設 Qwen3-30B-A3B 已經在跑(scripts/gx10-vllm-setup.sh 的輸出),記憶體headroom 檢查會以目前 GPU 實際用量為準,請自行確認前提是否成立。"
  fi

  log "前置檢查通過。"
}

# ============================================================================
# STEP 4.1: QDRANT
# ============================================================================
start_qdrant() {
  step "Step 4.1/4: 啟動 Qdrant"

  mkdir -p "${QDRANT_STORAGE_DIR}" || die "建立 Qdrant 持久化目錄 ${QDRANT_STORAGE_DIR} 失敗。"
  chown -R "${TARGET_USER}:${TARGET_USER}" "${QDRANT_STORAGE_DIR}" 2>/dev/null || true

  if docker inspect "${QDRANT_CONTAINER_NAME}" &>/dev/null; then
    local state
    state="$(docker inspect -f '{{.State.Running}}' "${QDRANT_CONTAINER_NAME}")"
    if [[ "${state}" == "true" ]]; then
      log "容器 '${QDRANT_CONTAINER_NAME}' 已在執行中,略過重新建立。"
    else
      log "發現已存在但未在執行的同名容器,啟動它: docker start ${QDRANT_CONTAINER_NAME}"
      docker start "${QDRANT_CONTAINER_NAME}" &>/dev/null || die "啟動既有 Qdrant 容器失敗。"
    fi
  else
    log "拉取映像: ${QDRANT_IMAGE}"
    docker pull "${QDRANT_IMAGE}" || die "拉取 Qdrant 映像失敗,請檢查網路連線。"

    log "執行: docker run -d --name ${QDRANT_CONTAINER_NAME} -p ${QDRANT_HTTP_PORT}:6333 -p ${QDRANT_GRPC_PORT}:6334 -v ${QDRANT_STORAGE_DIR}:/qdrant/storage ${QDRANT_IMAGE}"
    docker run -d \
      --name "${QDRANT_CONTAINER_NAME}" \
      --restart unless-stopped \
      -p "${QDRANT_HTTP_PORT}:6333" \
      -p "${QDRANT_GRPC_PORT}:6334" \
      -v "${QDRANT_STORAGE_DIR}:/qdrant/storage" \
      "${QDRANT_IMAGE}" \
      || die "docker run 啟動 Qdrant 失敗。"
  fi

  log "等待 Qdrant /healthz 回應 200(逾時 ${QDRANT_READY_TIMEOUT} 秒)..."
  local elapsed=0
  while (( elapsed < QDRANT_READY_TIMEOUT )); do
    if curl -sf -o /dev/null "http://localhost:${QDRANT_HTTP_PORT}/healthz"; then
      log "Qdrant 健康檢查通過。"
      return 0
    fi
    if ! docker inspect -f '{{.State.Running}}' "${QDRANT_CONTAINER_NAME}" 2>/dev/null | grep -q true; then
      warn "Qdrant 容器提早結束,完整 log 如下:"
      docker logs "${QDRANT_CONTAINER_NAME}" >&2 || true
      die "Qdrant 容器沒有維持執行,請看上面 log。"
    fi
    sleep 2
    elapsed=$(( elapsed + 2 ))
  done

  warn "Qdrant log 如下:"
  docker logs "${QDRANT_CONTAINER_NAME}" >&2 || true
  die "在 ${QDRANT_READY_TIMEOUT} 秒內 /healthz 未回應 200,請看上面 log。"
}

# ============================================================================
# GPU HEADROOM CHECK (before starting a second GPU workload)
# ============================================================================
check_gpu_headroom() {
  step "檢查 GPU 統一記憶體剩餘空間(BGE-M3 啟動前)"

  local mem_line used_mib total_mib free_mib free_gb
  local num_re='^[0-9]+(\.[0-9]+)?$'

  mem_line="$(nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits | head -1)"
  log "nvidia-smi 原始輸出: ${mem_line}"
  used_mib="$(echo "${mem_line}" | awk -F',' '{gsub(/[^0-9.]/,"",$1); print $1}')"
  total_mib="$(echo "${mem_line}" | awk -F',' '{gsub(/[^0-9.]/,"",$2); print $2}')"
  free_mib="$(echo "${mem_line}" | awk -F',' '{gsub(/[^0-9.]/,"",$3); print $3}')"

  if [[ ! "${total_mib}" =~ ${num_re} ]]; then
    warn "nvidia-smi 沒有回傳可解析的 memory.total(拿到 '${total_mib}')。這台 GX10 上 host 層級的 'nvidia-smi --query-gpu=memory.*' 已知會整組回傳 N/A(見 docs/gx10-known-issues.md #5),此事前檢查形同略過。實際記憶體是否足夠會在容器啟動時由 vLLM 自己判斷並在 log 中報錯(本腳本會攔截並給出明確訊息),直接繼續啟動 embedding 服務。"
    return 0
  fi

  if [[ ! "${free_mib}" =~ ${num_re} ]]; then
    if [[ "${used_mib}" =~ ${num_re} ]]; then
      warn "nvidia-smi 沒有回報 memory.free(GB10 統一記憶體架構常見此狀況),改用 total - used 推算剩餘空間。"
      free_mib="$(awk "BEGIN{printf \"%.0f\", ${total_mib} - ${used_mib}}")"
    else
      warn "nvidia-smi 沒有回傳可解析的 memory.used/memory.free(used='${used_mib}', free='${free_mib}'),略過記憶體 headroom 檢查,直接繼續啟動 embedding 服務。"
      return 0
    fi
  fi

  free_gb="$(awk "BEGIN{printf \"%.1f\", ${free_mib}/1024}")"
  local used_gb_display total_gb_display
  used_gb_display="$([[ "${used_mib}" =~ ${num_re} ]] && awk "BEGIN{printf \"%.1f\", ${used_mib}/1024}" || echo "N/A")"
  total_gb_display="$(awk "BEGIN{printf \"%.1f\", ${total_mib}/1024}")"

  log "目前 GPU 記憶體: 已用 ${used_gb_display}GB / 總量 ${total_gb_display}GB,剩餘(推算)約 ${free_gb}GB"

  if awk "BEGIN{exit !(${free_gb} < ${MIN_FREE_MEM_GB})}"; then
    die "剩餘 GPU 記憶體約 ${free_gb}GB,低於安全門檻 ${MIN_FREE_MEM_GB}GB。BAAI/bge-m3 本身很小(fp16 約 1.1GB),但仍建議先確認主要 vLLM 服務(Qwen3-30B-A3B)沒有把記憶體用滿。可調降 MIN_FREE_MEM_GB 環境變數強制略過此檢查,或先降低主模型的 --gpu-memory-utilization。"
  fi
  log "剩餘記憶體足夠,繼續啟動 embedding 服務。"
}

# ============================================================================
# STEP 4.2: EMBEDDING SERVICE (BGE-M3 via vLLM --runner pooling)
# ============================================================================
start_embedding_service() {
  step "Step 4.2/4: 啟動 BGE-M3 embedding 服務"

  check_gpu_headroom

  mkdir -p "${HF_CACHE_DIR}" || die "建立 Hugging Face 快取目錄 ${HF_CACHE_DIR} 失敗。"

  if docker inspect "${EMBED_CONTAINER_NAME}" &>/dev/null; then
    local state restart_count
    state="$(docker inspect -f '{{.State.Running}}' "${EMBED_CONTAINER_NAME}")"
    restart_count="$(docker inspect -f '{{.RestartCount}}' "${EMBED_CONTAINER_NAME}" 2>/dev/null || echo 0)"
    if [[ "${state}" == "true" && "${restart_count}" -lt 3 ]]; then
      log "容器 '${EMBED_CONTAINER_NAME}' 已在執行中且穩定(RestartCount=${restart_count}),略過重新建立。若要換模型/參數,請先 'docker rm -f ${EMBED_CONTAINER_NAME}' 再重跑。"
      SKIP_WAIT_FOR_EMBED_STARTUP=1
      return 0
    elif [[ "${restart_count}" -ge 3 ]]; then
      warn "容器 '${EMBED_CONTAINER_NAME}' 的 RestartCount=${restart_count},疑似因參數錯誤陷入 crash loop(--restart unless-stopped 會無限重試)。強制移除重建: docker rm -f ${EMBED_CONTAINER_NAME}"
      docker rm -f "${EMBED_CONTAINER_NAME}" &>/dev/null || die "強制移除 crash-loop 容器 ${EMBED_CONTAINER_NAME} 失敗。"
    else
      log "發現已存在但未在執行的同名容器,先移除: docker rm ${EMBED_CONTAINER_NAME}"
      docker rm "${EMBED_CONTAINER_NAME}" &>/dev/null || die "移除舊容器 ${EMBED_CONTAINER_NAME} 失敗。"
    fi
  fi

  log "拉取映像(若已存在會直接使用快取): ${EMBED_VLLM_IMAGE}"
  docker pull "${EMBED_VLLM_IMAGE}" || die "拉取 embedding 用 vLLM 映像失敗。"

  local hf_token_args=()
  if [[ -n "${HF_TOKEN}" ]]; then
    hf_token_args=(-e "HF_TOKEN=${HF_TOKEN}")
  else
    warn "未設定 HF_TOKEN,下載 ${EMBED_MODEL_HANDLE} 可能被 Hugging Face Hub 限速(見 docs/gx10-known-issues.md #3)。"
  fi

  log "執行: docker run -d --name ${EMBED_CONTAINER_NAME} --gpus all -p ${EMBED_PORT}:8000 ... vllm serve ${EMBED_MODEL_HANDLE} --runner pooling"

  EMBED_START_TS="$(date +%s)"

  docker run -d \
    --name "${EMBED_CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus all \
    -p "${EMBED_PORT}:8000" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    "${hf_token_args[@]}" \
    "${EMBED_VLLM_IMAGE}" \
    vllm serve "${EMBED_MODEL_HANDLE}" \
      --runner pooling \
      --host 0.0.0.0 \
      --port 8000 \
      --gpu-memory-utilization "${EMBED_GPU_MEM_UTIL}" \
    || die "docker run 啟動 embedding 容器失敗。"

  log "容器已啟動。容器 ID: $(docker inspect -f '{{.Id}}' "${EMBED_CONTAINER_NAME}" | cut -c1-12)"
}

wait_for_embed_ready() {
  step "等待 embedding 服務載入完成(監看 docker logs)"

  if [[ "${SKIP_WAIT_FOR_EMBED_STARTUP:-0}" == "1" ]]; then
    log "容器先前已在執行中,略過等待啟動流程。"
    return 0
  fi

  local start_ts="${EMBED_START_TS:-$(date +%s)}"
  local elapsed=0
  local ready=0

  log "開始輪詢 docker logs,逾時設定 ${EMBED_STARTUP_TIMEOUT} 秒..."
  while (( elapsed < EMBED_STARTUP_TIMEOUT )); do
    if ! docker inspect -f '{{.State.Running}}' "${EMBED_CONTAINER_NAME}" 2>/dev/null | grep -q true; then
      warn "容器提早結束,完整 log 如下:"
      docker logs "${EMBED_CONTAINER_NAME}" >&2 || true
      die "embedding 容器在載入模型過程中結束(crash)。請看上面完整 log 找出原因(常見:OOM、下載失敗、模型不支援 --runner pooling)。"
    fi

    local logs
    logs="$(docker logs "${EMBED_CONTAINER_NAME}" 2>&1 || true)"

    if echo "${logs}" | grep -Eqi 'Free memory on device .* is less than desired GPU memory utilization'; then
      warn "偵測到記憶體不足錯誤,完整 log 如下:"
      echo "${logs}" >&2
      die "GPU 剩餘記憶體不足以啟動 embedding 服務。--gpu-memory-utilization 是相對『總量』預先保留,不是看實際用量 —— 主要 vLLM 服務(vllm-server)目前的 GPU_MEM_UTIL 設定可能已經把大部分統一記憶體吃掉了。解法:降低主服務的 GPU_MEM_UTIL 並重啟(例如 'docker rm -f vllm-server && GPU_MEM_UTIL=0.75 sudo -E ./scripts/gx10-vllm-setup.sh'),或降低本腳本的 EMBED_GPU_MEM_UTIL(目前 ${EMBED_GPU_MEM_UTIL})後再重跑。詳見 docs/gx10-known-issues.md #5。"
    fi

    if echo "${logs}" | grep -Eqi 'sm_121a? not recognized|CUDA error|CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|RuntimeError'; then
      warn "偵測到啟動錯誤訊息,完整 log 如下:"
      echo "${logs}" >&2
      die "embedding 服務啟動過程中出現錯誤,請看上面完整 log。"
    fi

    if echo "${logs}" | grep -Eqi 'Uvicorn running on|Application startup complete|Started server process'; then
      ready=1
      break
    fi

    sleep "${POLL_INTERVAL}"
    elapsed=$(( $(date +%s) - start_ts ))
    log "仍在載入中... 已等待 ${elapsed} 秒(逾時上限 ${EMBED_STARTUP_TIMEOUT} 秒)"
  done

  if [[ "${ready}" -ne 1 ]]; then
    warn "等待逾時,完整 log 如下:"
    docker logs "${EMBED_CONTAINER_NAME}" >&2 || true
    die "在 ${EMBED_STARTUP_TIMEOUT} 秒內未偵測到 embedding 服務啟動完成訊息。"
  fi

  EMBED_LOAD_SECONDS=$(( $(date +%s) - start_ts ))
  log "embedding 服務啟動完成,共花了 ${EMBED_LOAD_SECONDS} 秒。"
}

test_embedding_dim() {
  step "測試 embedding 服務回傳向量維度"

  local resp
  resp="$(curl -sS -X POST "http://localhost:${EMBED_PORT}/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"${EMBED_MODEL_HANDLE}\", \"input\": \"這是一段測試文字,用來確認 embedding 服務正常運作。\"}")" \
    || die "呼叫 embedding 測試請求失敗。"

  local dim
  dim="$(echo "${resp}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["data"][0]["embedding"]))' 2>/dev/null)" \
    || die "解析 embedding 回應失敗,原始回應: ${resp}"

  if [[ "${dim}" != "${EMBED_VECTOR_SIZE}" ]]; then
    die "embedding 向量維度是 ${dim},預期 ${EMBED_VECTOR_SIZE}。"
  fi
  log "embedding 測試成功,向量維度 = ${dim}。"
}

# ============================================================================
# STEP 4.3: TEST COLLECTION + INGEST + SEMANTIC SEARCH
# ============================================================================
run_rag_smoke_test() {
  step "Step 4.3/4: 建立測試 collection、寫入測試文件、語意搜尋"

  EMBED_URL="http://localhost:${EMBED_PORT}/v1/embeddings" \
  QDRANT_URL="http://localhost:${QDRANT_HTTP_PORT}" \
  COLLECTION_NAME="${COLLECTION_NAME}" \
  VECTOR_SIZE="${EMBED_VECTOR_SIZE}" \
  EMBED_MODEL="${EMBED_MODEL_HANDLE}" \
  TEST_DOCS_FILE="${TEST_DOCS_FILE}" \
  QUERY_TEXT="${QUERY_TEXT}" \
  python3 "${SCRIPT_DIR}/rag_smoke_test.py" \
    || die "rag_smoke_test.py 執行失敗,請看上面輸出。"
}

# ============================================================================
# STEP 4.4: PERSISTENCE CHECK (restart Qdrant, confirm data survives)
# ============================================================================
verify_qdrant_persistence() {
  step "Step 4.4/4: 驗證 Qdrant 資料持久化(重啟 container 後資料還在)"

  local before_count after_count
  before_count="$(curl -sS "http://localhost:${QDRANT_HTTP_PORT}/collections/${COLLECTION_NAME}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null)" \
    || die "重啟前查詢 collection point 數量失敗。"
  log "重啟前 '${COLLECTION_NAME}' 的 points_count = ${before_count}"

  log "重啟 Qdrant 容器: docker restart ${QDRANT_CONTAINER_NAME}"
  docker restart "${QDRANT_CONTAINER_NAME}" || die "重啟 Qdrant 容器失敗。"

  local elapsed=0
  while (( elapsed < QDRANT_READY_TIMEOUT )); do
    if curl -sf -o /dev/null "http://localhost:${QDRANT_HTTP_PORT}/healthz"; then
      break
    fi
    sleep 2
    elapsed=$(( elapsed + 2 ))
  done
  if (( elapsed >= QDRANT_READY_TIMEOUT )); then
    die "Qdrant 重啟後在 ${QDRANT_READY_TIMEOUT} 秒內 /healthz 未回應 200。"
  fi

  after_count="$(curl -sS "http://localhost:${QDRANT_HTTP_PORT}/collections/${COLLECTION_NAME}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["points_count"])' 2>/dev/null)" \
    || die "重啟後查詢 collection point 數量失敗。"
  log "重啟後 '${COLLECTION_NAME}' 的 points_count = ${after_count}"

  if [[ "${before_count}" != "${after_count}" ]] || [[ "${before_count}" -eq 0 ]]; then
    die "持久化驗證失敗:重啟前後 points_count 不一致或為 0(before=${before_count}, after=${after_count})。請檢查 volume mount ${QDRANT_STORAGE_DIR} 是否正確掛載。"
  fi

  QDRANT_PERSISTENCE_OK=1
  log "持久化驗證通過:重啟前後 points_count 一致 (${after_count})。"
}

print_summary() {
  step "總結"

  local gpu_mem embed_load_display
  gpu_mem="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")"
  if ! echo "${gpu_mem}" | grep -Eq '[0-9]'; then
    gpu_mem="無法從 host nvidia-smi 取得(已知 GB10 限制,見 docs/gx10-known-issues.md #6)。請直接執行不帶 --query-gpu 的 'nvidia-smi' 查看記憶體表格。"
  fi

  if [[ -n "${EMBED_LOAD_SECONDS:-}" ]]; then
    embed_load_display="${EMBED_LOAD_SECONDS} 秒"
  else
    embed_load_display="不適用(本次執行沿用既有容器,未重新啟動)"
  fi

  cat <<SUMMARY

Qdrant:
  容器名稱:            ${QDRANT_CONTAINER_NAME}
  HTTP port:            ${QDRANT_HTTP_PORT}
  持久化目錄(host):     ${QDRANT_STORAGE_DIR}
  持久化驗證(重啟後):   $([[ "${QDRANT_PERSISTENCE_OK:-0}" == "1" ]] && echo "通過" || echo "未驗證/失敗")

Embedding 服務:
  方案:                 vLLM --runner pooling (與主模型共用已驗證的 NGC 映像)
  模型:                 ${EMBED_MODEL_HANDLE}
  容器名稱:              ${EMBED_CONTAINER_NAME}
  Port:                  ${EMBED_PORT}
  啟動耗時:              ${embed_load_display}
  向量維度:              ${EMBED_VECTOR_SIZE}(已驗證)
  --gpu-memory-utilization: ${EMBED_GPU_MEM_UTIL}

GPU 記憶體(vLLM 主模型 + embedding 服務加總):
  ${gpu_mem}

測試 collection:        ${COLLECTION_NAME}(詳細語意搜尋結果見上方輸出)

SUMMARY
  log "Step 4 全部步驟完成。"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
  check_preconditions
  start_qdrant
  start_embedding_service
  wait_for_embed_ready
  test_embedding_dim
  run_rag_smoke_test
  verify_qdrant_persistence
  print_summary
}

main "$@"
