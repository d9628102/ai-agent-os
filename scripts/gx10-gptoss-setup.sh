#!/usr/bin/env bash
#
# gx10-gptoss-setup.sh
#
# Stage 0 Task 2: deploy gpt-oss-120b (MXFP4) on the GX10, following the
# same shape as gx10-rag-setup.sh — idempotent, log-watching, loud on
# failure.
#
# READ THIS BEFORE RUNNING. As of vLLM 0.20.1, gpt-oss-120b MXFP4 has no
# working path on SM121 (GB10 / DGX Spark):
#
#   * Triton MXFP4 MoE kernel fails to compile:
#       ptxas error: Feature '.tile::scatter4' not supported on .target 'sm_121a'
#     (vllm-project/vllm#41477, closed as not-planned/stale)
#   * The Marlin fallback runs but returns content: null / reasoning: null
#     while still reporting completion_tokens > 0 — wrong logits from an
#     SM80-targeted kernel (vllm-project/vllm#37030, still open)
#   * The emulation backend works at <=5 tok/s
#   * Unquantized bf16 needs ~240GB; this device has ~121.6GB
#
# So this script is written to FIND OUT which of those you hit, quickly,
# rather than to promise a working deployment. The Marlin case is the one
# that matters most: it looks like success to anything that only checks
# HTTP status, so the smoke test here checks the answer's content and
# calls out the null-content signature explicitly.
#
# Usage:
#   ./gx10-gptoss-setup.sh                  # budget-check, deploy, verify
#   GPU_MEM_UTIL=0.58 ./gx10-gptoss-setup.sh
#   ./gx10-gptoss-setup.sh --smoke-only     # re-test an already-running one
#
# On failure, check which of the three known modes it was:
#   cat /var/tmp/gx10-gptoss-last-failure
# (one of SM121_TRITON_UNSUPPORTED / SM121_MARLIN_NULL_OUTPUT / OOM, or
# absent if it wasn't one of these three — see FAILURE CLASSIFICATION below)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# CONFIG
# ============================================================================
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(whoami)}}"

GPTOSS_IMAGE="${GPTOSS_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
GPTOSS_CONTAINER="${GPTOSS_CONTAINER:-vllm-gptoss}"
GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-120b}"
# 8000 = vllm-server (Qwen3), 8001 = vllm-embed (bge-m3). 8002 is free.
GPTOSS_PORT="${GPTOSS_PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Approximate MXFP4 weight size, used for the budget check.
GPTOSS_WEIGHTS_GB="${GPTOSS_WEIGHTS_GB:-61}"
# Left empty on purpose: derived from the live budget unless you set it.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"

HF_CACHE_DIR="${HF_CACHE_DIR:-/home/${TARGET_USER}/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
HF_ENV_FILE="${HF_ENV_FILE:-/home/${TARGET_USER}/.config/gx10-llm/env}"

# 120B weights take a while to download on a first run.
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

# Below this, the model is almost certainly on the emulation path.
EMULATION_TPS_THRESHOLD="${EMULATION_TPS_THRESHOLD:-6}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[INFO  $(_ts)] $*"; }
warn() { echo "[WARN  $(_ts)] $*" >&2; }
die()  { echo "[ERROR $(_ts)] $*" >&2; exit 1; }
step() { echo; echo "==================================================================="; echo "  $*"; echo "==================================================================="; }

trap 'die "腳本在第 $LINENO 行意外中止。請看上面的錯誤輸出。"' ERR

# ============================================================================
# FAILURE CLASSIFICATION
# ============================================================================
# This device has exactly three known ways gpt-oss-120b MXFP4 fails to
# work (see the header comment). Whoever hits one of them — now or a year
# from now, on this machine or the next GX10 — should see which one
# immediately instead of debugging it as a fresh crash. Each known mode
# gets a fixed tag, printed as a single greppable line and written to
# FAILURE_MODE_FILE, distinct from each other and from an unclassified
# failure (which stays untagged rather than being guessed into one of
# these three).
#
#   SM121_TRITON_UNSUPPORTED  — Triton MXFP4 kernel won't compile on sm_121a
#   SM121_MARLIN_NULL_OUTPUT  — Marlin fallback runs but returns empty content
#   OOM                       — insufficient GPU memory, at pre-check or at
#                               runtime (CUDA OOM / vLLM's own "Free memory
#                               on device" refusal are the same class of
#                               problem, just caught at different points)
FAILURE_MODE_FILE="${FAILURE_MODE_FILE:-/var/tmp/gx10-gptoss-last-failure}"
rm -f "${FAILURE_MODE_FILE}" 2>/dev/null || true

fail_mode() {
  local tag="$1"; shift
  echo "${tag}" > "${FAILURE_MODE_FILE}" 2>/dev/null || true
  echo "[FAILURE_MODE ${tag} $(_ts)] 已知問題分類 — 見 ${FAILURE_MODE_FILE}" >&2
  die "$@"
}

# ============================================================================
# HF_TOKEN (same sanitisation as the other scripts — see known-issues #7)
# ============================================================================
sanitize_and_validate_hf_token() {
  HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '\r\n')"
  HF_TOKEN="${HF_TOKEN%\"}"; HF_TOKEN="${HF_TOKEN#\"}"
  HF_TOKEN="${HF_TOKEN%\'}"; HF_TOKEN="${HF_TOKEN#\'}"
  HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "${HF_TOKEN}" ]] && return 0

  local bad
  bad="$(HF_TOKEN="${HF_TOKEN}" python3 -c '
import os
t = os.environ["HF_TOKEN"]
for i, c in enumerate(t):
    if ord(c) > 127:
        print(f"  第 {i + 1} 個字元: {c!r} (U+{ord(c):04X})")
')"
  if [[ -n "${bad}" ]]; then
    warn "HF_TOKEN 含有非 ASCII 字元:"
    echo "${bad}" >&2
    die "HF_TOKEN 不是純 ASCII，vLLM 組 Authorization header 時會 crash。見 docs/gx10-known-issues.md #7。"
  fi
}

load_hf_token() {
  if [[ -z "${HF_TOKEN}" && -f "${HF_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(tr -d '\r' < "${HF_ENV_FILE}")
    set +a
    HF_TOKEN="${HF_TOKEN:-}"
  fi
  sanitize_and_validate_hf_token
  if [[ -n "${HF_TOKEN}" ]]; then
    log "HF_TOKEN 已載入 (${HF_TOKEN:0:5}…，共 ${#HF_TOKEN} 字元)。"
  else
    warn "未設定 HF_TOKEN。gpt-oss-120b 權重約 61GB，未驗證身分下載會被 HF Hub 限速。"
  fi
}

# ============================================================================
# PRECONDITIONS + BUDGET
# ============================================================================
check_preconditions() {
  step "前置檢查"

  command -v docker &>/dev/null || die "找不到 docker。"
  command -v curl &>/dev/null || die "找不到 curl。"
  command -v python3 &>/dev/null || die "找不到 python3（煙霧測試需要解析 JSON）。"

  if ss -tln 2>/dev/null | grep -q ":${GPTOSS_PORT}\b"; then
    local holder
    holder="$(docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null \
              | grep ":${GPTOSS_PORT}->" | awk '{print $1}' || true)"
    if [[ -n "${holder}" && "${holder}" != "${GPTOSS_CONTAINER}" ]]; then
      die "port ${GPTOSS_PORT} 已被容器 '${holder}' 佔用。請改用 GPTOSS_PORT=<其他埠> 重跑。"
    fi
  fi
  log "port ${GPTOSS_PORT} 可用。"

  load_hf_token
}

resolve_budget() {
  step "記憶體預算"

  if [[ -n "${GPU_MEM_UTIL}" ]]; then
    log "使用指定的 --gpu-memory-utilization ${GPU_MEM_UTIL}（略過自動計算）。"
    return 0
  fi

  local budget_script="${SCRIPT_DIR}/gx10-gpu-budget.sh"
  [[ -x "${budget_script}" ]] || die "找不到 ${budget_script}。請先 git pull，或自行指定 GPU_MEM_UTIL=<值>。"

  log "呼叫 gx10-gpu-budget.sh 依實際佔用計算可用的 utilization…"
  local out
  if ! out="$("${budget_script}" --plan "${GPTOSS_WEIGHTS_GB}" 2>&1)"; then
    echo "${out}" >&2
    fail_mode "OOM" "記憶體預算不足，無法部署（上面是完整分析）。
  依 Stage 0 Task 2 的建議順序，先停掉 Qwen3 再重跑：
      docker rm -f vllm-server
      $0"
  fi
  echo "${out}"

  GPU_MEM_UTIL="$(echo "${out}" | sed -n 's/.*建議 --gpu-memory-utilization \([0-9.]*\).*/\1/p' | tail -1)"
  [[ -n "${GPU_MEM_UTIL}" ]] || die "從預算輸出解析不到建議值。請自行指定 GPU_MEM_UTIL=<值> 重跑。"
  log "採用 --gpu-memory-utilization ${GPU_MEM_UTIL}"
}

# ============================================================================
# DEPLOY (idempotent)
# ============================================================================
start_gptoss() {
  step "啟動 gpt-oss-120b"

  mkdir -p "${HF_CACHE_DIR}"

  if docker inspect "${GPTOSS_CONTAINER}" &>/dev/null; then
    local state restarts
    state="$(docker inspect -f '{{.State.Running}}' "${GPTOSS_CONTAINER}")"
    restarts="$(docker inspect -f '{{.RestartCount}}' "${GPTOSS_CONTAINER}" 2>/dev/null || echo 0)"
    if [[ "${state}" == "true" && "${restarts}" -lt 3 ]]; then
      log "容器 '${GPTOSS_CONTAINER}' 已在執行中且穩定 (RestartCount=${restarts})，略過重建。"
      log "要換參數請先: docker rm -f ${GPTOSS_CONTAINER}"
      SKIP_STARTUP_WAIT=1
      return 0
    elif [[ "${restarts}" -ge 3 ]]; then
      warn "RestartCount=${restarts}，疑似 crash loop，強制移除重建。"
      docker rm -f "${GPTOSS_CONTAINER}" &>/dev/null
    else
      log "移除已停止的同名容器。"
      docker rm "${GPTOSS_CONTAINER}" &>/dev/null
    fi
  fi

  log "拉取映像: ${GPTOSS_IMAGE}"
  docker pull "${GPTOSS_IMAGE}" >/dev/null || die "拉取映像失敗。"

  local hf_args=()
  [[ -n "${HF_TOKEN}" ]] && hf_args=(-e "HF_TOKEN=${HF_TOKEN}")

  # Passed through if set, so the Triton/Marlin backend can be forced while
  # investigating. Deliberately not defaulted: reports disagree on which
  # value selects which backend, so guessing here would be worse than
  # letting vLLM choose and reading what it says in the log.
  local mxfp4_args=()
  if [[ -n "${VLLM_MXFP4_USE_MARLIN:-}" ]]; then
    mxfp4_args=(-e "VLLM_MXFP4_USE_MARLIN=${VLLM_MXFP4_USE_MARLIN}")
    warn "已傳入 VLLM_MXFP4_USE_MARLIN=${VLLM_MXFP4_USE_MARLIN}（請自行確認語意）。"
  fi

  log "執行: docker run -d --name ${GPTOSS_CONTAINER} --gpus all -p ${GPTOSS_PORT}:8000 \\"
  log "        ... vllm serve ${GPTOSS_MODEL} --max-model-len ${MAX_MODEL_LEN} \\"
  log "        --gpu-memory-utilization ${GPU_MEM_UTIL}"

  START_TS="$(date +%s)"
  docker run -d \
    --name "${GPTOSS_CONTAINER}" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    -p "${GPTOSS_PORT}:8000" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    "${hf_args[@]}" \
    "${mxfp4_args[@]}" \
    "${GPTOSS_IMAGE}" \
    vllm serve "${GPTOSS_MODEL}" \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    || die "docker run 失敗。"

  log "容器已啟動: $(docker inspect -f '{{.Id}}' "${GPTOSS_CONTAINER}" | cut -c1-12)"
}

# ============================================================================
# LOG WATCH — the three known SM121 failure modes first, then the generics
# ============================================================================
wait_for_ready() {
  step "等待載入完成（監看 docker logs）"

  if [[ "${SKIP_STARTUP_WAIT:-0}" == "1" ]]; then
    log "容器先前已在執行，略過等待。"
    return 0
  fi

  local start_ts="${START_TS:-$(date +%s)}" elapsed=0 ready=0 logs
  log "首次執行需下載約 61GB 權重，逾時設定 ${STARTUP_TIMEOUT} 秒。"

  while (( elapsed < STARTUP_TIMEOUT )); do
    if ! docker inspect -f '{{.State.Running}}' "${GPTOSS_CONTAINER}" 2>/dev/null | grep -q true; then
      warn "容器提早結束，完整 log:"
      docker logs "${GPTOSS_CONTAINER}" >&2 || true
      die "gpt-oss 容器在載入過程中結束。請看上面 log。"
    fi

    logs="$(docker logs "${GPTOSS_CONTAINER}" 2>&1 || true)"

    # --- SM121 failure mode 1: Triton MXFP4 kernel won't compile ---
    if echo "${logs}" | grep -Eqi "tile::scatter4|not supported on .target 'sm_121|ptxas error"; then
      warn "偵測到 Triton MXFP4 kernel 在 sm_121 上編譯失敗，相關 log:"
      echo "${logs}" | grep -Ei "tile::scatter4|ptxas|sm_121" | head -20 >&2
      fail_mode "SM121_TRITON_UNSUPPORTED" "這是 vllm-project/vllm#41477：Triton MXFP4 MoE kernel 用到 '.tile::scatter4' PTX，
  該指令只在 Hopper/SM100 可用，sm_121a（GB10）不支援。該 issue 已被關成 not-planned。
  可嘗試的方向（都有代價，見腳本開頭註解）：
    - 讓 vLLM 退到 Marlin：會觸發 SM121_MARLIN_NULL_OUTPUT（本腳本的煙霧測試會抓到）
    - Emulation backend：可運作但 <=5 tok/s
    - 改用其他模型作為主力推理層"
    fi

    # --- Early warning for failure mode 2 ---
    if echo "${logs}" | grep -qi "does not have native support for FP4"; then
      if [[ "${FP4_WARNED:-0}" != "1" ]]; then
        warn "vLLM 回報這張 GPU 沒有原生 FP4 支援 —— 代表正走 Marlin fallback。"
        warn "若載入成功，請特別注意煙霧測試的內容檢查（#37030 會回傳空內容）。"
        FP4_WARNED=1
      fi
    fi

    # --- Memory (the pattern from known-issues #6) ---
    if echo "${logs}" | grep -Eqi 'Free memory on device .* is less than desired GPU memory utilization'; then
      warn "記憶體不足，完整 log:"
      echo "${logs}" >&2
      fail_mode "OOM" "GPU 剩餘記憶體不足。--gpu-memory-utilization 目前為 ${GPU_MEM_UTIL}。
  先跑 ./scripts/gx10-gpu-budget.sh --plan ${GPTOSS_WEIGHTS_GB} 重新確認，
  或依 Task 2 建議先 docker rm -f vllm-server 釋放空間。"
    fi

    if echo "${logs}" | grep -Eqi 'CUDA out of memory|OutOfMemoryError'; then
      warn "OOM，完整 log:"
      echo "${logs}" >&2
      fail_mode "OOM" "載入過程 OOM。請調低 --gpu-memory-utilization 或 --max-model-len（目前 ${MAX_MODEL_LEN}）。"
    fi

    if echo "${logs}" | grep -Eqi 'Traceback \(most recent call last\)|RuntimeError|Error(Code)?: '; then
      warn "偵測到錯誤，完整 log:"
      echo "${logs}" >&2
      die "gpt-oss 啟動過程出現錯誤，請看上面完整 log。"
    fi

    if echo "${logs}" | grep -Eqi 'Uvicorn running on|Application startup complete'; then
      ready=1
      break
    fi

    sleep "${POLL_INTERVAL}"
    elapsed=$(( $(date +%s) - start_ts ))
    log "載入中… 已等待 ${elapsed} 秒（上限 ${STARTUP_TIMEOUT}）"
  done

  if [[ "${ready}" -ne 1 ]]; then
    warn "逾時，最後 60 行 log:"
    docker logs --tail 60 "${GPTOSS_CONTAINER}" >&2 || true
    die "在 ${STARTUP_TIMEOUT} 秒內未看到啟動完成訊息。"
  fi

  LOAD_SECONDS=$(( $(date +%s) - start_ts ))
  log "啟動完成，共 ${LOAD_SECONDS} 秒。"
}

# ============================================================================
# SMOKE TEST — must catch the null-content case, which looks like success
# ============================================================================
smoke_test() {
  step "煙霧測試（中文長文本摘要）"

  local url="http://localhost:${GPTOSS_PORT}/v1/chat/completions"
  local prompt
  prompt=$(cat <<'TXT'
請用三句話摘要以下內容：企業智慧基礎設施的核心概念，是讓組織內部散落在各系統的知識、決策紀錄與流程經驗，能夠被統一檢索與共用。傳統做法是各部門各自導入工具，結果是資料留在各自的孤島裡，跨部門要用時只能靠人工轉述。基礎設施的做法則是先把身分、權限、知識、記憶這幾層打通，再讓上層的應用共用同一套底層能力，如此新增應用時不需要重建一次資料串接。
TXT
)
  local payload
  payload="$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": 512,
    "temperature": 0.3,
}, ensure_ascii=False))' "${GPTOSS_MODEL}" "${prompt}")"

  log "送出請求…"
  local t0 t1 resp code body
  t0="$(date +%s.%N)"
  resp="$(curl -sS -w '\n%{http_code}' -X POST "${url}" \
          -H 'Content-Type: application/json' -d "${payload}" 2>&1)" \
    || die "curl 失敗，無法連到 ${url}。確認容器是否還在跑: docker ps --filter name=${GPTOSS_CONTAINER}"
  t1="$(date +%s.%N)"

  code="$(echo "${resp}" | tail -1)"
  body="$(echo "${resp}" | sed '$d')"
  [[ "${code}" == "200" ]] || { warn "回應: ${body}"; die "HTTP ${code}（預期 200）。"; }

  SMOKE_SECONDS="$(awk "BEGIN{printf \"%.2f\", ${t1} - ${t0}}")"

  # Parse content and token count together: #37030's signature is tokens
  # generated but content empty, which a status-code check sails right past.
  local parsed
  parsed="$(echo "${body}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ch = (d.get("choices") or [{}])[0]
msg = ch.get("message") or {}
content = msg.get("content")
reasoning = msg.get("reasoning_content") or msg.get("reasoning")
usage = d.get("usage") or {}
print(json.dumps({
    "content": content or "",
    "content_is_null": content is None,
    "reasoning_is_null": reasoning is None,
    "completion_tokens": usage.get("completion_tokens", 0),
    "finish_reason": ch.get("finish_reason", ""),
}, ensure_ascii=False))')" || die "解析回應失敗，原始回應: ${body}"

  local content tokens content_null
  content="$(echo "${parsed}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])')"
  tokens="$(echo "${parsed}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["completion_tokens"])')"
  content_null="$(echo "${parsed}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["content_is_null"])')"

  echo
  log "completion_tokens: ${tokens}"
  log "回應時間: ${SMOKE_SECONDS}s"
  echo
  log "【回答內容】"
  if [[ -n "${content}" ]]; then
    echo "${content}" | sed 's/^/    /'
  else
    echo "    (空白)"
  fi
  echo

  # --- SM121 failure mode 2: the silent one ---
  if [[ -z "${content}" && "${tokens}" -gt 0 ]]; then
    warn "HTTP 200、completion_tokens=${tokens}，但內容是空的"
    warn "（content_is_null=${content_null}）。"
    fail_mode "SM121_MARLIN_NULL_OUTPUT" "這正是 vllm-project/vllm#37030 的特徵：SM121 沒有原生 FP4 支援，
  退回針對 SM80 的 Marlin kernel 後產生錯誤 logits，取樣到錯誤的首個 token，
  於是回傳 content: null / reasoning: null。該 issue 仍為 open。

  這個部署「看起來成功」但實際不可用 —— 任何只檢查 HTTP 狀態碼的監控都會誤判。
  目前沒有已知可在 GB10 上正確執行 gpt-oss-120b MXFP4 的設定。
  建議：改用其他模型作為主力推理層，或等待上游修復。"
  fi

  if [[ -z "${content}" ]]; then
    die "回答為空且 completion_tokens=${tokens}。請看 docker logs ${GPTOSS_CONTAINER} 確認原因。"
  fi

  # --- SM121 failure mode 3: emulation backend ---
  # Guard the division: a zero elapsed time yields "inf", which awk then
  # reads back as 0 in the threshold comparison and reports as far too
  # slow — the same shape as the [N/A]/1024 crash in known-issues #6.
  if ! awk "BEGIN{exit !(${SMOKE_SECONDS} > 0)}"; then
    warn "回應時間為 ${SMOKE_SECONDS}s，無法計算 tokens/s，略過吞吐量判斷。"
    SMOKE_TPS="N/A"
    log "煙霧測試通過：有實際內容產出。"
    return 0
  fi

  local tps
  tps="$(awk "BEGIN{printf \"%.1f\", ${tokens} / ${SMOKE_SECONDS}}")"
  SMOKE_TPS="${tps}"
  log "約 ${tps} tokens/s"
  if awk "BEGIN{exit !(${tps} < ${EMULATION_TPS_THRESHOLD})}"; then
    warn "吞吐量只有 ${tps} tokens/s，低於 ${EMULATION_TPS_THRESHOLD} 的門檻。"
    warn "這通常代表落在 MXFP4 emulation backend 上（回報值約 <=5 tok/s），"
    warn "功能正確但實務上難以使用。可在 docker logs 中搜尋 backend 相關訊息確認。"
  else
    log "吞吐量正常，不像是 emulation backend。"
  fi

  log "煙霧測試通過：有實際內容產出。"
}

print_summary() {
  step "總結"
  local restart_policy
  restart_policy="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "${GPTOSS_CONTAINER}" 2>/dev/null || echo N/A)"

  cat <<SUMMARY

  容器名稱:          ${GPTOSS_CONTAINER}
  模型:              ${GPTOSS_MODEL}
  Port:              ${GPTOSS_PORT}
  --max-model-len:   ${MAX_MODEL_LEN}
  --gpu-memory-utilization: ${GPU_MEM_UTIL}
  啟動耗時:          ${LOAD_SECONDS:-N/A} 秒
  煙霧測試回應時間:  ${SMOKE_SECONDS:-N/A} 秒（約 ${SMOKE_TPS:-N/A} tokens/s）
  重開機自動啟動:    ${restart_policy}

  下一步：在 LiteLLM 設定中新增路由，指向 http://localhost:${GPTOSS_PORT}/v1
  記憶體實際佔用請用: ./scripts/gx10-gpu-budget.sh

SUMMARY
}

main() {
  if [[ "${1:-}" == "--smoke-only" ]]; then
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-(既有)}"
    smoke_test
    exit 0
  fi
  check_preconditions
  resolve_budget
  start_gptoss
  wait_for_ready
  smoke_test
  print_summary
}

main "$@"
