#!/usr/bin/env bash
#
# gx10-gpu-budget.sh
#
# Work out how much GB10 unified memory is actually available for a new
# model service, and what --gpu-memory-utilization it should use.
#
# Why this exists: every memory failure in this project came from picking
# a utilization value by feel. --gpu-memory-utilization reserves that
# fraction of TOTAL device memory up front, for the life of the process,
# so the values across concurrent services must sum to under 1.0 — a rule
# that is invisible when you look at one service at a time.
#
# Getting the real numbers needs a detour. Host nvidia-smi on this machine
# returns [N/A] for every memory field (docs/gx10-known-issues.md #6), so
# this probes the CUDA API from inside a container, which does report
# correctly here — that is where the 121.63 GiB figure came from
# originally. It also reads each running vLLM container's configured
# utilization straight out of its docker command, so the committed total
# is derived rather than remembered.
#
# Usage:
#   ./gx10-gpu-budget.sh                  # show the current budget
#   ./gx10-gpu-budget.sh --plan 61        # ...and what fits a 61GB model
#   ./gx10-gpu-budget.sh --plan 61 --reserve-kv 10
#
set -euo pipefail

PROBE_IMAGE="${PROBE_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
# Memory left unreserved for the OS, Docker, Qdrant and page cache. Headless
# mode (Task 1) is what buys room to lower this.
HOST_RESERVE_GB="${HOST_RESERVE_GB:-12}"
# Refuse to plan a service that would push total committed utilization past
# this. 1.0 is the hard limit; the gap absorbs allocator overhead.
MAX_TOTAL_UTIL="${MAX_TOTAL_UTIL:-0.92}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[INFO  $(_ts)] $*"; }
warn() { echo "[WARN  $(_ts)] $*" >&2; }
die()  { echo "[ERROR $(_ts)] $*" >&2; exit 1; }
step() { echo; echo "==================================================================="; echo "  $*"; echo "==================================================================="; }

trap 'die "腳本在第 $LINENO 行意外中止。請看上面的錯誤輸出。"' ERR

# ============================================================================
# Probe real device memory via the CUDA API inside a container
# ============================================================================
probe_device_memory() {
  step "步驟 1/3：讀取裝置實際記憶體"

  log "host 的 nvidia-smi 記憶體查詢在這台機器上不可用，改由容器內的 CUDA API 讀取。"
  log "（啟動探測容器約需 10-20 秒）"

  local out
  if ! out="$(docker run --rm --gpus all "${PROBE_IMAGE}" python3 -c '
import torch
free, total = torch.cuda.mem_get_info()
print(f"PROBE {free} {total}")
' 2>/dev/null | grep '^PROBE ')"; then
    die "無法透過容器讀取 GPU 記憶體。請確認 docker 與 GPU passthrough 正常：
    docker run --rm --gpus all ${PROBE_IMAGE} nvidia-smi"
  fi

  DEVICE_FREE_B="$(echo "${out}" | awk '{print $2}')"
  DEVICE_TOTAL_B="$(echo "${out}" | awk '{print $3}')"
  [[ "${DEVICE_TOTAL_B}" =~ ^[0-9]+$ ]] || die "解析不到裝置總記憶體，探測輸出: ${out}"

  DEVICE_TOTAL_GB="$(awk "BEGIN{printf \"%.2f\", ${DEVICE_TOTAL_B}/1024/1024/1024}")"
  DEVICE_FREE_GB="$(awk "BEGIN{printf \"%.2f\", ${DEVICE_FREE_B}/1024/1024/1024}")"
  DEVICE_USED_GB="$(awk "BEGIN{printf \"%.2f\", ${DEVICE_TOTAL_GB} - ${DEVICE_FREE_GB}}")"

  log "裝置總記憶體:   ${DEVICE_TOTAL_GB} GB"
  log "目前已佔用:     ${DEVICE_USED_GB} GB   ← 這是實際佔用，不是設定值"
  log "目前可用:       ${DEVICE_FREE_GB} GB"
}

# ============================================================================
# Read each running container's configured --gpu-memory-utilization
# ============================================================================
scan_committed_utilization() {
  step "步驟 2/3：盤點各服務已「預先保留」的比例"

  COMMITTED_UTIL=0
  SERVICE_ROWS=()
  UNSPECIFIED_SERVICES=0

  local names
  names="$(docker ps --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -z "${names}" ]]; then
    warn "目前沒有執行中的容器。"
    return 0
  fi

  local name cmd util
  while IFS= read -r name; do
    [[ -z "${name}" ]] && continue
    cmd="$(docker inspect -f '{{join .Config.Cmd " "}}' "${name}" 2>/dev/null || true)"
    if ! echo "${cmd}" | grep -q -- '--gpu-memory-utilization'; then
      # Not a vLLM service, or it left the flag at vLLM's own default.
      if echo "${cmd}" | grep -q 'vllm'; then
        warn "  ${name}: 有 vllm 但沒有明示 --gpu-memory-utilization"
        warn "    → 會套用 vLLM 預設值（通常 0.9），這是最容易被忽略的超額來源"
        # Count the default rather than zero. Leaving it out of the total
        # would make the budget most optimistic in exactly the case that is
        # most likely to be over-committed.
        local assumed="${VLLM_DEFAULT_UTIL:-0.9}"
        local assumed_gb
        assumed_gb="$(awk "BEGIN{printf \"%.1f\", ${assumed} * ${DEVICE_TOTAL_GB}}")"
        COMMITTED_UTIL="$(awk "BEGIN{printf \"%.4f\", ${COMMITTED_UTIL} + ${assumed}}")"
        SERVICE_ROWS+=("${name}|未指定(以 ${assumed} 計)|${assumed_gb}")
        UNSPECIFIED_SERVICES=$(( ${UNSPECIFIED_SERVICES:-0} + 1 ))
      fi
      continue
    fi
    util="$(echo "${cmd}" | sed -n 's/.*--gpu-memory-utilization[ =]\([0-9.]*\).*/\1/p')"
    [[ -n "${util}" ]] || continue
    local gb
    gb="$(awk "BEGIN{printf \"%.1f\", ${util} * ${DEVICE_TOTAL_GB}}")"
    COMMITTED_UTIL="$(awk "BEGIN{printf \"%.4f\", ${COMMITTED_UTIL} + ${util}}")"
    SERVICE_ROWS+=("${name}|${util}|${gb}")
  done <<< "${names}"

  printf '\n    %-20s %-22s %s\n' "容器" "--gpu-memory-utilization" "保留量"
  printf '    %-20s %-22s %s\n' "--------------------" "----------------------" "--------"
  local row
  for row in "${SERVICE_ROWS[@]:-}"; do
    [[ -z "${row}" ]] && continue
    IFS='|' read -r n u g <<< "${row}"
    printf '    %-20s %-22s %s GB\n' "${n}" "${u}" "${g}"
  done
  echo

  COMMITTED_GB="$(awk "BEGIN{printf \"%.1f\", ${COMMITTED_UTIL} * ${DEVICE_TOTAL_GB}}")"
  log "已保留比例合計: ${COMMITTED_UTIL}（約 ${COMMITTED_GB} GB）"
  if (( UNSPECIFIED_SERVICES > 0 )); then
    warn "其中 ${UNSPECIFIED_SERVICES} 個服務是以推測的預設值計入，不是讀到的實際設定。"
    warn "建議明確指定它們的 --gpu-memory-utilization，這份預算才會準確。"
  fi

  if awk "BEGIN{exit !(${COMMITTED_UTIL} >= 1.0)}"; then
    warn "合計已達或超過 1.0 —— 目前的組合本身就無法再容納任何新服務。"
  fi
}

# ============================================================================
# Plan a new service
# ============================================================================
plan_new_service() {
  local weights_gb="$1" reserve_kv_gb="$2"

  step "步驟 3/3：規劃新服務的 --gpu-memory-utilization"

  log "模型權重:       ${weights_gb} GB"
  log "預留 KV cache:  ${reserve_kv_gb} GB"
  log "系統保留:       ${HOST_RESERVE_GB} GB（OS/Docker/Qdrant/page cache）"

  local need_gb need_util available_util
  need_gb="$(awk "BEGIN{printf \"%.1f\", ${weights_gb} + ${reserve_kv_gb}}")"
  need_util="$(awk "BEGIN{printf \"%.3f\", ${need_gb} / ${DEVICE_TOTAL_GB}}")"

  # What's left after existing reservations and the host allowance.
  local host_util
  host_util="$(awk "BEGIN{printf \"%.4f\", ${HOST_RESERVE_GB} / ${DEVICE_TOTAL_GB}}")"
  available_util="$(awk "BEGIN{printf \"%.3f\", ${MAX_TOTAL_UTIL} - ${COMMITTED_UTIL} - ${host_util}}")"
  local available_gb
  available_gb="$(awk "BEGIN{printf \"%.1f\", ${available_util} * ${DEVICE_TOTAL_GB}}")"

  echo
  log "這個新服務需要:   ${need_gb} GB → util ${need_util}"
  log "目前還能給:       ${available_gb} GB → util ${available_util}"
  log "  （上限 ${MAX_TOTAL_UTIL} − 已保留 ${COMMITTED_UTIL} − 系統 $(printf '%.3f' "${host_util}")）"
  echo

  if awk "BEGIN{exit !(${available_util} <= 0)}"; then
    die "沒有可用空間了。必須先降低既有服務的 --gpu-memory-utilization，
  或停掉其中一個服務，才能加入新模型。
  參考: docker rm -f <容器> 後用較低的 GPU_MEM_UTIL 重新啟動。"
  fi

  if awk "BEGIN{exit !(${need_util} > ${available_util})}"; then
    local max_kv
    max_kv="$(awk "BEGIN{printf \"%.1f\", ${available_gb} - ${weights_gb}}")"
    warn "放不下：需要 ${need_util} 但只剩 ${available_util}。"
    echo
    if awk "BEGIN{exit !(${max_kv} > 0)}"; then
      warn "  在不動既有服務的前提下，這個模型最多只能配到 ${available_gb} GB，"
      warn "  扣掉 ${weights_gb} GB 權重後 KV cache 只剩 ${max_kv} GB。"
      warn "  選項："
      warn "    1. 降低 --max-model-len（KV cache 需求隨上下文長度線性下降）"
      warn "    2. 加上 --kv-cache-dtype fp8（KV cache 約減半）"
      warn "    3. 降低或停用既有服務（見上方盤點表）"
    else
      warn "  連權重都放不下（權重 ${weights_gb} GB > 可用 ${available_gb} GB）。"
      warn "  必須先騰出既有服務的空間。"
    fi
    RECOMMENDED_UTIL=""
    return 1
  fi

  RECOMMENDED_UTIL="${need_util}"
  local new_total
  new_total="$(awk "BEGIN{printf \"%.3f\", ${COMMITTED_UTIL} + ${need_util}}")"
  log "✅ 放得下。建議 --gpu-memory-utilization ${RECOMMENDED_UTIL}"
  log "   加入後各服務保留合計: ${new_total}（上限 ${MAX_TOTAL_UTIL}）"
  echo
  log "注意：這是「保留」不是「用滿」。實際權重佔 ${weights_gb} GB，"
  log "      其餘 ${reserve_kv_gb} GB 由 vLLM 作為 KV cache 池使用。"
}

# ============================================================================
# MAIN
# ============================================================================
main() {
  local plan_gb="" reserve_kv=10

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)        plan_gb="${2:-}"; shift 2 ;;
      --reserve-kv)  reserve_kv="${2:-}"; shift 2 ;;
      -h|--help)
        sed -n '2,30p' "$0" | sed 's/^# \?//'
        exit 0 ;;
      *) die "未知參數: $1（用 --help 看用法）" ;;
    esac
  done

  command -v docker &>/dev/null || die "找不到 docker。"

  probe_device_memory
  scan_committed_utilization

  if [[ -n "${plan_gb}" ]]; then
    [[ "${plan_gb}" =~ ^[0-9.]+$ ]] || die "--plan 需要數字（模型權重 GB），收到: ${plan_gb}"
    plan_new_service "${plan_gb}" "${reserve_kv}" || exit 1
  else
    echo
    log "要規劃新服務，加上 --plan <權重GB>，例如："
    log "    $0 --plan 61                 # gpt-oss-120b (MXFP4) 約 61GB"
    log "    $0 --plan 61 --reserve-kv 16 # 想要更多 KV cache 空間"
  fi
}

main "$@"
