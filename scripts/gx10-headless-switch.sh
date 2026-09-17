#!/usr/bin/env bash
#
# gx10-headless-switch.sh
#
# Stage 0 Task 1: switch the GX10 to multi-user (no desktop) so GNOME's
# share of the 128GB unified memory goes to models and KV cache instead.
#
# Ordering matters here. The original checklist verified SSH access AFTER
# the reboot, which is the one point where it is too late to help: if sshd
# isn't running and the desktop is gone, the only way back in is physically
# at the machine. Everything that could cost access is therefore checked
# BEFORE anything is changed, and this script never reboots for you.
#
# It also refuses to proceed if a running container would not come back on
# its own, since "services still run after reboot" is part of the
# acceptance criteria and a container without a restart policy silently
# fails it.
#
# Usage:
#   sudo ./gx10-headless-switch.sh              # pre-flight checks only
#   sudo ./gx10-headless-switch.sh --apply      # apply after checks pass
#   sudo ./gx10-headless-switch.sh --verify     # run after the reboot
#
set -euo pipefail

BASELINE_FILE="${BASELINE_FILE:-/var/tmp/gx10-headless-baseline.txt}"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[INFO  $(_ts)] $*"; }
warn() { echo "[WARN  $(_ts)] $*" >&2; }
die()  { echo "[ERROR $(_ts)] $*" >&2; exit 1; }
step() { echo; echo "==================================================================="; echo "  $*"; echo "==================================================================="; }

trap 'die "腳本在第 $LINENO 行意外中止。請看上面的錯誤輸出。"' ERR

require_root() {
  [[ "${EUID}" -eq 0 ]] || die "需要 root 權限。請用: sudo $0 $*"
}

# ============================================================================
# PRE-FLIGHT: everything that could cost you access to the machine
# ============================================================================
check_remote_access() {
  step "前置檢查 1/3：確認重開機後還進得來"

  local ssh_ok=1

  if ! systemctl list-unit-files 2>/dev/null | grep -Eq '^(ssh|sshd)\.service'; then
    warn "找不到 ssh/sshd service —— 這台機器沒有安裝 SSH server。"
    ssh_ok=0
  else
    local unit
    unit="$(systemctl list-unit-files 2>/dev/null | grep -Eo '^(ssh|sshd)\.service' | head -1)"
    if systemctl is-enabled "${unit}" &>/dev/null; then
      log "${unit} 已設為開機啟動。"
    else
      warn "${unit} 沒有設為開機啟動 —— 重開機後不會自己起來。"
      ssh_ok=0
    fi
    if systemctl is-active "${unit}" &>/dev/null; then
      log "${unit} 目前正在執行。"
    else
      warn "${unit} 目前沒有在執行。"
      ssh_ok=0
    fi
  fi

  if ss -tlnp 2>/dev/null | grep -q ':22\b'; then
    log "port 22 正在監聽:"
    ss -tlnp 2>/dev/null | grep ':22\b' | sed 's/^/    /'
  else
    warn "沒有偵測到 port 22 監聽中。"
    ssh_ok=0
  fi

  # Is this session itself local (a desktop terminal) or remote?
  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    log "你目前是透過 SSH 連線操作（${SSH_CONNECTION%% *} → 本機），"
    log "所以關閉桌面不會影響你現在這條連線。"
  else
    warn "你目前不是透過 SSH 操作（可能是實體機的桌面終端機或 TTY）。"
    warn "切成 headless 之後，這台機器不會再進入圖形桌面。"
    warn "屆時本機仍可用文字 TTY（Ctrl+Alt+F2 之類）登入，"
    warn "但若你平常是靠桌面環境操作，請先確認你能接受純文字介面。"
  fi

  if [[ "${ssh_ok}" -ne 1 ]]; then
    warn ""
    warn "SSH 遠端存取尚未就緒。若你需要重開機後從別台電腦連進來，"
    warn "請先完成（在切 headless 之前）:"
    warn "    sudo apt install -y openssh-server"
    warn "    sudo systemctl enable --now ssh"
    warn "    ss -tlnp | grep :22        # 確認有在監聽"
    warn "  並實際從另一台電腦成功 ssh 進來一次，再回來跑這個腳本。"
    PREFLIGHT_SSH_OK=0
  else
    PREFLIGHT_SSH_OK=1
    log "遠端存取檢查通過。"
  fi
}

check_container_restart_policies() {
  step "前置檢查 2/3：確認服務重開機後會自己回來"

  if ! command -v docker &>/dev/null; then
    warn "找不到 docker，略過容器檢查。"
    PREFLIGHT_CONTAINERS_OK=1
    return 0
  fi

  local names bad=0
  names="$(docker ps --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -z "${names}" ]]; then
    warn "目前沒有執行中的容器 —— 沒有東西需要在重開機後復原。"
    PREFLIGHT_CONTAINERS_OK=1
    return 0
  fi

  log "檢查每個執行中容器的 restart policy:"
  while IFS= read -r name; do
    [[ -z "${name}" ]] && continue
    local policy
    policy="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "${name}" 2>/dev/null || echo '?')"
    case "${policy}" in
      always|unless-stopped)
        log "  ✅ ${name}: ${policy}"
        ;;
      *)
        warn "  ❌ ${name}: '${policy}' —— 重開機後不會自動啟動"
        bad=$(( bad + 1 ))
        ;;
    esac
  done <<< "${names}"

  if (( bad > 0 )); then
    warn ""
    warn "有 ${bad} 個容器重開機後不會自己回來。修法（會重建容器）:"
    warn "    docker update --restart unless-stopped <容器名稱>"
    warn "  （docker update 可直接改既有容器的 restart policy，不需重建）"
    PREFLIGHT_CONTAINERS_OK=0
  else
    PREFLIGHT_CONTAINERS_OK=1
    log "所有執行中的容器都會在重開機後自動復原。"
  fi
}

record_baseline() {
  step "前置檢查 3/3：記錄切換前的基準值"

  # Each probe is individually guarded. A baseline is a record, not a gate:
  # one missing tool should cost that one section, not the whole file — an
  # incomplete baseline is what makes the post-reboot comparison impossible.
  _probe() {
    local title="$1"; shift
    echo "## ${title}"
    if "$@" 2>&1; then :; else echo "(取得失敗: $* )"; fi
    echo
  }

  {
    echo "# gx10 headless 切換前基準"
    echo "# 記錄時間: $(_ts)"
    echo
    _probe "systemctl get-default" systemctl get-default
    _probe "free -h" free -h
    _probe "nvidia-smi (記憶體區塊)" nvidia-smi
    _probe "執行中的容器" docker ps --format '{{.Names}}\t{{.Status}}'
  } > "${BASELINE_FILE}" 2>&1

  log "基準值已寫入 ${BASELINE_FILE}"
  echo
  sed 's/^/    /' "${BASELINE_FILE}"
}

# ============================================================================
# APPLY
# ============================================================================
apply_changes() {
  step "套用變更"

  local current
  current="$(systemctl get-default)"
  log "目前開機目標: ${current}"
  if [[ "${current}" == "multi-user.target" ]]; then
    log "已經是 multi-user.target，不需變更。"
  else
    systemctl set-default multi-user.target
    log "已設定開機目標為 multi-user.target（原為 ${current}）。"
  fi

  local unit
  for unit in gdm gdm3 lightdm sddm; do
    if systemctl list-unit-files 2>/dev/null | grep -q "^${unit}\.service"; then
      if systemctl is-enabled "${unit}" &>/dev/null; then
        systemctl disable "${unit}" &>/dev/null \
          && log "已停用 ${unit}（重開機後不再啟動圖形登入）。" \
          || warn "停用 ${unit} 失敗。"
      else
        log "${unit} 本來就沒有啟用。"
      fi
    fi
  done

  step "snap 相關項目"
  if command -v snap &>/dev/null; then
    log "目前安裝的 snap 套件:"
    snap list 2>&1 | sed 's/^/    /' || true
    echo
    log "snap 相關 timer:"
    systemctl list-timers --all 2>/dev/null | grep -i snap | sed 's/^/    /' || log "    (無)"
    echo
    # Disabling the refresh timer is safe and reversible; removing snapd
    # is not, so this script deliberately stops short of that.
    if systemctl list-unit-files 2>/dev/null | grep -q '^snapd\.refresh\.timer'; then
      systemctl disable --now snapd.refresh.timer &>/dev/null \
        && log "已停用 snapd.refresh.timer（停止背景自動更新）。" \
        || warn "停用 snapd.refresh.timer 失敗（可能本來就沒啟用）。"
    fi
    warn "本腳本「不會」移除 snapd —— 移除是不可逆的，且需要先確認"
    warn "上面列出的 snap 套件沒有你正在依賴的服務。請自行判斷後手動處理。"
  else
    log "這台機器沒有 snap，略過。"
  fi

  step "變更完成 —— 尚未重開機"
  cat <<'NEXT'

  變更只在重開機後生效。重開機前請再確認一次：

    1. 你能從另一台電腦 ssh 進來（若你需要遠端存取）
    2. 上面列出的容器 restart policy 都是 always 或 unless-stopped

  確認後執行：

      sudo reboot

  重開機後回到這個目錄執行，會自動與切換前的基準值比較：

      sudo ./scripts/gx10-headless-switch.sh --verify

NEXT
}

# ============================================================================
# VERIFY (run after reboot)
# ============================================================================
verify_after_reboot() {
  step "重開機後驗證"

  log "目前開機目標: $(systemctl get-default)"
  echo
  log "目前記憶體:"
  free -h | sed 's/^/    /'

  if [[ -f "${BASELINE_FILE}" ]]; then
    echo
    log "切換前的基準值（${BASELINE_FILE}）:"
    sed -n '/## free -h/,/^$/p' "${BASELINE_FILE}" | sed 's/^/    /'
    echo
    log "請比對上下兩組 free -h 的 available 欄位，確認記憶體確實釋放。"
  else
    warn "找不到基準檔 ${BASELINE_FILE}，無法自動比對。"
  fi

  echo
  log "GPU 狀態:"
  { nvidia-smi 2>&1 || echo "(nvidia-smi 不可用)"; } | head -15 | sed 's/^/    /'
  echo
  log "若上面的 GPU 記憶體使用量仍偏高，確認是模型服務佔用而非殘留的 X server:"
  log "    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"

  echo
  log "服務復原狀況:"
  if command -v docker &>/dev/null; then
    docker ps --format '{{.Names}}\t{{.Status}}' 2>&1 | sed 's/^/    /'
    echo
    local expected
    expected="$(sed -n '/## 執行中的容器/,$p' "${BASELINE_FILE}" 2>/dev/null | tail -n +2 | awk 'NF{print $1}' || true)"
    if [[ -n "${expected}" ]]; then
      local missing=0
      while IFS= read -r name; do
        [[ -z "${name}" ]] && continue
        if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
          log "  ✅ ${name} 已復原"
        else
          warn "  ❌ ${name} 沒有回來 —— 檢查: docker logs ${name}"
          missing=$(( missing + 1 ))
        fi
      done <<< "${expected}"
      echo
      if (( missing > 0 )); then
        die "有 ${missing} 個切換前在跑的容器沒有復原。驗收標準未達成。"
      fi
      log "切換前在跑的容器全部復原，驗收標準達成。"
    fi
  fi
}

# ============================================================================
# MAIN
# ============================================================================
main() {
  local mode="${1:-}"

  case "${mode}" in
    --verify)
      require_root "$@"
      verify_after_reboot
      ;;
    --apply)
      require_root "$@"
      check_remote_access
      check_container_restart_policies
      record_baseline
      if [[ "${PREFLIGHT_SSH_OK}" -ne 1 || "${PREFLIGHT_CONTAINERS_OK}" -ne 1 ]]; then
        echo
        die "前置檢查未全部通過（見上方 WARN）。請先處理後再加 --apply 重跑，"\
"或確定你接受這些風險時用 FORCE_HEADLESS=1 sudo $0 --apply 強制執行。"
      fi
      apply_changes
      ;;
    "")
      require_root "$@"
      check_remote_access
      check_container_restart_policies
      record_baseline
      step "這是僅檢查模式，尚未變更任何設定"
      if [[ "${PREFLIGHT_SSH_OK}" -eq 1 && "${PREFLIGHT_CONTAINERS_OK}" -eq 1 ]]; then
        log "前置檢查全部通過。確認無誤後執行: sudo $0 --apply"
      else
        warn "前置檢查有項目未通過（見上方 WARN），建議先處理再套用。"
      fi
      ;;
    *)
      die "用法: sudo $0 [--apply|--verify]"
      ;;
  esac
}

# FORCE_HEADLESS lets the pre-flight gate be overridden deliberately.
if [[ "${FORCE_HEADLESS:-0}" == "1" && "${1:-}" == "--apply" ]]; then
  require_root "$@"
  check_remote_access || true
  check_container_restart_policies || true
  record_baseline
  warn "FORCE_HEADLESS=1：略過前置檢查結果，依你的指示繼續。"
  apply_changes
else
  main "$@"
fi
