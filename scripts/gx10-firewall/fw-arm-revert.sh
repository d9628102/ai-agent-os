#!/bin/bash
# fw-arm-revert.sh <秒數> —— 只做一件事：「先排定自動還原」（到時間自動刪除表 gx10_fw）。
# 用來在「刪表→用服務載入」的驗證流程之前先設好保險。不含任何密碼。
# 使用與 fw-apply.sh 相同的計時單元名稱（gx10-fw-revert-gx10_fw），所以可以用現有的
# fw-cancel-revert.sh gx10_fw 取消它。
# 需要 root：sudo /home/psf01/trial/net-hardening/boot/fw-arm-revert.sh 600     ← 需要 root（systemd-run、systemctl）
set -u
DELAY="${1:-}"
[ "$(id -u)" = 0 ] || { echo "錯誤：需要 root，請用 sudo 執行本腳本"; exit 1; }
case "$DELAY" in ''|*[!0-9]*) echo "用法：sudo $0 <秒數，至少 30>"; exit 1;; esac
[ "$DELAY" -ge 30 ] || { echo "錯誤：自動還原至少 30 秒"; exit 1; }
UNIT="gx10-fw-revert-gx10_fw"
systemctl stop "$UNIT.timer" "$UNIT.service" 2>/dev/null
systemd-run --unit="$UNIT" --on-active="${DELAY}s" --description="gx10 firewall auto-revert (gx10_fw)" \
  /bin/sh -c "/usr/sbin/nft delete table inet gx10_fw 2>/dev/null; exit 0" \
  || { echo "排定自動還原失敗"; exit 1; }
echo "已排定：$DELAY 秒後（$(date -d "+${DELAY} seconds" '+%H:%M:%S')）自動刪除表 gx10_fw"
echo "取消：sudo /home/psf01/trial/net-hardening/fw-cancel-revert.sh gx10_fw"
systemctl list-timers --all --no-pager 2>/dev/null | grep -F "$UNIT" || echo "警告：找不到計時任務"
