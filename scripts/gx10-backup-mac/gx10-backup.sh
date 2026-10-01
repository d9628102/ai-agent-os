#!/bin/bash
# Mac 端：從 GX10 拉取備份、核對、輪替、記錄狀態。由 launchd 每天 09:00 執行（也可手動執行）。
# 只用「唯讀專用金鑰」連線：那把金鑰在 GX10 上被限制成只能執行三個唯讀動作（latest／failed／get 名稱），
# 不能開 shell、不能執行其他指令。用 ssh＋tar 傳輸，不需要另外安裝任何東西（macOS 內建的 rsync 與 GX10 的
# rrsync 不相容，所以不用 rsync）。
#
# 設定（環境變數）：
#   GX10_BACKUP_HOST  必填，沒有預設值，例如 user@host（GX10 上被限制成強制指令的那個帳號與主機）
#   GX10_BACKUP_KEY   選填，預設 $HOME/.ssh/gx10_backup（專用金鑰檔路徑）
#   GX10_BACKUP_DEST  選填，預設 $HOME/GX10Backup（備份存放資料夾）
# 用 launchd 執行時，GX10_BACKUP_HOST 放在 plist 的 EnvironmentVariables（見同資料夾的 plist 範例與 README）。
set -u
HOST="${GX10_BACKUP_HOST:?請設定 GX10_BACKUP_HOST，例如 user@host}"
KEY="${GX10_BACKUP_KEY:-$HOME/.ssh/gx10_backup}"
DEST="${GX10_BACKUP_DEST:-$HOME/GX10Backup}"
STATUS="$DEST/STATUS.txt"
LOGF="$DEST/backup.log"
SSHCMD="${GX10_BACKUP_SSH:-ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20}"
KEEP_DAILY=14; KEEP_WEEKLY=8; KEEP_MONTHLY=6

mkdir -p "$DEST/daily" "$DEST/weekly" "$DEST/monthly" "$DEST/.incoming"
chmod 700 "$DEST" "$DEST/daily" "$DEST/weekly" "$DEST/monthly" "$DEST/.incoming"

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(now) $1" >> "$LOGF"; }
OSA="${GX10_BACKUP_OSASCRIPT:-/usr/bin/osascript}"
notify() { log "$1"; "$OSA" -e "display notification \"$1\" with title \"GX10 備份\"" >/dev/null 2>&1; }
fail() { notify "失敗：$1"; write_status "最近一次嘗試失敗（$(now)）：$1"; exit 1; }

last_success_epoch() { [ -f "$DEST/.last_success_epoch" ] && cat "$DEST/.last_success_epoch" || echo 0; }
write_status() {
  {
    echo "最近一次嘗試：$(now)"
    echo "結果：$1"
    if [ -f "$DEST/.last_success_epoch" ]; then
      echo "上次成功：$(cat "$DEST/.last_success_text" 2>/dev/null)"
    else
      echo "上次成功：從未成功"
    fi
    echo "每日 $(ls -1 "$DEST/daily" | wc -l | tr -d ' ') 份／每週 $(ls -1 "$DEST/weekly" | wc -l | tr -d ' ') 份／每月 $(ls -1 "$DEST/monthly" | wc -l | tr -d ' ') 份，總大小 $(du -sh "$DEST" 2>/dev/null | cut -f1)"
  } > "$STATUS"
}

# 0. 超過 36 小時沒成功過就先提醒（不管這次能不能成功）
AGE=$(( $(date +%s) - $(last_success_epoch) ))
if [ "$(last_success_epoch)" != 0 ] && [ "$AGE" -gt 129600 ]; then notify "已超過 36 小時沒有成功備份"; fi

# 1. GX10 端有沒有標記失敗
rm -rf "$DEST/.incoming/FAILED" "$DEST/.incoming/LATEST"
if $SSHCMD "$HOST" failed > "$DEST/.incoming/FAILED" 2>/dev/null && [ -s "$DEST/.incoming/FAILED" ]; then
  notify "GX10 端整理備份失敗：$(cat "$DEST/.incoming/FAILED")"
fi

# 2. 最新一份是哪個
$SSHCMD "$HOST" latest > "$DEST/.incoming/LATEST" 2>/dev/null && [ -s "$DEST/.incoming/LATEST" ] || fail "連不上 GX10 或讀不到最新備份的名稱"
NAME=$(tr -d '[:space:]' < "$DEST/.incoming/LATEST")
case "$NAME" in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]) ;; *) fail "最新備份名稱格式不對" ;; esac

if [ -d "$DEST/daily/$NAME" ]; then
  date +%s > "$DEST/.last_success_epoch"; echo "$(now)（已確認最新 ${NAME}）" > "$DEST/.last_success_text"
  write_status "已是最新（${NAME}），沒有新的備份可拉"; exit 0
fi

# 3. 拉取（先清掉上次遺留的半成品）
find "$DEST/.incoming" -mindepth 1 -maxdepth 1 ! -name FAILED ! -name LATEST -exec rm -rf {} + 2>/dev/null
$SSHCMD "$HOST" "get $NAME" | tar -x -C "$DEST/.incoming" 2>/dev/null
ST=("${PIPESTATUS[@]}")
[ "${ST[0]:-1}" = 0 ] && [ "${ST[1]:-1}" = 0 ] || { rm -rf "$DEST/.incoming/$NAME"; fail "拉取備份 $NAME 失敗"; }
[ -f "$DEST/.incoming/$NAME/DONE" ] || { rm -rf "$DEST/.incoming/$NAME"; fail "備份 $NAME 沒有完成標記（DONE），不採用"; }

# 4. 核對每個檔案的雜湊
( cd "$DEST/.incoming/$NAME" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ) || { rm -rf "$DEST/.incoming/$NAME"; fail "備份 $NAME 雜湊核對不符，不採用"; }

# 5. 收下、設權限
mv "$DEST/.incoming/$NAME" "$DEST/daily/$NAME"
chmod -R go-rwx "$DEST/daily/$NAME"
DOW=$(date +%u); DOM=$(date +%d)
[ "$DOW" = "7" ] && cp -R "$DEST/daily/$NAME" "$DEST/weekly/$NAME"
[ "$DOM" = "01" ] && cp -R "$DEST/daily/$NAME" "$DEST/monthly/$NAME"

# 6. 成功後才輪替（最新這份已核對通過，才會刪舊的）
prune() { ls -1 "$1" | sort -r | tail -n +"$(( $2 + 1 ))" | while read -r d; do [ -n "$d" ] && rm -rf "${1:?}/$d"; done; }
prune "$DEST/daily" "$KEEP_DAILY"; prune "$DEST/weekly" "$KEEP_WEEKLY"; prune "$DEST/monthly" "$KEEP_MONTHLY"

date +%s > "$DEST/.last_success_epoch"; echo "$(now)（備份 ${NAME}）" > "$DEST/.last_success_text"
write_status "成功：已拉取並核對 $NAME"
log "成功 $NAME"
