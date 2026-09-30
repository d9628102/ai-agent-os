#!/bin/bash
# fw-install.sh —— 安裝「開機自動載入防火牆」的檔案。只安裝；不啟動、不啟用、不載入任何規則。不含任何密碼。
#
# 用法（需要 root，由林劭瑋自己輸入 sudo 密碼）：
#   sudo /home/psf01/trial/net-hardening/boot/fw-install.sh              安裝
#   sudo /home/psf01/trial/net-hardening/boot/fw-install.sh --uninstall  移除（須先停用服務）
#
# 安裝前的檢查（任何一項失敗就停止，什麼都不安裝）：
#   ① 必要檔案都在  ② 規則檔不含 "flush ruleset"  ③ 規則檔含表 inet gx10_fw
#   ④ nft -c 語法檢查通過                                     ← 需要 root（nft）
#   ⑤ nftables.service 不是啟用狀態（它的停止動作會清空 Docker 的規則）
# 安裝的東西與各自需要 root 的原因：
#   /etc/gx10-fw/                       root:root 755   ← 需要 root（寫 /etc）
#   /etc/gx10-fw/gx10_fw.nft            root:root 644   規則檔（若已有不同內容，舊的另存 .prev-時間）
#   /etc/gx10-fw/README.txt、SOURCE.sha256（來源檔雜湊）  root:root 644
#   /etc/systemd/system/gx10-firewall.service、gx10-firewall-alert.service   root:root 644   ← 需要 root
#   /usr/local/sbin/gx10-fw-record、gx10-fw-alert   root:root 755   ← 需要 root
#   systemctl daemon-reload（讓 systemd 讀到新單元）           ← 需要 root
# 測試用：設定 DESTDIR=/某個暫存資料夾 時，所有目的地都加上這個前綴，並且不檢查 root、不改擁有者、
#         不呼叫 daemon-reload（平常不要設定）。
set -u
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
D="${DESTDIR:-}"
ETC="$D/etc/gx10-fw"; UNITDIR="$D/etc/systemd/system"; SBIN="$D/usr/local/sbin"
OWN=""; [ -z "$D" ] && OWN="-o root -g root"
die() { echo "錯誤：$1"; echo "→ 什麼都沒有安裝／變更。"; exit 1; }

[ -n "$D" ] || [ "$(id -u)" = 0 ] || die "需要 root，請用 sudo 執行本腳本"

nft_state() { systemctl is-enabled nftables.service 2>&1 | head -1; }

if [ "${1:-}" = "--uninstall" ]; then
  for u in gx10-firewall.service; do
    en="$(systemctl is-enabled $u 2>/dev/null | head -1)"; ac="$(systemctl is-active $u 2>/dev/null | head -1)"
    case "$en" in enabled|enabled-runtime) die "$u 還是啟用狀態，請先：systemctl disable --now $u";; esac
    [ "$ac" = "active" ] && die "$u 還在運作，請先：systemctl stop $u"
  done
  rm -f "$UNITDIR/gx10-firewall.service" "$UNITDIR/gx10-firewall-alert.service" "$SBIN/gx10-fw-record" "$SBIN/gx10-fw-alert"
  rm -f "$ETC/gx10_fw.nft" "$ETC/README.txt" "$ETC/SOURCE.sha256" "$ETC"/gx10_fw.nft.prev-* 2>/dev/null
  rmdir "$ETC" 2>/dev/null || echo "註：$ETC 內還有其他檔案，保留資料夾"
  [ -n "$D" ] || systemctl daemon-reload
  echo "已移除單元、腳本與 /etc/gx10-fw 內的安裝檔（沒有動任何 nft 規則）。"
  exit 0
fi

echo "== 1/5 檢查來源檔案"
for f in gx10_fw.nft README.txt gx10-firewall.service gx10-firewall-alert.service gx10-fw-record gx10-fw-alert; do
  [ -f "$HERE/$f" ] || die "找不到 $HERE/$f"
done
echo "== 2/5 檢查規則檔內容"
grep -qE 'flush[[:space:]]+ruleset' "$HERE/gx10_fw.nft" && die "規則檔含有 flush ruleset（會清空 Docker 的規則）"
grep -qE '^table inet gx10_fw[[:space:]]*\{' "$HERE/gx10_fw.nft" || die "規則檔裡找不到 table inet gx10_fw"
echo "== 3/5 nft -c 語法檢查（只檢查，不載入）"
nft -c -f "$HERE/gx10_fw.nft" || die "規則檔語法檢查失敗"
echo "== 4/5 確認 nftables.service 沒有啟用"
NS="$(nft_state)"; echo "   nftables.service 狀態：$NS"
case "$NS" in enabled|enabled-runtime|linked|alias) die "nftables.service 是啟用的：它的停止動作會執行 nft flush ruleset，請先 systemctl disable nftables.service";; esac

echo "== 5/5 安裝"
install -d $OWN -m 755 "$ETC" "$UNITDIR" "$SBIN" || die "建立資料夾失敗"
if [ -f "$ETC/gx10_fw.nft" ] && ! cmp -s "$HERE/gx10_fw.nft" "$ETC/gx10_fw.nft"; then
  PREV="$ETC/gx10_fw.nft.prev-$(date +%Y%m%d-%H%M%S)"; cp -p "$ETC/gx10_fw.nft" "$PREV" && echo "   舊規則檔另存：$PREV"
fi
install $OWN -m 644 "$HERE/gx10_fw.nft" "$ETC/gx10_fw.nft" || die "複製規則檔失敗"
install $OWN -m 644 "$HERE/README.txt" "$ETC/README.txt" || die "複製 README 失敗"
( cd "$HERE" && sha256sum gx10_fw.nft ) > "$ETC/SOURCE.sha256" && chmod 644 "$ETC/SOURCE.sha256"
[ -z "$D" ] && chown root:root "$ETC/SOURCE.sha256"
install $OWN -m 644 "$HERE/gx10-firewall.service" "$UNITDIR/gx10-firewall.service" || die "安裝服務單元失敗"
install $OWN -m 644 "$HERE/gx10-firewall-alert.service" "$UNITDIR/gx10-firewall-alert.service" || die "安裝警報單元失敗"
install $OWN -m 755 "$HERE/gx10-fw-record" "$SBIN/gx10-fw-record" || die "安裝 gx10-fw-record 失敗"
install $OWN -m 755 "$HERE/gx10-fw-alert" "$SBIN/gx10-fw-alert" || die "安裝 gx10-fw-alert 失敗"
if [ -z "$D" ]; then
  systemctl daemon-reload || echo "警告：daemon-reload 失敗"
  echo "   （單元檢查，僅供參考）"; systemd-analyze verify gx10-firewall.service gx10-firewall-alert.service 2>&1 | head -8 | sed 's/^/   /'
fi

echo
echo "== 安裝完成（沒有啟動、沒有啟用、沒有載入任何規則）。結果："
ls -l "$ETC" "$UNITDIR/gx10-firewall.service" "$UNITDIR/gx10-firewall-alert.service" "$SBIN/gx10-fw-record" "$SBIN/gx10-fw-alert" 2>&1 | awk '{print "   "$0}'
echo "   規則檔來源雜湊：$(cut -c1-16 "$ETC/SOURCE.sha256")…"
if [ -z "$D" ]; then
  echo "   gx10-firewall.service：啟用狀態 $(systemctl is-enabled gx10-firewall.service 2>&1 | head -1)、運作狀態 $(systemctl is-active gx10-firewall.service 2>&1 | head -1)"
  echo "   nftables.service：$(nft_state)（必須維持 disabled）"
  echo "   目前 nft 表：$(nft list tables 2>&1 | grep -c gx10_fw) 個 gx10_fw 開頭的表（本腳本沒有載入任何東西）"
fi
