#!/bin/bash
# fw-blocktest.sh（2026-09-30 加入 5678 與 step3 模式）—「確實會擋」的測試（不需要 sudo，不修改防火牆）
#
# 它做什麼（白話）：
#   防火牆規則的驗證通常只能證明「該通的有通」。這支腳本補上另一半：「不該通的有被擋」。
#   做法：建立一個暫時的 Docker 網路（網段 172.30.0.0/24，不在允許清單 127.0.0.0/8、192.168.1.0/24、
#   172.17.0.0/16 之內），在裡面開一個暫時容器，從容器連 GX10（192.168.1.128）的 22、8000、6333、8001、5678（n8n），
#   每個埠最多等 4 秒。連得上＝「連得上」；等到逾時＝「被擋（逾時）」；立刻失敗＝「連線被拒（立刻失敗）」。
#   這同時實測步驟 2 的「轉送鏈」規則（從別的 Docker 網路連過來，會被 Docker 轉送到容器）。
#   結束時（不管成功、失敗或被中斷）一定會刪掉暫時容器與暫時網路（trap）。
#
# 用法：  fw-blocktest.sh [baseline|step1|step2|step3]
#   baseline（防火牆還沒載入）  預期：22、8000、6333、8001、5678 全部「連得上」
#   step1（載入步驟 1 之後）     預期：22 被擋；8000、6333、8001、5678 連得上
#   step2（載入步驟 2 之後，5678 尚未納入） 預期：22、8000、6333、8001 被擋；5678 連得上
#   step3（5678 也納入之後）     預期：五個埠（22、8000、6333、8001、5678）都被擋
#   （注意：6333、8001 現在只綁在本機，區網位址上沒有服務在聽；沒有防火牆時會顯示「連線被拒（立刻失敗）」，
#    所以 baseline 模式在容器已改成只開本機之後，這兩個埠不會是「連得上」——baseline 只適用於當時的舊狀態。）
#   不帶參數：只顯示結果，不比對預期。
# 需要的權限：docker 群組（psf01 已在）。不需要 sudo。不下載任何映像檔（只用 GX10 上已有的）。
# 不含任何密碼或 token。
set -u
MODE="${1:-}"
NET="gx10-fwtest-net"; CT="gx10-fwtest-ct"; SUBNET="172.30.0.0/24"
TARGET="192.168.1.128"; PORTS="22 8000 6333 8001 5678"; TIMEOUT=4
IMG="nvidia/cuda:12.4.0-base-ubuntu22.04"      # GX10 上已有；有 bash 與 timeout，用 bash 內建的 /dev/tcp 測連線，不需要 nc

cleanup() { docker rm -f "$CT" >/dev/null 2>&1; docker network rm "$NET" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM
cleanup   # 先清掉上次遺留（若有）

case "$MODE" in ""|baseline|step1|step2|step3) ;; *) echo "用法：$0 [baseline|step1|step2|step3]"; exit 2;; esac
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "錯誤：GX10 上找不到映像檔 $IMG（本腳本不會下載）"; exit 2; }
docker network inspect "$NET" >/dev/null 2>&1 && { echo "錯誤：暫時網路 $NET 清不掉"; exit 2; }
docker network create --driver bridge --subnet "$SUBNET" "$NET" >/dev/null || { echo "錯誤：建立暫時網路失敗（網段 $SUBNET 可能與現有網路衝突）"; exit 2; }
docker run -d --pull never --name "$CT" --network "$NET" "$IMG" sleep 120 >/dev/null || { echo "錯誤：啟動暫時容器失敗"; exit 2; }
SRC=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$NET\").IPAddress}}" "$CT")
echo "暫時容器來源位址：$SRC（網段 $SUBNET；不在允許清單內）→ 目標 $TARGET，每個埠最多等 ${TIMEOUT} 秒"

declare -A RESULT
for P in $PORTS; do
  START=$(date +%s.%N)
  docker exec "$CT" timeout "$TIMEOUT" bash -c "exec 3<>/dev/tcp/$TARGET/$P" >/dev/null 2>&1
  RC=$?
  SECS=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
  case "$RC" in
    0)   RESULT[$P]="連得上";               echo "  埠 $P：連得上（${SECS} 秒）";;
    124) RESULT[$P]="被擋（逾時）";         echo "  埠 $P：被擋（逾時，等滿 ${TIMEOUT} 秒）";;
    *)   RESULT[$P]="連線被拒（立刻失敗）"; echo "  埠 $P：連線被拒（立刻失敗，${SECS} 秒，代碼 $RC）";;
  esac
done

case "$MODE" in
  baseline) EXPECT_OPEN="22 8000 6333 8001 5678"; EXPECT_BLOCK="";;
  step1)    EXPECT_OPEN="8000 6333 8001 5678";    EXPECT_BLOCK="22";;
  step2)    EXPECT_OPEN="5678";                   EXPECT_BLOCK="22 8000 6333 8001";;
  step3)    EXPECT_OPEN="";                       EXPECT_BLOCK="22 8000 6333 8001 5678";;
  *)        echo "（未指定預期，僅顯示結果）"; exit 0;;
esac
FAIL=0
for P in $EXPECT_OPEN;  do [ "${RESULT[$P]}" = "連得上" ] || { echo "不符預期：埠 $P 應該連得上，實際：${RESULT[$P]}"; FAIL=1; }; done
for P in $EXPECT_BLOCK; do [ "${RESULT[$P]}" = "被擋（逾時）" ] || { echo "不符預期：埠 $P 應該被擋（逾時），實際：${RESULT[$P]}"; FAIL=1; }; done
[ "$FAIL" = 0 ] && echo "結果：符合 $MODE 的預期" || echo "結果：不符合 $MODE 的預期（請把整段畫面貼給 Claude）"
exit $FAIL
