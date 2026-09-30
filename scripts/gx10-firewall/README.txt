GX10 防火牆說明（白話版）
=========================
這個資料夾（/etc/gx10-fw/）放的是 GX10 的防火牆規則。請「不要手動修改」這裡的檔案，
更新規則一律走下面的「更新規則」流程。

它是什麼
--------
一張 nftables 規則表，名稱 gx10_fw，開機時由 gx10-firewall.service 自動載入：
  - SSH（22）：只允許本機、區網（192.168.1.0/24）與 Docker 內部（172.17.0.0/16，n8n 需要）。
    IPv6 只允許本機（::1）與鏈路本地（fe80::/10）。其他來源一律擋掉。
  - 大型模型 8000、向量模型 8001、Qdrant 6333／6334：同樣只允許區網（IPv4 含 Docker 轉送路徑）。
  - 5678（n8n）目前沒有納入。
  - 已建立的連線一律放行，所以現有的連線不會被切斷。
它獨立於 Docker：Docker 不會刪它，它也不動 Docker 的規則。

失敗時
------
規則載入失敗時，「不會」全部擋住（沒有防火牆＝放行），而且會大聲報警：
  - 系統日誌有一筆 crit：  journalctl -b -u gx10-firewall.service
  - 旗標檔 /run/gx10-fw-FAILED 存在（成功載入後會被移除）
成功載入時會在 /var/log/gx10-fw-alert.log 與系統日誌寫一筆紀錄（時間與表雜湊前 12 碼）。

看目前狀態
----------
  systemctl status gx10-firewall.service          （成功時是 active (exited)）
  sudo nft list table inet gx10_fw               （看規則與被擋的計數）
  tail /var/log/gx10-fw-alert.log
  ls /run/gx10-fw-FAILED                          （不存在＝正常）

暫時關閉（不影響 Docker）
------------------------
  sudo systemctl stop gx10-firewall.service       （立刻刪掉表；重開機或 start 後會回來）
  重新開啟：sudo systemctl start gx10-firewall.service
  或直接立刻刪表：sudo /home/psf01/trial/net-hardening/fw-remove.sh gx10_fw

永久停用
--------
  sudo systemctl disable --now gx10-firewall.service
（之後重開機不會再載入。要完全移除檔案：sudo ./fw-install.sh --uninstall，須先停用。）

更新規則（照順序，不要跳步）
----------------------------
  1. 在草稿資料夾（/home/psf01/trial/net-hardening/boot/）改好新的 gx10_fw.nft。
  2. 臨時載入並先排定 10 分鐘自動還原：
       sudo /home/psf01/trial/net-hardening/fw-apply.sh <新規則檔> 600
  3. 驗證：Mac 新開 SSH 連線、備份拉取、8000 可連、表單送一題、
       /home/psf01/trial/net-hardening/fw-blocktest.sh step2 …… 全部通過。
  4. 取消自動還原：sudo /home/psf01/trial/net-hardening/fw-cancel-revert.sh gx10_fw
  5. 安裝到這裡：sudo /home/psf01/trial/net-hardening/boot/fw-install.sh
  6. 重新載入：sudo systemctl reload gx10-firewall.service
     （用 reload，不要用 restart：restart 會先刪表再載入，中間有空窗。）
  7. 確認：sudo nft list table inet gx10_fw 與預期一致。

路由器改了網段，SSH 被擋（例如 192.168.1.x 變成 192.168.0.x）
-----------------------------------------------------------
遠端連不上時，只能在 GX10 本機處理：
  1. 接鍵盤與螢幕，在本機登入。
  2. 立刻恢復連線：  sudo /home/psf01/trial/net-hardening/fw-remove.sh gx10_fw
     （刪表；重開機或重新 start 服務之前不會回來）
  3. 若要永久停用：  sudo systemctl disable --now gx10-firewall.service
  4. 把規則檔裡 lan4 的網段改成新的（草稿：/home/psf01/trial/net-hardening/boot/gx10_fw.nft），
     再照「更新規則」流程安裝與載入。
沒有鍵盤螢幕時沒有遠端恢復的路，所以強烈建議在路由器為 GX10 設定 DHCP 保留位址。

不要做的事
----------
  - 不要啟用 nftables.service：它的停止動作是「nft flush ruleset」，會清空 Docker 的所有規則。
  - 不要執行「ufw enable」。
  - 不要手動改 /etc/gx10-fw/ 裡的檔案；不要用 nft flush ruleset。
