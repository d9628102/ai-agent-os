# Mac 端 GX10 備份拉取（gx10-backup-mac）

這是「Mac 每天從 GX10 拉取備份」的腳本，配合 GX10 端的 `scripts/gx10_backup_prepare.py`（每晚整理備份）與
`scripts/gx10_backup_serve.py`（只讀的強制指令）。**這裡所有主機、帳號、路徑都是佔位字**，請換成你自己的值。

## 它做什麼
每天 09:00（launchd）連到 GX10，讀取最新備份的名稱，用 ssh＋tar 拉下來、核對 SHA256、收進 `daily/`
（週日另存 `weekly/`、每月 1 日另存 `monthly/`），保留 14 天／8 週／6 個月，並把結果寫到 `STATUS.txt`。
超過 36 小時沒有成功、或 GX10 端標記失敗時，會用 macOS 通知提醒。

## 需要設定的環境變數
| 變數 | 必填 | 預設 | 說明 |
|---|---|---|---|
| `GX10_BACKUP_HOST` | **必填，沒有預設值** | 無 | 例如 `user@host`；沒設就會直接報錯結束，不會連線 |
| `GX10_BACKUP_KEY` | 否 | `$HOME/.ssh/gx10_backup` | 專用金鑰檔（只用來拉備份） |
| `GX10_BACKUP_DEST` | 否 | `$HOME/GX10Backup` | 備份存放資料夾 |

## 放置與排程（macOS）
1. 把 `gx10-backup.sh` 放到 `~/bin/gx10-backup.sh`，`chmod 700`。
2. 複製 `com.psf.gx10-backup.plist.example` 成 `~/Library/LaunchAgents/com.psf.gx10-backup.plist`，把裡面的
   `YOUR_USER` 換成你的 Mac 使用者名稱、`user@host` 換成實際的 GX10 帳號與主機。
3. 載入排程（常見指令；請依你的 macOS 版本確認）：
   `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.psf.gx10-backup.plist`
   卸載：`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.psf.gx10-backup.plist`
   plist 設了 `RunAtLoad`，載入時也會立刻跑一次。
4. 手動試跑：`GX10_BACKUP_HOST=user@host ~/bin/gx10-backup.sh`，再看 `$GX10_BACKUP_DEST/STATUS.txt`。
   錯誤輸出在 `/tmp/gx10-backup.err`。

## GX10 端的金鑰限制
Mac 用的專用金鑰，在 GX10 的 `~/.ssh/authorized_keys` 裡被限制成**強制指令**
`command="/usr/bin/python3 <GX10 上的路徑>/gx10_backup_serve.py"`。因此這把金鑰不能開 shell，只能做三件事：
`latest`（印出最新備份名稱）、`failed`（印出失敗標記）、`get <YYYYMMDD-HHMMSS>`（把已完成〔有 DONE〕的備份打成 tar 串流）。
詳細請看 `gx10_backup_serve.py` 檔頭的說明。macOS 內建的 rsync（openrsync）與 rrsync 不相容，所以不用 rsync。

## 注意
- 備份內含問答全文與 n8n 憑證加密金鑰檔，Mac 上的備份資料夾權限是 700，請不要放到雲端同步資料夾。
- 這支腳本不會刪除 GX10 上的任何東西；它只在 Mac 本機輪替（刪除超過保留數量的舊備份）。
