# Mac 端 GX10 狀態頁（gx10-console-mac）

這是 GX10 控制台階段 1 的 Mac 端：每 5 分鐘從 GX10 拉取 `status.json`，驗證、處理過期與拉取失敗，產生一個本機的靜態狀態頁
`status.html`，並在整體燈號變紅時用 macOS 通知提醒。配合 GX10 端的 `scripts/console_status.py`（產生狀態）與
`scripts/console_forced_command.py`（`gx10_console` 金鑰的強制指令，只允許 `status`、`summary`）。
**這裡所有主機、帳號、路徑都是佔位字**，請換成你自己的值。

## 它做什麼（與不做什麼）
- **燈號的唯一來源是 GX10 產生的 `status.json`。** Mac 端不重新計算任何項目的紅黃綠，只做：驗證格式、處理過期、處理拉取失敗、顯示、通知決策。
- 格式不對（欄位缺漏、未知燈號值、型別錯誤）→ 一律視為紅，頁面標示「資料格式異常」，不會顯示成綠。
- 資料產生時間距今超過 15 分鐘，或比現在晚超過 60 秒 → 整體燈號改為紅並顯示紅色橫幅（這是 Mac 端才有的判斷）。
- 拉取失敗 → 保留上一份成功的資料並標示「拉取失敗」，整體燈號至少黃；連續失敗超過 15 分鐘或資料過期 → 紅；從未成功拉取 → 「尚無資料」紅。
- 通知：只在整體燈號「進入紅色」時通知一次；持續紅色時每 6 小時最多再提醒一次；回到非紅色後重置；連續拉取失敗超過 15 分鐘也算紅色。
  通知文字只含固定詞彙與項目編號，不含任何來自 GX10 的文字。
- 頁面固定顯示「整體燈號不含防火牆規則檢查（需管理權限）」；第 12 項固定「未確認」，不判紅綠。
- 頁面是單一靜態 HTML：不載入任何外部資源（有內容安全政策）、所有資料以純文字顯示；內嵌的小程式固定且不含資料，
  只在 Mac 排程停止、頁面超過 15 分鐘沒更新時讓舊頁面自己標示過期。
- 不寫入任何東西到 GX10；唯一的連線動作是 `ssh … status`。

## 需要設定的環境變數
| 變數 | 必填 | 預設 | 說明 |
|---|---|---|---|
| `GX10_CONSOLE_HOST` | **必填，沒有預設值** | 無 | 例如 `user@host`；沒設就會直接報錯結束（結束碼 2），不會連線。不能以 `-` 開頭、只允許字母數字與 `. _ @ : -` |
| `GX10_CONSOLE_KEY` | 否 | `$HOME/.ssh/gx10_console` | 專用金鑰檔（只用來讀狀態；不要重用備份金鑰） |
| `GX10_CONSOLE_DIR` | 否 | `$HOME/GX10Console` | 輸出資料夾（權限 700，檔案 600） |
| `GX10_CONSOLE_SSH` | 否 | `/usr/bin/ssh` | ssh 的絕對路徑；必須是絕對路徑，只允許字母、數字與 `. _ / -`，不能含 `..`，不合格就報錯結束（結束碼 2） |

輸出檔：`status.html`（狀態頁，用瀏覽器開）、`last_good.json`（上一份成功且格式正確的資料）、`state.json`（通知與失敗時間狀態）、
`last_run.txt`（最近一次執行的一行結果；每次覆寫，不會無限成長）。

## 放置與手動測試
1. 把 `console_pull.py` 放到 `~/bin/console_pull.py`，`chmod 700`。
2. 手動試跑：`GX10_CONSOLE_HOST=user@host /usr/bin/python3 -B ~/bin/console_pull.py`，再用瀏覽器打開 `~/GX10Console/status.html`。
3. 只想看版面、不連線：`python3 console_pull.py --preview-dir ~/gx10-console-preview`，會產生 6 個假資料情境頁（頁面上標示「預覽（假資料）」）。

## 排程（macOS，不要在測試完成前安裝）
1. 複製 `com.psf.gx10-console.plist.example` 成 `~/Library/LaunchAgents/com.psf.gx10-console.plist`，把 `YOUR_USER` 換成你的 Mac 使用者名稱、
   `user@host` 換成實際的 GX10 帳號與主機。plist 設為每 300 秒（5 分鐘）執行一次，載入時也立刻跑一次。
2. 載入（常見指令；請依你的 macOS 版本確認）：`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.psf.gx10-console.plist`
   卸載：`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.psf.gx10-console.plist`
3. 錯誤輸出在 `/tmp/gx10-console.err`（正常情況是空的）。

## 為什麼用絕對路徑並設定 PATH
launchd 執行工作時，環境變數與你在終端機裡的完全不同：`PATH` 很小，也沒有你的 shell 設定。所以拉取指令的可執行檔用絕對路徑
`/usr/bin/ssh`（程式內的預設值，可用 `GX10_CONSOLE_SSH` 覆蓋），通知用 `/usr/bin/osascript`，plist 裡的程式用 `/usr/bin/python3`，
並在 `EnvironmentVariables` 設 `PATH` 為 `/usr/bin:/bin`（只含系統目錄，不放使用者目錄）。這樣不管從 launchd 或手動執行，找到的都是同一支程式，
也不會因為 PATH 裡多了別的目錄而執行到不預期的 `ssh`。

## 通知權限
macOS 通知是用 `osascript` 的 `display notification`。第一次通知時，系統可能要求允許「腳本編輯器」或終端機顯示通知；
請在「系統設定 → 通知」確認該應用程式允許通知，並確認沒有開啟專注模式把它擋掉。這個權限需要你自己在 Mac 上設定。

## GX10 端的金鑰限制
`gx10_console` 在 GX10 的 `~/.ssh/authorized_keys` 裡被限制成：`restrict`、`from="<Mac 的位址>"`、強制指令
`console_forced_command.py`。這把金鑰不能開 shell、不能轉發、不能傳檔，只能讀 `status` 與 `summary`（完全相同的字串，大小寫與空白都不能變）。

## 回復
- 卸載 launchd（上面的 `bootout`），刪除 `~/Library/LaunchAgents/com.psf.gx10-console.plist`、`~/bin/console_pull.py` 與 `~/GX10Console`。
- GX10 端的回復見 GX10 端的紀錄（移除 `gx10_console status-only` 那一行與 `trial/console/` 內的檔案）。

## 已知限制
- **金鑰沒有密碼短語**（給排程自動使用）；私鑰外洩者只能讀狀態與摘要，且受 `from=` 限制。建議存放在只有你能讀的位置。
- **`from=` 綁定 Mac 的位址**：Mac 的 IP 變動（例如 DHCP 換位址）時，這把金鑰會被拒絕，頁面會變成「拉取失敗」並在 15 分鐘後轉紅。
- **監控者本身壞掉**：Mac 排程停止時不會有任何通知；頁面內嵌的小程式只能讓「已開著的舊頁面」自己標示過期。階段 1 接受 Mac 單點，之後再評估第二層心跳。
- 過期判斷用兩台機器的時鐘比較，Mac 與 GX10 的時間差需小於 60 秒，否則會誤報。
- 第 12 項（防火牆規則是否實際載入）需要管理權限，固定顯示「未確認」，整體燈號不包含它。
