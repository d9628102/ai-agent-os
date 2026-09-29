# n8n 流程匯出（版本紀錄）

正式的 n8n 流程只存在 n8n 的資料庫裡，不在版控中。這個資料夾放它們的**去識別化匯出檔**，
讓之後修改流程時看得到 diff。**這裡的檔案是紀錄，不是部署來源**：真正生效的是 n8n 裡的流程。

| 檔案 | 流程 | 用途 |
|---|---|---|
| `workflows/psf-eim-rag-qa-form.json` | My workflow | 「PSF EIM 問答」表單：呼叫 `rag_answer.py`、品質檢查、送執行資料到 Langfuse |
| `workflows/psf-eim-qa-approval.json` | PSF EIM QA Approval | 核准網址 `/webhook/qa-approve`：把核准／駁回的決定送到 Langfuse |

## Langfuse 只收執行資料

- 問題、文件內容、檢索片段、回答內容、評審理由、核准備註一律**不送** Langfuse。
- 送出的欄位有白名單：耗時、輸出 token 數、模型名稱、腳本名稱、成功狀態、品質分數（只有數字）、
  以及核准決定／核准人／時間。
- 問題與回答只留在 n8n 本機的執行紀錄。表單問答的執行紀錄狀態是 `waiting`，n8n 的自動清理會跳過這個狀態，
  所以不會被清掉。
- `tests/test_langfuse_payloads.py` 會逐項檢查這裡的檔案（白名單、憑證、頂層欄位），接在 pre-push 閘門。
  這是靜態檢查；執行時實際組出的內容，用金絲雀字串另外驗證。

## 匯出時的去識別化規則

從 n8n 匯出後，放進 repo 之前一定要：

1. 拿掉 `shared`（裡面有使用者的 email）以及 `activeVersionId`、`versionId`、`versionCounter`、`triggerCount`、
   `createdAt`、`updatedAt`、`versionMetadata`、`sourceWorkflowId`、`nodeGroups`、`isArchived`、`staticData`、`description`。
2. 憑證只留名稱和 id（n8n 的匯出本來就不含憑證內容）。
3. 掃描確認沒有金鑰、token、密碼、email。測試會擋。

備份與回復：修改前的原始匯出在 GX10 的 `/home/psf01/trial/n8n-backup/`（不在 repo 內）。
