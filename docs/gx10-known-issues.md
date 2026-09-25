# GX10 / DGX Spark 已知問題與解法

在 ASUS Ascent GX10(NVIDIA GB10 Grace Blackwell, ARM64/SM_121)上用
`scripts/gx10-vllm-setup.sh` 建置 vLLM 推理環境時遇到的問題記錄。第二台
GX10 建置前建議先看過這份文件。

## 1. NGC 官方 vLLM 映像要求的驅動版本高於出廠預裝版本

**現象**:`nvcr.io/nvidia/vllm:26.05-py3` 要求驅動 **595.58+**,但 GX10
出廠預裝的驅動可能只有 **580.x**(例如 580.159.03)。版本不足時容器啟動
會失敗或報驅動不相容的錯誤。

**解法**:先升級驅動再拉取/啟動映像:

```bash
sudo apt update
sudo apt install nvidia-driver-595-open
sudo reboot
```

重開機後用 `nvidia-smi` 確認 `Driver Version` 已經 ≥ 595.58,再繼續跑
vLLM 容器。

`scripts/gx10-vllm-setup.sh` 現在會在 Step 3 開始前主動檢查驅動版本,
版本不足會直接印出警告與升級指令並中止,不會等到 container 啟動失敗才
發現(見本文件同目錄下腳本內 `check_driver_version` 函式)。

## 2. Secure Boot 啟用時,升級後的驅動核心模組被拒絕載入

**現象**:裝完 `nvidia-driver-595-open` 並重開機後,`nvidia-smi` 找不到
GPU,`dmesg` 或 `modprobe nvidia` 會出現:

```
modprobe: ERROR: could not insert 'nvidia': Key was rejected by service
```

原因是 GX10 預設啟用 **Secure Boot**,而新裝的驅動核心模組是用系統自動
產生的 MOK(Machine Owner Key)簽署的,這把 key 還沒有被韌體信任,所以
核心拒絕載入。

**解法**:匯入 MOK,並在下次重開機時於**實體螢幕**上手動完成註冊流程
(這一步**無法透過 SSH 遠端完成**,必須有實體螢幕或 IPMI/KVM 等同等
console 存取):

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
```

執行後會要求設定一組**一次性密碼**(僅用於下一次開機時驗證,設定完後
即失效)。接著:

```bash
sudo reboot
```

重開機時,在藍色的 **MOK Manager** 畫面(而不是一般 OS 登入畫面)手動
操作:

1. **Enroll MOK**
2. **Continue**
3. **Yes**
4. 輸入剛剛設定的一次性密碼
5. **Reboot**

完成後系統會正常開機,此時 `nvidia-smi` 應該能正常列出 GPU 且驅動模組
成功載入。

> 提醒:如果是透過 SSH 對 GX10 做無人值守安裝,升級驅動這一步會讓機器
> 在下次重開機時卡在 MOK Manager 畫面等待實體操作,**遠端連線在這段
> 期間會失去反應**,不是當機。務必安排有人能在現場(或透過 KVM)完成
> 這五個步驟後,遠端連線才會恢復。

## 3. 未設定 `HF_TOKEN` 導致限速,且模型權重接近可用 RAM 上限拖慢載入

**現象**:啟動 `Qwen/Qwen3-30B-A3B` 時沒有設定 `HF_TOKEN`,對 Hugging
Face Hub 的下載請求被**限速**;同時該模型權重約 **56.87GB**,非常接近
這台機器當時可用 RAM 上限(**57GB**),導致 vLLM/HF 的
**auto-prefetch 被自動關閉**(記憶體餘裕不足以預先預取下一批權重檔),
整體模型載入耗時達 **729 秒**(約 12 分鐘)。

**解法 / 後續建議**:

1. **一定要設定 `HF_TOKEN`**,可大幅降低被限速的機率:
   ```bash
   HF_TOKEN=hf_xxxxxxxx sudo -E ./scripts/gx10-vllm-setup.sh
   ```
2. 重新評估這台機器留給模型/系統的**記憶體餘裕**是否足夠 —— 56.87GB
   權重 vs. 57GB 可用 RAM 幾乎沒有緩衝空間,任何其他常駐服務都可能導致
   OOM。可考慮:
   - 調高 `GPU_MEM_UTIL` 前,先確認系統背景服務(尤其是其他容器)不會
     佔用過多統一記憶體;
   - 若要在同一台機器上疊加跑多個模型或之後上 gpt-oss-120B,評估改用
     **量化版本**(AWQ / FP8 / NVFP4,依相容性)以降低權重佔用,換取
     auto-prefetch 能重新開啟、縮短載入時間。

## 4. RAG embedding 服務(BGE-M3)選擇用 vLLM 而非 sentence-transformers

**背景**:Step 4 需要另外起一個 embedding 服務(BGE-M3)給 Qdrant 用。
原本考慮的兩個方案是 vLLM 的 embedding 端點,或直接用
`sentence-transformers` 跑。

**查證結果**:在 GB10/ARM64(aarch64 + Blackwell)上,直接用一般 pip 版
`sentence-transformers`/`transformers` 常會撞到已知相容性問題 —— 官方
transformers 函式庫需要手動 patch 才能在 GB10 上正確載入/推論,常見錯誤
是缺少 `kernels-community/vllm-flash-attn3` 導致的 `FileNotFoundError`/
`KeyError`,原因是一般 pip 版 PyTorch 沒有針對 Blackwell 做編譯優化。

**解法**:既然 Step 3 已經驗證過 NGC 官方 vLLM 映像
(`nvcr.io/nvidia/vllm:26.05-py3`)在這台機器上能正常運作,直接**用同一個
映像再起一個容器**,用 `vllm serve BAAI/bge-m3 --runner pooling` 模式提供
OpenAI 相容的 `/v1/embeddings` 端點,完全繞開上述 ARM64 相容性地雷,不用
再驗證一套新的 Python 環境。BGE-M3 本身只有約 568M 參數(fp16 權重約
1.1GB),只需要很小的 `--gpu-memory-utilization` 即可,不會跟主要的
Qwen3-30B-A3B 服務搶記憶體(實際踩到的記憶體衝突見問題 5)。

見 [`scripts/gx10-rag-setup.sh`](../scripts/gx10-rag-setup.sh)。

## 5. 用 `--task embed` 啟動 embedding 服務失敗:`unrecognized arguments`

**現象**:`vllm serve BAAI/bge-m3 --task embed` 在這台機器的 vLLM
版本(0.20.1,NGC 26.05 映像)直接報錯:

```
vllm: error: unrecognized arguments: --task embed
```

且因為容器設了 `--restart unless-stopped`,會無限重試崩潰,同一段錯誤
在 log 裡重複洗版。

**原因**:vLLM 已把 `--task` 參數廢棄,serve embedding/pooling 類模型
改用 `--runner pooling`。

**解法**:改用 `vllm serve BAAI/bge-m3 --runner pooling`(已更新進
`scripts/gx10-rag-setup.sh`)。若容器已經因為舊參數卡在 crash loop,
腳本會偵測 `RestartCount >= 3` 自動 `docker rm -f` 強制重建,不需要
手動介入;真的要手動處理的話:`docker rm -f vllm-embed` 後再重跑腳本。

## 6. 主要 vLLM 服務把記憶體整包保留,導致第二個 GPU 服務啟動失敗

**現象**:修好問題 5 之後,`vllm-embed` 容器改成正確的
`--runner pooling` 啟動,卻在初始化時報錯:

```
ValueError: Free memory on device cuda:0 (2.6/121.63 GiB) on startup
is less than desired GPU memory utilization (0.08, 9.73 GiB)
```

GB10 總共 121.63 GiB 統一記憶體,啟動當下卻只剩 2.6 GiB 可用。

**原因**:`--gpu-memory-utilization` 是「啟動時就把這個比例的**總記憶體**
整包預先保留」,不是只保留模型權重實際大小。主要服務(`vllm-server`)
原本預設 `GPU_MEM_UTIL=0.90`,代表一啟動就跟系統要走
121.63 × 0.90 ≈ **109.5GB**,即使 Qwen3-30B-A3B 權重只有約 58.5GB,
其餘保留給 KV cache 池,導致同一張卡上幾乎沒有空間再開第二個 GPU 服務。

另外也發現這台機器上 host 層級的 `nvidia-smi --query-gpu=memory.used,
memory.total,memory.free` 會**整組回傳 `[N/A]`**,不只 `memory.free`。
因此 `gx10-rag-setup.sh` 裡事前的記憶體 headroom 檢查在這台機器上形同
略過 —— 真正有效的把關是容器啟動時 vLLM 自己丟出的錯誤,腳本靠監看
`docker logs` 抓錯誤訊息才攔下這個問題(而不是靜默放行)。

**解法**:

1. 降低主要服務的 `GPU_MEM_UTIL` 並重啟,騰出空間給其他 GPU 服務:
   ```bash
   docker rm -f vllm-server
   GPU_MEM_UTIL=0.75 sudo -E ./scripts/gx10-vllm-setup.sh
   ```
   `scripts/gx10-vllm-setup.sh` 的預設值已經從 0.90 調降為 **0.75**
   (≈91.2GB,仍保留充足 KV cache 空間),讓第二台 GX10 從一開始就有
   足夠 headroom,不會重演這個問題。
2. `scripts/gx10-rag-setup.sh` 的 `EMBED_GPU_MEM_UTIL` 預設也從 0.08
   調降為 **0.03**(≈3.65GB,對 BGE-M3 這種小模型綽綽有餘)。
3. 兩個服務的 `--gpu-memory-utilization` 加總必須留有餘裕(不能接近
   1.0),因為兩者都是各自對「總量」的保留,不是對「剩餘量」的保留。

## 7. HF_TOKEN 含非 ASCII 字元導致 vLLM 啟動時 crash-loop

**現象**:設定好 `HF_TOKEN`(見問題 3 的解法)後,`vllm-embed` 容器不斷
restart(`docker ps` 顯示 `Up 1 second` 但 `Created` 是好幾分鐘前),
`docker logs` 裡看到:

```
UnicodeEncodeError: 'ascii' codec can't encode characters in position 7-10: ordinal not in range(128)
```

同時對該服務的所有 API 呼叫都得到 `ConnectionResetError` /
`curl: Recv failure: 連線被對方重設`,因為容器根本沒有真正啟動起來過。

**原因**:`HF_TOKEN` 裡混進了看不見的非 ASCII 字元(全形空格、智慧引號、
零寬字元等),通常是複製貼上時夾帶的。vLLM 用這個 token 組
`Authorization: Bearer <token>` HTTP header 時,Python 的 httpx 函式庫
預設用 ASCII 編碼 header 值,遇到非 ASCII 字元直接丟
`UnicodeEncodeError`,整個程序在下載模型 config 這一步就崩潰。錯誤位置
「position 7-10」不是巧合——`"Bearer "` 剛好是 7 個字元,所以壞字元就在
token 開頭附近。

**解法**:兩支腳本的 `load_hf_token()` 現在會在使用 token 前自動:

1. 去除頭尾空白、CRLF、外層引號
2. 用 Python 逐字元檢查是否為 ASCII(**不是**用 bash 的
   `[[:ascii:]]` — 這個 POSIX character class 在部分 locale 下不可靠,
   實測連純 ASCII 字串都會誤判為不符合,已改用 Python 檢查)
3. 一旦發現非 ASCII 字元,**在容器啟動前**就 `die`,並精確印出是第幾個
   字元、對應的 Unicode 碼位(例如 `第 1 個字元: '　' (U+3000)`),
   不會再讓這個問題一路埋到 vLLM 內部才用一段難懂的 Python traceback
   表現出來

同時修掉一個相關的 CRLF 問題:若 `~/.config/gx10-llm/env` 是用會存成
Windows 換行(`\r\n`)的編輯器建立的,原本直接 `source` 這個檔案會先因為
`$'\r': command not found` 而失敗,連 HF_TOKEN 都讀不到。現在會先用
`tr -d '\r'` 過濾掉 `\r` 再 source。

**如果你也踩到這個錯誤**,建議重新寫入 token 檔案,避免用會自動排版/
轉換引號的編輯器或聊天視窗複製貼上:
```bash
printf '%s' 'HF_TOKEN=hf_你的真實token' > ~/.config/gx10-llm/env
chmod 600 ~/.config/gx10-llm/env
```

## 8. MemAvailable 可能高估約 8 GiB,真正能不靠 swap 取得的記憶體比較少

**狀態**:已記錄,尚未追查。

**現象**:Qwen-Image-2.1 評估(見
[`qwen-image-eval-summary.md`](qwen-image-eval-summary.md))第二晚結束後
(2026-09-26 01:03),`/proc/meminfo` 顯示:

| 欄位 | 數值 |
|---|---|
| MemAvailable | 25.1 GiB |
| MemFree | 14.0 GiB |
| Cached | 2.7 GiB |
| SReclaimable | 0.5 GiB |
| AnonPages | 3.9 GiB |

MemFree + Cached + SReclaimable 約 **17 GiB**,比 MemAvailable 少約 **8 GiB**,
差額的來源還沒查清楚。另外 `free` 顯示 used 約 96 GiB,但 AnonPages 只有
3.9 GiB,大部分是 vLLM 透過驅動預先保留的統一記憶體(見問題 6),不算在
一般的匿名頁裡。

**影響**:兩晚的評估都是 swap 門檻(增長 > 4 GiB)先觸發,MemAvailable
門檻(< 4 GiB)從沒接近過,最低還有 10.6 / 13.87 GiB。以 MemAvailable 規劃
記憶體,可能會高估能用的空間;而且在 page cache 很少時,kernel 會直接把
vLLM、n8n 的記憶體換出到 swap,即使 MemAvailable 看起來還很充裕。

**規劃時的做法**(追查前的保守做法):

- 估算可用空間時看 MemFree + Cached + SReclaimable,不要只看 MemAvailable。
- 監控或保護門檻要同時看 swap 增長,不能只看 MemAvailable。
- 這只是一次量測,追查時要在不同負載下多量幾次,並確認差額是不是跟 GPU
  驅動的保留記憶體有關。

## 9. Qwen-Image 評估容器出現 `Unable to import torchao Tensor objects` 警告

**狀態**:已記錄,影響未確認,尚未追查。

**現象**:評估容器(`qwen-image-eval-noncommercial:26.08`,基底 NGC PyTorch
26.08,torchao 用映像內建的版本)啟動時,log 有一行:

```
Unable to import `torchao` Tensor objects. This may affect loading checkpoints serialized with `torchao`
```

**可能的影響**:Unsloth FP8 transformer 權重是用 torchao Float8Tensor 格式
序列化的,評估腳本用 torchao 的 `unflatten_tensor_state_dict` 還原。建映像
時的 smoke test 確認過 torchao Float8 可用、零記憶體的 key/形狀檢查也通過,
但這個警告來自 diffusers/transformers 的載入路徑,兩者差異沒查。兩晚評估都
沒走到 transformer 載入就停止,所以**沒有實際驗證過** FP8 transformer 能
正確還原。

**未來重新評估前**:先查警告來自哪個套件、哪個 import 失敗,再確認
transformer 的 FP8 權重還原結果正確。

## 相關檔案

- 建置腳本:[`scripts/gx10-vllm-setup.sh`](../scripts/gx10-vllm-setup.sh)
  — 已內建 Step 3 前的驅動版本檢查(對應本文件問題 1),`GPU_MEM_UTIL`
  預設 0.75(對應問題 6),HF_TOKEN 載入前會做 ASCII 驗證(對應問題 7)。
- RAG 基礎設施腳本:[`scripts/gx10-rag-setup.sh`](../scripts/gx10-rag-setup.sh)
  + [`scripts/rag_smoke_test.py`](../scripts/rag_smoke_test.py)
  — Qdrant + BGE-M3 embedding(`--runner pooling`,對應問題 5)+
  collection 建立/寫入/語意搜尋/持久化驗證(對應本文件問題 4、6)。
- Qwen-Image-2.1 評估摘要:[`qwen-image-eval-summary.md`](qwen-image-eval-summary.md)
  (對應本文件問題 8、9;完整原始資料在 GX10 的 eval 目錄,不進版控)。
