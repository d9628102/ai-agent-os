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
映像再起一個容器**,用 `vllm serve BAAI/bge-m3 --task embed` 模式提供
OpenAI 相容的 `/v1/embeddings` 端點,完全繞開上述 ARM64 相容性地雷,不用
再驗證一套新的 Python 環境。BGE-M3 本身只有約 568M 參數(fp16 權重約
1.1GB),用很小的 `--gpu-memory-utilization`(預設 0.08)即可,不會跟主要
的 Qwen3-30B-A3B 服務搶記憶體。

見 [`scripts/gx10-rag-setup.sh`](../scripts/gx10-rag-setup.sh)。

## 相關檔案

- 建置腳本:[`scripts/gx10-vllm-setup.sh`](../scripts/gx10-vllm-setup.sh)
  — 已內建 Step 3 前的驅動版本檢查(對應本文件問題 1)。
- RAG 基礎設施腳本:[`scripts/gx10-rag-setup.sh`](../scripts/gx10-rag-setup.sh)
  + [`scripts/rag_smoke_test.py`](../scripts/rag_smoke_test.py)
  — Qdrant + BGE-M3 embedding + collection 建立/寫入/語意搜尋/持久化驗證
  (對應本文件問題 4)。
