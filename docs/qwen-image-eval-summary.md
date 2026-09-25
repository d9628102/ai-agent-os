# Qwen-Image-2.1 可行性評估摘要(已結案:在目前前提下不可行)

> **評估用、非商用**:Qwen-Image-2.1 依 Qwen Research License 僅限非商業
> 用途,Unsloth FP8 為社群衍生版本,同受此限制。評估產出一律不可用於客戶
> 交付物。

評估 Qwen-Image-2.1(7B DiT,文字編碼器為 Qwen3-VL)能不能在 GX10 上跟
正式問答服務(vllm-server + vllm-embed + Qdrant + n8n)共存,做為設計
工廠的候選。完整報告與原始資料留在 GX10 的
`/home/psf01/eval/qwen-image-2.1-EVAL-NONCOMMERCIAL/`,不進版控。

## 結論

**決策(2026-09-26,林劭瑋)**:停止評估,結論定為「在目前前提下不可行」。
方向 B(調低 vllm 記憶體預留)、C(分時切換)都不核准。

1. **授權**:限非商業用途,技術上可行也不能用在客戶交付物。
2. **代價**:每次評估都讓正式服務付出代價,包括 swap 累積(兩晚各增加約
   3~4.4 GiB,不會自動回收)以及問答延遲加倍。

兩晚都沒有產出任何一張圖,所以**生成品質(包括繁體中文字形)沒有評估到**。
「不可行」是這台機器、這組前提下跑不起來,不是對模型品質的結論。

## 重新評估的觸發條件(滿足任一項)

- **(a)** 取得商業授權,而且有明確的海報業務需求。
- **(b)** 加裝第二台 GX10,讓圖像生成不跟問答服務共用 GPU 和記憶體。

## 關鍵發現:問答延遲加倍,跟記憶體設定無關

第一晚去噪期間,vllm-server 的問答延遲從基準 0.84~0.99 秒**加倍到約
2.0 秒**,去噪一停就回到 0.91 秒。兩個工作共用同一顆 GB10 的計算資源,
這個代價不是調記憶體預留或量化能改變的。所以就算未來記憶體夠用,在同一台
機器上圖像生成也只能當離峰批次工作。

## 兩晚的起始條件與結果

保護門檻兩晚相同:MemAvailable < 4 GiB,或 swap 比階段開始時增長 > 4 GiB,
就停掉評估容器(`--oom-score-adj=1000`)。

| 項目 | 第一晚(09-25 01:00) | 第二晚(09-26 01:01) |
|---|---|---|
| 起始 MemAvailable | 25.19 GiB | 20.75 GiB |
| 起始 swap 已使用 | 5.18 GiB | 0.10 GiB(前一天做過 swap 清理) |
| 起始 page cache | 約 23 GiB | 2.2 GiB |
| 權重讀法 | mmap | streaming + `posix_fadvise(DONTNEED)` |
| 走到哪一步 | 去噪完第一張圖,VAE 解碼時觸發 | 載入文字編碼器第 10 秒觸發 |
| 觸發原因 | swap 增長 4.06 GiB | swap 增長 4.37 GiB |
| MemAvailable 最低 | 10.6 GiB | 13.87 GiB |
| swap(結束後) | 8.1 GiB | 4.49 GiB |
| 產出圖片 | 0 | 0 |

判讀:兩晚都是 swap 門檻先觸發,MemAvailable 門檻從沒接近過。第一晚有約
23 GiB page cache 可以先丟,撐到去噪結束;第二晚 page cache 只剩 2.2 GiB,
kernel 一開始配置權重就把正式服務的匿名記憶體換出(推論,未驗證)。
MemAvailable 可能高估的問題見
[`gx10-known-issues.md`](gx10-known-issues.md) 問題 8。

第一晚各階段(1024²,分階段載入):pipeline 外殼 + VAE 6 秒;FP8 文字編碼器
以 mmap 載入 191 秒(約 49 MB/s,磁碟循序讀是 5.1 GB/s,慢在記憶體壓力下的
page fault 小塊讀取);編碼 8 個 prompt 7.7 秒;FP8 transformer 載入 40 秒;
去噪 1.75 秒/步。

## 保留的資產

條件成立時可以直接重用,不用重新下載(合計約 43 GB):

| 項目 | 位置 | 大小 |
|---|---|---|
| 評估容器映像 | `qwen-image-eval-noncommercial:26.08`(基底 `nvcr.io/nvidia/pytorch:26.08-py3`,共用層) | 25.9 GB |
| 權重 | 上述 eval 目錄下 `models/`(官方設定 + VAE、Unsloth FP8 transformer 與文字編碼器) | 17 GB |
| 腳本、題目、原始資料 | 同目錄 `*.py`、`run_night.sh`、`jobs/`、`runs/`、`logs/` | 小 |

重新啟用前要確認:

- 驅動仍是 595.84。**禁止升級 GPU 驅動**(Secure Boot 的 MOK 註冊需要人到
  現場),容器不相容時只能退回 25.10~26.05 的映像。
- [`gx10-known-issues.md`](gx10-known-issues.md) 問題 9 的 torchao 警告有沒有影響。
- 繁體中文字形的品質驗證仍須人工評分,OCR 只能當輔助。

## 未採用 / 另案處理

- 調高看門狗 swap 門檻、調整 `vm.swappiness`:不核准。
- vllm-server 加 `--safetensors-load-strategy eager`(縮短重啟載入時間):
  跟圖像評估無關,另外提案。
- 向 Qwen 申請商業授權:暫不寄,等真的有海報需求再說。
