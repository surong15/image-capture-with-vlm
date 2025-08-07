# VLM + Milvus 向量資料庫整合系統

一個整合視覺語言模型（Vision Language Model）與 Milvus 向量資料庫的桌面應用程式，支援圖像問答、對話歷史管理、向量搜尋等功能。

## 🌟 主要功能

### 核心功能
- **即時圖像問答**：使用相機拍攝照片，透過 AI 模型進行圖像理解和問答
- **多模型支援**：可選擇本機已安裝的 Ollama 模型（支援圖像輸入的模型）
- **向量資料庫儲存**：將對話和圖像直接儲存在 Milvus 向量資料庫中
- **對話歷史管理**：查看、搜尋歷史對話記錄
- **圖像檢視**：從歷史記錄中檢視之前拍攝的圖像

### 技術特色
- **圖像內嵌儲存**：圖像以 base64 格式直接儲存在向量資料庫中
- **語義搜尋**：使用向量相似度搜尋相關對話
- **多模態支援**：支援 LLaVA、gemma3 等多模態模型
- **即時處理**：相機即時預覽，快速圖像分析

## 📋 系統需求

### 硬體需求
- 相機設備（內建或外接）
- 至少 8GB RAM（推薦 16GB）
- 支援 CUDA 的 GPU（可選，用於加速推理）

### 軟體需求
- Python 3.8+
- Docker（用於運行 Milvus）
- Ollama（本地 AI 模型服務）

## 🚀 安裝步驟

### 1. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 2. 安裝並啟動 Ollama

```bash
# 安裝 Ollama（macOS）
curl -fsSL https://ollama.ai/install.sh | sh

# 啟動 Ollama 服務
ollama serve

# 下載支援圖像的模型
ollama pull llava-phi3:latest
# 或
ollama pull llava:latest
```

### 3. 啟動 Milvus 向量資料庫

```bash
# 使用 Docker Compose 啟動 Milvus
docker-compose up -d
```

### 4. 啟動 Attu 管理介面（可選）

```bash
docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest
```

## 📦 專案結構

```
vlm-project/
├── vlm_with_milvus_model_select.py  # 主程式（模型可選版本）
├── requirements.txt                  # Python 依賴
├── docker-compose.yml               # Milvus 容器配置
├── README.md                        # 專案說明文件
└── volumes/                         # Milvus 資料儲存目錄
```

## 🎯 使用方法

### 啟動程式

```bash
python vlm_with_milvus_model_select.py
```

### 基本操作流程

1. **啟動相機**：點擊「開始相機」按鈕
2. **選擇模型**：從下拉選單選擇要使用的 AI 模型
3. **調整參數**：設定 Temperature 等模型參數
4. **輸入問題**：在問題輸入框輸入想要問的問題
5. **拍照分析**：點擊「拍照分析」進行圖像問答
6. **查看結果**：在右側查看 AI 回答

### 進階功能

#### 對話歷史管理
- **搜尋對話**：在搜尋框輸入關鍵字，按 Enter 或點擊「搜尋對話」
- **查看歷史**：點擊「查看歷史」查看最近的對話記錄
- **檢視影像**：點擊「檢視影像」輸入對話編號查看歷史圖像

#### 模型管理
- 程式會自動偵測本機已安裝的 Ollama 模型
- 支援所有支援圖像輸入的模型（如 LLaVA、Phi3 系列）
- 可即時切換不同模型進行推理

## 🔧 配置說明

### 模型配置
- **預設模型**：llava-phi3:latest
- **支援模型**：所有支援圖像輸入的 Ollama 模型
- **模型切換**：可在 GUI 中即時切換

### 資料庫配置
- **向量資料庫**：Milvus
- **集合名稱**：vlm_conversations_with_images
- **向量維度**：384（使用 paraphrase-multilingual-MiniLM-L12-v2）
- **索引類型**：IVF_FLAT with COSINE similarity

### 圖像處理
- **儲存格式**：JPEG（base64 編碼）
- **最大尺寸**：800px（自動縮放）
- **品質控制**：自動調整以符合資料庫限制

## 🐛 常見問題

### Q: 相機無法啟動
A: 檢查相機權限，確保沒有其他程式佔用相機

### Q: Ollama 連接失敗
A: 確認 Ollama 服務正在運行：`ollama serve`

### Q: Milvus 連接失敗
A: 檢查 Docker 是否運行，確認 Milvus 容器狀態

### Q: 模型推理失敗
A: 確認選擇的模型支援圖像輸入，純文字模型（如 gemma3:1b）不支援

### Q: 圖像儲存失敗
A: 檢查 Milvus 連接狀態，確認向量資料庫正常運行

## 📊 效能優化

### 圖像處理優化
- 自動圖像壓縮和品質調整
- 支援多種圖像格式
- 記憶體使用優化

### 推理效能
- 支援 GPU 加速（如果可用）
- 模型快取機制
- 非阻塞式 UI 更新

### 資料庫效能
- 向量索引優化
- 批量操作支援
- 連接池管理

## 🔒 安全性

- 圖像資料本地儲存
- 無需網路連接（除模型下載外）
- 資料庫存取控制
- 隱私保護設計

**享受使用 VLM + Milvus 整合系統！** 🎉 