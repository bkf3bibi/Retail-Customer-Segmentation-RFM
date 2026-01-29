
# 零售業客戶分群分析：基於 RFM 模型與 K-Means 機器學習

## 📌 專案簡介

本專案利用零售交易數據，透過 **RFM 模型** (Recency, Frequency, Monetary) 對客戶進行特徵提取，並應用 **K-Means 聚類演算法** 實施自動化客戶分群。目標是協助企業識別高價值客戶、識別流失風險，並制定差異化的精準行銷策略。

---

## 🛠️ 技術堆疊

* **程式語言**: Python 3.11
* **數據處理**: Pandas, NumPy
* **機器學習**: Scikit-learn (K-Means Clustering)
* **數據視覺化**:
* **Seaborn / Matplotlib**: 用於繪製靜態蛇形圖 (Snake Plot) 觀察特徵。
* **Plotly**: 用於建立 3D 互動式空間分佈圖，支援懸停檢查。



---

## 📊 實作流程

1. **數據清洗 (Data Cleaning)**: 處理缺失值、格式化日期並計算每筆交易的總金額。
2. **特徵工程 (RFM Extraction)**:
* **Recency (最近消費)**: 計算客戶最後一次購買距今的天數。
* **Frequency (消費頻率)**: 客戶不重複的訂單總數。
* **Monetary (消費金額)**: 客戶消費的總金額。


3. **數據預處理**:
* **Log 轉換**: 修正零售數據常見的偏態分佈（長尾效應）。
* **標準化 (Standardization)**: 使用 `StandardScaler` 將 R, F, M 縮放至同一尺度，確保模型公平性。


4. **模型訓練**: 應用 **K-Means 演算法** 將客戶自動區分為 4 個族群。

---

## 📈 視覺化成果與分析

### 1. 客戶分群特徵圖 (Snake Plot)

透過蛇形圖，我們可以直觀觀察各分群在 R、F、M 三個維度的強弱表現：

* **核心 VIP (Champions)**: R 極低（近期購買）、F 與 M 極高。
* **潛力主力**: 各項指標穩定高於平均。
* **流失風險**: R 很高（久未消費），需進行喚醒。

### 2. 互動式 3D 空間分佈圖

利用 3D 散佈圖觀察機器學習如何切分數據空間，支援滑鼠旋轉與游標懸停顯示客戶 ID。

---

## 💡 商業行動建議 (Business Insights)

| 客戶分群 | 特徵描述 | 建議行動 |
| --- | --- | --- |
| **黃金 VIP** | 消費力最強、回購率最高 | 提供尊榮會員禮、新品優先體驗，維持黏著度。 |
| **忠誠鐵粉** | 消費穩定，具備成長潛力 | 實施滿額贈或升等優惠，引導提升消費單價。 |
| **流失警示** | 曾有高消費但近期久未露面 | 發送「想念禮券」或進行流失原因問卷調查。 |
| **新客/一般客** | 消費次數少且金額低 | 推薦熱銷入門品，透過小額折扣誘發第二次購買。 |

---

## 🚀 如何執行

1. 複製儲存庫：
```bash
git clone https://github.com/你的用戶名/你的專案名.git

```


2. 安裝必要套件：
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly




```


3. 執行主程式：
```bash
python 專案練習.py
<img width="1176" height="720" alt="image" src="https://github.com/user-attachments/assets/fe395832-f500-441d-9d9b-44cf4aca14ed" />
<img width="1820" height="855" alt="螢幕擷取畫面 2026-01-29 153429" src="https://github.com/user-attachments/assets/7fbfa736-61e3-4227-bb33-4a6447ae1fda" />
https://bkf3bibi.github.io/Retail-Customer-Segmentation-RFM/
