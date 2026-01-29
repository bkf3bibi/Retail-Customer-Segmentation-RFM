import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. 檔案讀取與環境設定 ---
file_path = r"C:\Users\user\Desktop\黃沛瑜\python\專案相關\retail_cleans.csv"

# 設定 Matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

try:
    # 讀取數據
    df = pd.read_csv(file_path, parse_dates=['InvoiceDate'], low_memory=False)
    print("✅ 數據讀取成功！")

    # 欄位名稱校正
    id_col = 'Customer ID'
    invoice_col = 'Invoice'

    # --- 2. 計算 RFM 指標 ---
    # 設定基準日
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    # 聚合數據
    rfm = df.groupby(id_col).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        invoice_col: 'nunique',
        'TotalPrice': 'sum'
    })

    # 重新命名欄位
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # 排除異常值並進行數據轉換 (Log 轉換處理偏態)
    rfm = rfm[rfm['Monetary'] > 0]
    rfm_log = np.log1p(rfm)

    # --- 3. 機器學習分群 (K-Means) ---
    # 數據標準化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 執行分群
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    print("\n--- RFM 分群計算完成 ---")
    print(rfm.head())

    # --- 4. 視覺化 A：2D 中文蛇形圖 (Snake Plot) ---
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=['最近消費(R)', '消費頻率(F)', '消費金額(M)'])
    rfm_scaled_df['分群'] = rfm['Cluster']
    rfm_melted = pd.melt(rfm_scaled_df.reset_index(), id_vars=[id_col, '分群'], 
                         value_vars=['最近消費(R)', '消費頻率(F)', '消費金額(M)'], 
                         var_name='指標', value_name='標準化數值')

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='指標', y='標準化數值', hue='分群', data=rfm_melted, palette='bright', marker='o')
    plt.title('客戶分群特徵圖 (RFM Snake Plot)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # --- 5. 視覺化 B：Plotly 互動式 3D 空間圖 ---
    # 準備繪圖 DataFrame
    plot_df = rfm.reset_index()
    plot_df['Cluster'] = plot_df['Cluster'].astype(str)

    fig = px.scatter_3d(
        plot_df, 
        x='Recency', 
        y='Frequency', 
        z='Monetary',
        color='Cluster',
        hover_name=id_col,
        hover_data={
            'Recency': ':,.0f', 
            'Frequency': ':,.0f', 
            'Monetary': ':,.1f',
            'Cluster': False
        },
        title='互動式客戶分群 3D 空間圖 (滑鼠旋轉/縮放/懸停檢查)',
        labels={
            'Recency': '最近消費(天)', 
            'Frequency': '消費頻率(次)', 
            'Monetary': '消費金額(元)'
        },
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))
    fig.show()

    # (選配) 儲存互動圖表為 HTML 網頁
    # fig.write_html("customer_segmentation_3d.html")

    # --- 6. 輸出中文摘要報告 ---
    summary = rfm.groupby('Cluster').mean().round(1)
    summary.columns = ['平均最近消費(天)', '平均次數(次)', '平均金額(元)']
    print("\n" + "="*40)
    print("📊 各族群行為特徵中文摘要")
    print("="*40)
    print(summary)

except FileNotFoundError:
    print(f"❌ 找不到檔案，請確認路徑：{file_path}")
except KeyError as e:
    print(f"❌ 欄位名稱錯誤：找不到 {e}")
except Exception as e:
    print(f"❌ 發生錯誤：{e}")