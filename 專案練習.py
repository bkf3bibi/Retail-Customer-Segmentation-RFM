import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. 環境設定與檔案讀取 ---
file_path = r"C:\Users\user\Desktop\wendy\python\project\retail_cleans.csv"
output_html = r"C:\Users\user\Desktop\wendy\python\project\customer_3d_report.html"

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

try:
    # 讀取數據
    df = pd.read_csv(file_path, parse_dates=['InvoiceDate'], low_memory=False)
    print("✅ 數據讀取成功！")

    # --- 2. 計算 RFM 指標 ---
    id_col = 'Customer ID'
    invoice_col = 'Invoice'
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby(id_col).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0] 

    # --- 3. 機器學習分群 (K-Means) ---
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    print("✅ 機器學習分群完成！")

    # --- 4. 視覺化：優化後的蛇形圖 (解決遮擋問題) ---
    summary_for_legend = rfm.groupby('Cluster').mean().round(1)
    
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=['最近消費(R)', '消費頻率(F)', '消費金額(M)'])
    rfm_scaled_df['分群'] = rfm['Cluster']
    
    # 建立清楚的圖例標籤
    cluster_labels = {}
    for i in range(4):
        cluster_labels[i] = (f"分群 {i}: 均消 ${summary_for_legend.loc[i, 'Monetary']:.0f} | "
                             f"{summary_for_legend.loc[i, 'Frequency']:.1f}次 | "
                             f"{summary_for_legend.loc[i, 'Recency']:.0f}天")
    
    rfm_scaled_df['分群'] = rfm_scaled_df['分群'].map(cluster_labels)

    rfm_melted = pd.melt(rfm_scaled_df.reset_index(), id_vars=[id_col, '分群'], 
                         value_vars=['最近消費(R)', '消費頻率(F)', '消費金額(M)'], 
                         var_name='指標', value_name='標準化數值')

    # 1. 增加畫布寬度以容納右側圖例
    plt.figure(figsize=(15, 8)) 
    
    # 繪製線條
    ax = sns.lineplot(x='指標', y='標準化數值', hue='分群', data=rfm_melted, palette='bright', marker='o', linewidth=3)
    
    # 加入平均參考線
    plt.axhline(0, color='red', linestyle='--', alpha=0.3) 
    plt.text(2.1, 0.05, '全體平均線', color='red', fontsize=10, fontweight='bold')

    plt.title('客戶分群特徵分析 (相對表現與真實數值對照)', fontsize=18, pad=25)
    plt.ylabel('相對表現強度 (0=平均)', fontsize=12)
    plt.xlabel('RFM 衡量指標', fontsize=12)
    
    # 2. 將圖例完全移出繪圖區（bbox_to_anchor 是關鍵）
    plt.legend(title='分群特徵描述 (真實平均值)', 
               bbox_to_anchor=(1.02, 1), 
               loc='upper left', 
               borderaxespad=0.,
               fontsize=10)
    
    # 3. 調整底部的解釋文字，避免與座標軸重疊
    plt.figtext(0.4, 0.02, "💡 如何解讀：數值越高代表表現越強；最近消費(R)數值越低代表越活躍。", 
                ha="center", fontsize=11, color='darkblue', fontweight='bold')

    # 4. 關鍵佈局自動調整，rect 參數預留底部 5% 的空間給解釋文字
    plt.tight_layout(rect=[0, 0.05, 0.95, 1]) 
    plt.show()

    # --- 5. 視覺化：互動式 3D 空間圖並存成 HTML ---
    plot_df = rfm.reset_index()
    plot_df['Cluster'] = plot_df['Cluster'].astype(str)

    fig = px.scatter_3d(
        plot_df, x='Recency', y='Frequency', z='Monetary',
        color='Cluster', hover_name=id_col,
        title='互動式客戶分群 3D 空間圖',
        labels={'Recency': '最近消費(天)', 'Frequency': '消費頻率(次)', 'Monetary': '消費金額(元)'}
    )
    
    fig.write_html(output_html)
    print(f"✨ 互動式網頁已產出：{output_html}")

    # --- 6. 輸出報告摘要 ---
    summary = rfm.groupby('Cluster').mean().round(1)
    summary.columns = ['平均最近消費(天)', '平均次數(次)', '平均金額(元)']
    print("\n" + "="*40)
    print("📊 各族群行為特徵中文摘要")
    print(summary)
    print("="*40)

except Exception as e:
    print(f"❌ 發生錯誤：{e}")
