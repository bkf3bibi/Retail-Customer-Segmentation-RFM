import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. 環境設定與檔案路徑 ---
base_path = r"C:\Users\user\Desktop\wendy\python\project"
file_path = f"{base_path}\\retail_cleans.csv"
output_html = f"{base_path}\\customer_3d_report.html"
output_png = f"{base_path}\\rfm_snake_plot.png"
output_pbi = f"{base_path}\\rfm_for_pbi.csv"

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

    # --- 4. 視覺化：蛇形圖並存成 PNG (用於 GitHub README) ---
    summary_for_legend = rfm.groupby('Cluster').mean().round(1)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=['最近消費(R)', '消費頻率(F)', '消費金額(M)'])
    rfm_scaled_df['Cluster'] = rfm['Cluster']
    
    cluster_labels = {}
    for i in range(4):
        cluster_labels[i] = (f"分群 {i}: 均消 ${summary_for_legend.loc[i, 'Monetary']:.0f} | "
                             f"{summary_for_legend.loc[i, 'Frequency']:.1f}次 | "
                             f"{summary_for_legend.loc[i, 'Recency']:.0f}天")
    
    rfm_scaled_df['分群標籤'] = rfm_scaled_df['Cluster'].map(cluster_labels)
    rfm_melted = pd.melt(rfm_scaled_df.reset_index(), id_vars=[id_col, '分群標籤'], 
                         value_vars=['最近消費(R)', '消費頻率(F)', '消費金額(M)'], 
                         var_name='指標', value_name='標準化數值')

    plt.figure(figsize=(15, 8)) 
    sns.lineplot(x='指標', y='標準化數值', hue='分群標籤', data=rfm_melted, palette='bright', marker='o', linewidth=3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.3) 
    plt.title('客戶分群特徵分析 (Snake Plot)', fontsize=18, pad=25)
    plt.legend(title='各群真實數據平均值', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.figtext(0.4, 0.02, "💡 數值越高代表表現越強；最近消費(R)越低代表越活躍。", 
                ha="center", fontsize=11, color='darkblue', fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 0.95, 1]) 
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📸 蛇形圖圖片已更新：{output_png}")
    plt.close() # 關閉視窗，讓自動化流程更順暢

    # --- 5. 視覺化：3D 空間圖並存成 HTML (用於 GitHub Pages) ---
    plot_df = rfm.reset_index()
    plot_df['Cluster_Str'] = plot_df['Cluster'].astype(str)
    fig = px.scatter_3d(
        plot_df, x='Recency', y='Frequency', z='Monetary',
        color='Cluster_Str', hover_name=id_col,
        title='互動式客戶分群 3D 空間圖',
        labels={'Recency': '最近消費(天)', 'Frequency': '消費頻率(次)', 'Monetary': '消費金額(元)'}
    )
    fig.write_html(output_html)
    print(f"✨ 互動式網頁已更新：{output_html}")

    # --- 6. 輸出 Power BI 專用資料 ---
    pbi_df = rfm.reset_index()
    cluster_names = {3: "核心 VIP", 2: "重點發展", 0: "潛力新客", 1: "預警流失"}
    pbi_df['群組名稱'] = pbi_df['Cluster'].map(cluster_names)
    pbi_df.to_csv(output_pbi, index=False, encoding='utf-8-sig')
    print(f"📊 Power BI 資料已更新：{output_pbi}")

    print("\n🚀 所有分析檔案已全數產出至專案資料夾！")

except Exception as e:
    print(f"❌ 發生錯誤：{e}")
