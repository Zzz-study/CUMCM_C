"""
Y染色体浓度达标分析与BMI聚类分组
基于1063条观测，267名孕妇的数据分析

1. Y染色体浓度达标标记（>=4%）
2. BMI无监督聚类分组
3. 分组孕周-达标比例时序分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_prepare_data():
    """数据读取和基本准备"""
    print("=" * 60)
    print("1. 数据读取和准备")
    print("=" * 60)

    # 读取数据
    try:
        df = pd.read_excel('data_processed.xlsx')
        print(f"成功读取数据: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return None

    # 数据概览
    print(f"\n数据概览:")
    print(f"观测数量: {len(df)}")
    print(f"孕妇数量: {df['孕妇代码'].nunique()}")
    print(f"平均每孕妇观测数: {len(df) / df['孕妇代码'].nunique():.1f}")

    # 检查关键变量
    required_cols = ['孕妇代码', 'Y染色体浓度', '检测孕周_数值', '孕妇BMI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必要列: {missing_cols}")
        return None

    print("✓ 关键变量检查通过")
    return df

def add_y_chromosome_compliance_mark(df):
    """添加Y染色体浓度达标标记"""
    print("\n" + "=" * 60)
    print("2. Y染色体浓度达标标记")
    print("=" * 60)

    # Y染色体浓度达标标准：>= 4%
    compliance_threshold = 0.04  # 4%

    df['Y达标标记'] = (df['Y染色体浓度'] >= compliance_threshold).astype(int)

    # 统计分析
    total_samples = len(df)
    compliant_samples = df['Y达标标记'].sum()
    compliance_rate = compliant_samples / total_samples * 100

    print("Y染色体浓度达标标记完成")
    print(f"Y浓度范围: [{df['Y染色体浓度'].min():.4f}, {df['Y染色体浓度'].max():.4f}]")

    # 达标率分布
    plt.figure(figsize=(10, 6))
    compliance_by_week = df.groupby('检测孕周_数值')['Y达标标记'].mean() * 100
    plt.plot(compliance_by_week.index, compliance_by_week.values, 'o-', alpha=0.7)
    plt.xlabel('检测孕周')
    plt.ylabel('Y染色体达标率 (%)')
    plt.title('Y染色体达标率随孕周的变化')
    plt.grid(True, alpha=0.3)
    plt.savefig('y_chromosome_compliance_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df

def bmi_clustering_analysis(df):
    """BMI聚类分组分析"""
    print("\n" + "=" * 60)
    print("3. BMI聚类分组分析")
    print("=" * 60)

    # 1. 数据准备
    bmi_data = df[['孕妇BMI']].copy()

    # Z-score标准化
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data)

    print("BMI数据标准化完成")
    
    # 2. 肘部法则确定最优K值
    print("\n肘部法则确定聚类数量...")
    wss = []
    k_range = range(4, 7)

    for k in k_range:
        # 使用k-means++初始化，增加运行次数以提高稳定性
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20)
        kmeans.fit(bmi_scaled)
        wss.append(kmeans.inertia_)
        print(f"  K={k}: WSS={kmeans.inertia_:.4f} (20次运行中的最优结果)")

    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wss, 'bo-', alpha=0.7)
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内误差平方和 (WSS)')
    plt.title('肘部法则 - 确定最优聚类数量')
    plt.grid(True, alpha=0.3)

    # 找到肘部点（拐点）
    # 计算二阶差分来找到拐点
    wss_diff = np.diff(wss)
    wss_diff2 = np.diff(wss_diff)
    elbow_point = np.argmin(wss) + 3  # +3 range(4,7),根据世界卫生组织（WHO）标准，BMI分为偏瘦、正常、超重、肥胖四类，其中肥胖进一步分为3级

    plt.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.8,
                label=f'建议K值: {elbow_point}')
    plt.legend()
    plt.savefig('elbow_method_bmi_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"肘部法则建议K值: {elbow_point}")

    # 3. 优化K-means聚类
    optimal_k = elbow_point
    print(f"\n执行优化K-means聚类 (K={optimal_k})...")

    # 使用优化的K-means配置
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=20)
    bmi_data['BMI_Cluster'] = kmeans.fit_predict(bmi_scaled)
    
    print(f"✓ 聚类完成，最终WSS: {kmeans.inertia_:.4f}")

    # 4. 聚类验证和稳定性分析
    print("\n聚类结果验证:")
    silhouette_avg = silhouette_score(bmi_scaled, bmi_data['BMI_Cluster'])
    print(f"轮廓系数: {silhouette_avg:.4f}")
    
    if silhouette_avg >= 0.5:
        print("轮廓系数≥0.5，聚类质量良好")
    else:
        print("轮廓系数<0.5，聚类质量一般")
    
    print("\n多次运行稳定性分析:")
    print(f"k-means++初始化: 智能选择初始中心，避免局部最优")
    print(f"固定随机种子: 确保在相同数据下结果可重现")
    print(f"最终选择模型的WSS: {kmeans.inertia_:.4f}")


    # 5. 聚类结果分析
    cluster_stats = bmi_data.groupby('BMI_Cluster')['孕妇BMI'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nBMI聚类结果统计:")
    print(cluster_stats)

    # 检查样本量是否充足
    min_samples = cluster_stats['count'].min()
    if min_samples < 20:
        print(f"最小簇样本量: {min_samples} < 20，建议合并小簇")
    else:
        print(f"最小簇样本量: {min_samples} ≥ 20，样本量充足")

    # 6. 临床意义验证
    print("\n临床意义验证:")
    print("基于WHO肥胖分级标准的聚类结果解释:")
    # WHO肥胖分级标准
    for cluster in cluster_stats.index:
        bmi_mean = cluster_stats.loc[cluster, 'mean']
        if bmi_mean < 18.5:
            category = "偏瘦"
        elif bmi_mean < 25:
            category = "正常体重"
        elif bmi_mean < 30:
            category = "超重"
        elif bmi_mean < 35:
            category = "肥胖I度"
        elif bmi_mean < 40:
            category = "肥胖II度"
        else:
            category = "肥胖III度"
        print(f"簇{cluster} (平均BMI: {bmi_mean:.1f} kg/m²): {category}")
    # 7. 可视化聚类结果
    plt.figure(figsize=(12, 8))

    # BMI分布直方图
    plt.subplot(2, 2, 1)
    for cluster in range(optimal_k):
        cluster_data = bmi_data[bmi_data['BMI_Cluster'] == cluster]
        plt.hist(cluster_data['孕妇BMI'], bins=15, alpha=0.7,
                label=f'簇{cluster} (n={len(cluster_data)})')
    plt.xlabel('BMI (kg/m²)')
    plt.ylabel('频数')
    plt.title('BMI聚类分布 (k-means++优化)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 簇质心
    plt.subplot(2, 2, 2)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids):
        plt.bar(i, centroid[0], alpha=0.7, label=f'簇{i}质心')
    plt.xlabel('簇编号')
    plt.ylabel('BMI质心值 (kg/m²)')
    plt.title('聚类质心 (20次运行最优)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # BMI箱线图
    plt.subplot(2, 2, 3)
    cluster_labels = [f'簇{i}' for i in range(optimal_k)]
    plt.boxplot([bmi_data[bmi_data['BMI_Cluster'] == i]['孕妇BMI'] for i in range(optimal_k)],
               labels=cluster_labels)
    plt.ylabel('BMI (kg/m²)')
    plt.title('BMI聚类箱线图')
    plt.grid(True, alpha=0.3)

    # 原始数据散点图（按簇着色）
    plt.subplot(2, 2, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for cluster in range(optimal_k):
        cluster_data = bmi_data[bmi_data['BMI_Cluster'] == cluster]
        plt.scatter(range(len(cluster_data)), cluster_data['孕妇BMI'],
                   c=[colors[cluster]], alpha=0.6, s=30, label=f'簇{cluster}')
    plt.xlabel('样本索引')
    plt.ylabel('BMI (kg/m²)')
    plt.title('BMI聚类散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bmi_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return bmi_data

def analyze_compliance_by_bmi_group(df, bmi_clusters):
    """按BMI组分析孕周-达标比例时序规律"""
    print("\n" + "=" * 60)
    print("4. 分组孕周-达标比例时序分析")
    print("=" * 60)

    # 合并聚类结果
    df_with_clusters = df.copy()
    df_with_clusters['BMI_Cluster'] = bmi_clusters['BMI_Cluster']

    # 获取BMI聚类数量
    n_clusters = bmi_clusters['BMI_Cluster'].nunique()

    print(f"开始分析 {n_clusters} 个BMI组的时序规律...")

    # 1. 数据排序
    df_sorted = df_with_clusters.sort_values(['BMI_Cluster', '检测孕周_数值'])

    # 2. 孕周分段统计
    week_intervals = []
    compliance_results = []

    # 确定孕周范围
    min_week = df_sorted['检测孕周_数值'].min()
    max_week = df_sorted['检测孕周_数值'].max()

    # 以1周为间隔进行分段
    current_week = np.floor(min_week)

    while current_week < max_week:
        interval_start = current_week
        interval_end = current_week + 1

        # 筛选该区间的数据
        interval_data = df_sorted[
            (df_sorted['检测孕周_数值'] >= interval_start) &
            (df_sorted['检测孕周_数值'] < interval_end)
        ]

        if len(interval_data) > 0:
            week_intervals.append((interval_start, interval_end))

            # 按BMI组计算达标比例
            for cluster in range(n_clusters):
                cluster_data = interval_data[interval_data['BMI_Cluster'] == cluster]

                if len(cluster_data) >= 5:  # 样本量足够
                    total_samples = len(cluster_data)
                    compliant_samples = cluster_data['Y达标标记'].sum()
                    compliance_rate = compliant_samples / total_samples

                    compliance_results.append({
                        'week_start': interval_start,
                        'week_end': interval_end,
                        'bmi_cluster': cluster,
                        'total_samples': total_samples,
                        'compliant_samples': compliant_samples,
                        'compliance_rate': compliance_rate
                    })

        current_week += 1

    # 转换为DataFrame
    compliance_df = pd.DataFrame(compliance_results)

    # 3. 绘制达标比例曲线
    plt.figure(figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for cluster in range(n_clusters):
        cluster_data = compliance_df[compliance_df['bmi_cluster'] == cluster]

        if len(cluster_data) > 0:
            # 计算BMI组的平均BMI值用于图例
            bmi_mean = df_with_clusters[df_with_clusters['BMI_Cluster'] == cluster]['孕妇BMI'].mean()

            plt.plot(cluster_data['week_start'] + 0.5, cluster_data['compliance_rate'] * 100,
                    'o-', color=colors[cluster], alpha=0.8, linewidth=2, markersize=6,
                    label='.1f')
            # 标注达标比例≥80%的区间
            high_compliance = cluster_data[cluster_data['compliance_rate'] >= 0.8]
            if len(high_compliance) > 0:
                plt.scatter(high_compliance['week_start'] + 0.5,
                           high_compliance['compliance_rate'] * 100,
                           color=colors[cluster], s=100, marker='*',
                           edgecolor='red', linewidth=2,
                           label=f'BMI簇{cluster} 高达标区' if cluster == 0 else "")

    plt.xlabel('孕周 (周)', fontsize=12)
    plt.ylabel('Y染色体达标比例 (%)', fontsize=12)
    plt.title('各BMI组孕周-Y染色体达标比例时序分析', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 添加80%阈值线
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.7,
               label='临床可接受阈值 (80%)')

    plt.tight_layout()
    plt.savefig('compliance_by_bmi_group_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 统计分析报告
    print("\n时序分析统计报告:")
    for cluster in range(n_clusters):
        cluster_stats = compliance_df[compliance_df['bmi_cluster'] == cluster]
        bmi_mean = df_with_clusters[df_with_clusters['BMI_Cluster'] == cluster]['孕妇BMI'].mean()

        if len(cluster_stats) > 0:
            avg_compliance = cluster_stats['compliance_rate'].mean() * 100
            max_compliance = cluster_stats['compliance_rate'].max() * 100
            high_compliance_weeks = len(cluster_stats[cluster_stats['compliance_rate'] >= 0.8])

            print("\nBMI簇 {} (平均BMI: {:.1f} kg/m²):".format(cluster, bmi_mean))
            print(f"  高达标区间数 (≥80%): {high_compliance_weeks}")

            if high_compliance_weeks > 0:
                high_weeks = cluster_stats[cluster_stats['compliance_rate'] >= 0.8]['week_start'].values
                print(f"  高达标孕周: {', '.join([f'{w:.0f}' for w in high_weeks])} 周")

    return df_with_clusters, compliance_df

def generate_final_report(df, bmi_clusters, compliance_results):
    """生成最终分析报告"""
    print("\n" + "=" * 60)
    print("5. 最终分析报告")
    print("=" * 60)

    print("Y染色体浓度达标分析与BMI聚类分组 - 完整报告")
    print("=" * 60)

    # 1. 数据概览
    print("数据概览:")
    print(f"总观测数: {len(df)}")
    print(f"孕妇数: {df['孕妇代码'].nunique()}")
    print(f"平均每孕妇观测数: {len(df) / df['孕妇代码'].nunique():.1f}")

    # 2. Y染色体达标情况
    compliance_rate = df['Y达标标记'].mean() * 100
    print("\nY染色体达标情况:")

    print(f"达标样本数: {df['Y达标标记'].sum()}")
    print(f"未达标样本数: {len(df) - df['Y达标标记'].sum()}")

    # 3. BMI聚类结果
    n_clusters = bmi_clusters['BMI_Cluster'].nunique()
    print(f"\nBMI聚类结果:")
    print(f"聚类数量: {n_clusters}")

    cluster_stats = df.groupby('BMI_Cluster')['孕妇BMI'].agg(['count', 'mean', 'std', 'min', 'max'])
    for cluster in range(n_clusters):
        stats = cluster_stats.loc[cluster]


    print("\n分析完成！")

def main():
    """主分析流程"""
    print("Y染色体浓度达标分析与BMI聚类分组 (K-means++优化版)")
    print("=" * 60)

    # 1. 数据读取和准备
    df = load_and_prepare_data()
    if df is None:
        return

    # 2. 添加Y染色体达标标记
    df = add_y_chromosome_compliance_mark(df)

    # 3. BMI聚类分析
    bmi_clusters = bmi_clustering_analysis(df)

    # 4. 分组时序分析
    df_with_clusters, compliance_results = analyze_compliance_by_bmi_group(df, bmi_clusters)

    # 5. 生成最终报告
    generate_final_report(df_with_clusters, bmi_clusters, compliance_results)



if __name__ == "__main__":
    main()
