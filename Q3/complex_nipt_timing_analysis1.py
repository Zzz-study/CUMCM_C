import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats
import warnings
from lifelines import CoxPHFitter
import prince

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)


class NIPTTimingAnalyzer:
    """NIPT最佳时点分析器（多因素聚类分析）"""

    def __init__(self):
        self.df = None
        self.cluster_results = {}
        self.timing_results = {}
        self.factor_importance = {}
        self.pca_model = None
        self.pca_components = None

    def load_and_preprocess_data(self, file_path='data_processed.xlsx'):
        """加载数据并进行多因素预处理"""
        try:
            self.df = pd.read_excel(file_path)
            print(f"数据加载成功: {self.df.shape[0]} 行")

            # 统一列名
            self.df = self.df.rename(columns={
                '孕妇代码': 'PatientID',
                '检测孕周_数值': 'GestationalWeek',
                '孕妇BMI': 'BMI',
                'Y染色体浓度': 'Y_Fraction',
                '年龄': 'Age',
                '身高': 'Height',
                '体重': 'Weight'
            })

            # 数据清洗
            self.df['Y_Fraction'] = pd.to_numeric(self.df['Y_Fraction'], errors='coerce')
            if self.df['Y_Fraction'].dropna().gt(1).mean() > 0.2:
                self.df['Y_Fraction'] = self.df['Y_Fraction'] / 100.0

            self.df['BMI'] = pd.to_numeric(self.df['BMI'], errors='coerce')
            self.df['GestationalWeek'] = pd.to_numeric(self.df['GestationalWeek'], errors='coerce')
            self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
            self.df['Height'] = pd.to_numeric(self.df['Height'], errors='coerce')
            self.df['Weight'] = pd.to_numeric(self.df['Weight'], errors='coerce')

            # 计算达标时间（首次Y_Fraction≥0.04的孕周）
            self.df['Y_Compliant'] = (self.df['Y_Fraction'] >= 0.04).astype(int)

            # 计算首次达标时间
            compliant_df = self.df[self.df['Y_Compliant'] == 1]
            first_compliant = compliant_df.groupby('PatientID')['GestationalWeek'].min().reset_index()
            first_compliant.columns = ['PatientID', 'FirstCompliantWeek']

            # 合并回主数据集
            self.df = pd.merge(self.df, first_compliant, on='PatientID', how='left')

            # 创建生存分析所需的事件和时间列
            self.df['Event'] = ~pd.isna(self.df['FirstCompliantWeek']).astype(int)
            self.df['Time'] = self.df.apply(
                lambda x: x['FirstCompliantWeek'] if pd.notna(x['FirstCompliantWeek']) else x['GestationalWeek'],
                axis=1
            )

            # 移除缺失值
            self.df = self.df.dropna(subset=['BMI', 'Y_Fraction', 'GestationalWeek', 'Age', 'Height', 'Weight'])

            print(f"总体数据概览：")
            print(f"  观测数量: {len(self.df)}")
            print(f"  孕妇数量: {self.df['PatientID'].nunique()}")
            print(f"  总体Y达标率: {self.df['Y_Compliant'].mean() * 100:.1f}%")

            return True

        except Exception as e:
            print(f"数据处理失败: {e}")
            return False

   def perform_pca_clustering(self, n_clusters=5, n_components=2):
        """执行PCA降维和聚类分析"""
        print(f"\n执行PCA降维和聚类分析...")

        # 选择用于PCA的特征
        pca_features = ['Age', 'Height', 'Weight']
        pca_data = self.df[pca_features].copy()

        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)

        # 执行PCA降维
        self.pca_model = PCA(n_components=n_components, random_state=42)
        pca_result = self.pca_model.fit_transform(scaled_data)

        # 保存PCA结果
        for i in range(n_components):
            self.df[f'PC{i + 1}'] = pca_result[:, i]

        # 保存PCA解释方差
        self.pca_components = {
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'total_variance_explained': sum(self.pca_model.explained_variance_ratio_)
        }

        print(
            f"PCA降维完成，前{n_components}个主成分解释方差: {self.pca_components['total_variance_explained'] * 100:.2f}%")
        for i, ratio in enumerate(self.pca_components['explained_variance_ratio']):
            print(f"  主成分{i + 1}解释方差: {ratio * 100:.2f}%")

        # 使用高斯混合模型进行聚类
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        self.df['Cluster'] = gmm.fit_predict(pca_result)

        # 分析各组特征
        self._analyze_pca_groups(pca_features)

        return True

    def _analyze_pca_groups(self, features):
        """分析PCA聚类结果"""
        print(f"\nPCA聚类结果:")
        print("-" * 100)

        for cluster in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            if len(cluster_data) == 0:
                continue

            # 计算各特征的统计量
            cluster_stats = {}
            for feature in features + ['BMI']:
                values = cluster_data[feature]
                cluster_stats[feature] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max()
                }

            # WHO BMI分类
            mean_bmi = cluster_stats['BMI']['mean']
            if mean_bmi < 18.5:
                who_category = "偏瘦"
            elif mean_bmi < 25:
                who_category = "正常体重"
            elif mean_bmi < 30:
                who_category = "超重"
            elif mean_bmi < 35:
                who_category = "肥胖I度"
            elif mean_bmi < 40:
                who_category = "肥胖II度"
            else:
                who_category = "肥胖III度"

            # 年龄分类
            mean_age = cluster_stats['Age']['mean']
            if mean_age < 25:
                age_category = "年轻"
            elif mean_age < 35:
                age_category = "适龄"
            else:
                age_category = "高龄"

            self.cluster_results[cluster] = {
                'sample_size': len(cluster_data),
                'patient_count': cluster_data['PatientID'].nunique(),
                'who_category': who_category,
                'age_category': age_category,
                'y_compliance_rate': cluster_data['Y_Compliant'].mean(),
                'stats': cluster_stats
            }

            print(f"聚类{cluster} ({who_category}, {age_category}): ")
            for feature in features + ['BMI']:
                stats = cluster_stats[feature]
                print(f"  {feature}: {stats['mean']:.1f}±{stats['std']:.1f} "
                      f"({stats['min']:.1f}-{stats['max']:.1f})")
            print(f"  样本量: {len(cluster_data)}, Y达标率: {cluster_data['Y_Compliant'].mean() * 100:.1f}%")
            print()

    def survival_analysis(self):
        """执行生存分析（Cox比例风险模型）"""
        print(f"\n执行生存分析...")

        # 准备生存分析数据
        survival_df = self.df.groupby('PatientID').first().reset_index()
        survival_df = survival_df[['PatientID', 'Cluster', 'BMI', 'Age', 'Height', 'Weight', 'Time', 'Event']]

        # 拟合Cox模型
        cox_df = survival_df[['Time', 'Event', 'BMI', 'Age', 'Height', 'Weight']].copy()

        # 标准化连续变量
        for col in ['BMI', 'Age', 'Height', 'Weight']:
            cox_df[col] = (cox_df[col] - cox_df[col].mean()) / cox_df[col].std()

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='Time', event_col='Event')

        # 提取因素重要性
        self.factor_importance = cph.summary.to_dict()['exp(coef)']

        print(f"Cox比例风险模型结果:")
        print(cph.summary)

        return cph

    def risk_function(self, t):
        """
        孕周风险函数 R(t)
        R(t)=0 (t≤12), R(t)=(t−12)/15 (12<t≤27), R(t)=1 (t>27)
        """
        if t <= 12:
            return 0.0
        elif t <= 27:
            return (t - 12) / 15.0
        else:
            return 1.0

    def wilson_lower_bound(self, successes, total, confidence=0.95):
        """Wilson置信区间下界"""
        if total == 0:
            return 0.0
        z = stats.norm.ppf((1 + confidence) / 2)
        p = successes / total
        denominator = 1 + z ** 2 / total
        centre = (p + z ** 2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denominator
        return max(0, centre - margin)

    def calculate_optimal_timing(self, min_patients=5):
        """优化的最佳NIPT检测时点计算算法"""
        print(f"\n{'=' * 60}")
        print("优化最佳NIPT检测时点计算（多维度风险最小化）")
        print(f"{'=' * 60}")

        for cluster in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[
                (self.df['Cluster'] == cluster) &
                (self.df['GestationalWeek'].between(10, 30))
                ].copy()

            if len(cluster_data) < 20:
                print(f"\n聚类{cluster}: 样本量不足 (n={len(cluster_data)})")
                continue

            # 按孕妇-孕周聚合，计算更详细的统计量
            patient_week_stats = (cluster_data.groupby(['PatientID', 'GestationalWeek'])
                                  .agg({
                'Y_Fraction': ['mean', 'std', 'count'],
                'Y_Compliant': 'mean'
            }).reset_index())

            patient_week_stats.columns = ['PatientID', 'GestationalWeek', 'Y_Fraction_mean',
                                          'Y_Fraction_std', 'Y_Fraction_count', 'Y_Compliant']

            # 按孕周统计，增加稳定性指标
            week_stats = (patient_week_stats.groupby('GestationalWeek')
                          .agg({
                'Y_Compliant': ['count', 'sum', 'mean', 'std'],
                'PatientID': 'nunique',
                'Y_Fraction_mean': ['mean', 'std'],
                'Y_Fraction_std': 'mean'
            }).reset_index())

            week_stats.columns = ['week', 'total_obs', 'compliant_obs', 'compliance_rate',
                                  'compliance_std', 'unique_patients', 'avg_y_fraction',
                                  'y_fraction_variability', 'avg_measurement_error']

            week_stats = week_stats[week_stats['unique_patients'] >= min_patients]

            if len(week_stats) == 0:
                print(f"\n聚类{cluster}: 无足够样本的孕周")
                continue

            # 计算优化的评估指标
            # 1. Wilson置信区间下界（统计可靠性）
            week_stats['wilson_lb'] = [
                self.wilson_lower_bound(int(k), int(n))
                for k, n in zip(week_stats['compliant_obs'], week_stats['total_obs'])
            ]

            # 2. 基础风险因子
            week_stats['risk_factor'] = week_stats['week'].apply(self.risk_function)

            # 3. 样本量权重（对数变换，避免过度偏向大样本）
            max_patients = week_stats['unique_patients'].max()
            week_stats['sample_weight'] = np.log(week_stats['unique_patients']) / np.log(max_patients)

            # 4. 稳定性评分（基于测量变异性）
            week_stats['stability_score'] = 1 / (1 + week_stats['y_fraction_variability'].fillna(0.1))

            # 5. 一致性评分（基于达标率的一致性）
            week_stats['consistency_score'] = 1 / (1 + week_stats['compliance_std'].fillna(0.1))

            # 6. 综合风险最小化评分（多维度加权）
            w_compliance = 0.35  # 达标率权重
            w_risk = 0.25  # 风险权重
            w_reliability = 0.20  # 统计可靠性权重
            w_stability = 0.10  # 稳定性权重
            w_consistency = 0.10  # 一致性权重

            # 归一化各指标到[0,1]区间
            week_stats['compliance_norm'] = week_stats['compliance_rate']
            week_stats['risk_norm'] = 1 - week_stats['risk_factor']
            week_stats['reliability_norm'] = week_stats['wilson_lb']
            week_stats['stability_norm'] = (week_stats['stability_score'] - week_stats['stability_score'].min()) / \
                                           (week_stats['stability_score'].max() - week_stats[
                                               'stability_score'].min() + 1e-8)
            week_stats['consistency_norm'] = (week_stats['consistency_score'] - week_stats['consistency_score'].min()) / \
                                             (week_stats['consistency_score'].max() - week_stats[
                                                 'consistency_score'].min() + 1e-8)

            # 计算综合评分
            week_stats['comprehensive_score'] = (
                    w_compliance * week_stats['compliance_norm'] +
                    w_risk * week_stats['risk_norm'] +
                    w_reliability * week_stats['reliability_norm'] +
                    w_stability * week_stats['stability_norm'] +
                    w_consistency * week_stats['consistency_norm']
            )

            # 多策略寻找最佳时点
            strategies = []

            # 策略1：高质量优先（达标率≥80% 且 Wilson下界≥70%）
            high_quality = week_stats[
                (week_stats['compliance_rate'] >= 0.8) &
                (week_stats['wilson_lb'] >= 0.7)
                ]
            if len(high_quality) > 0:
                best_hq = high_quality.loc[high_quality['comprehensive_score'].idxmax()]
                strategies.append(('高质量优先', best_hq, '高置信度'))

            # 策略2：平衡优化（综合评分最高）
            best_balanced = week_stats.loc[week_stats['comprehensive_score'].idxmax()]
            strategies.append(('平衡优化', best_balanced, '中等置信度'))

            # 策略3：低风险优先（风险因子<0.5 中选最佳）
            low_risk = week_stats[week_stats['risk_factor'] < 0.5]
            if len(low_risk) > 0:
                best_lr = low_risk.loc[low_risk['comprehensive_score'].idxmax()]
                strategies.append(('低风险优先', best_lr, '中等置信度'))

            # 选择最优策略（优先高质量，其次平衡优化）
            if len(strategies) > 0 and strategies[0][0] == '高质量优先':
                selected_strategy, best_row, confidence_level = strategies[0]
            else:
                selected_strategy, best_row, confidence_level = strategies[0] if strategies else ('默认', best_balanced,
                                                                                                  '低置信度')

            # 保存详细结果
            self.timing_results[cluster] = {
                'best_week': int(best_row['week']),
                'compliance_rate': best_row['compliance_rate'],
                'wilson_lb': best_row['wilson_lb'],
                'risk_factor': best_row['risk_factor'],
                'comprehensive_score': best_row['comprehensive_score'],
                'stability_score': best_row['stability_score'],
                'consistency_score': best_row['consistency_score'],
                'sample_weight': best_row['sample_weight'],
                'total_observations': int(best_row['total_obs']),
                'unique_patients': int(best_row['unique_patients']),
                'confidence_level': confidence_level,
                'strategy_used': selected_strategy,
                'y_fraction_variability': best_row['y_fraction_variability'],
                'all_weeks': week_stats.to_dict('records')
            }

            print(f"\n聚类{cluster} 优化结果:")
            print(f"  推荐孕周: 第{best_row['week']:.0f}周")
            print(f"  选择策略: {selected_strategy}")
            print(f"  Y达标率: {best_row['compliance_rate'] * 100:.1f}%")
            print(f"  Wilson下界: {best_row['wilson_lb'] * 100:.1f}%")
            print(f"  风险因子: {best_row['risk_factor']:.3f}")
            print(f"  综合评分: {best_row['comprehensive_score']:.3f}")
            print(f"  稳定性评分: {best_row['stability_score']:.3f}")
            print(f"  一致性评分: {best_row['consistency_score']:.3f}")
            print(f"  置信度: {confidence_level}")

    def error_impact_analysis(self, n_simulations=1000,
                              y_error_sd=0.01, week_error_sd=0.2,
                              age_error_sd=1.0, height_error_sd=0.02, weight_error_sd=1.0):
        """增强的误差影响蒙特卡洛分析（多因素误差）"""
        print(f"\n{'=' * 60}")
        print(f"🔬 多因素误差影响分析 ({n_simulations}次蒙特卡洛模拟)")
        print(f"Y浓度误差: ±{y_error_sd * 100:.1f}%, 孕周误差: ±{week_error_sd:.1f}周")
        print(
            f"年龄误差: ±{age_error_sd:.1f}岁, 身高误差: ±{height_error_sd * 100:.1f}cm, 体重误差: ±{weight_error_sd:.1f}kg")
        print(f"{'=' * 60}")

        original_timing = self.timing_results.copy()
        all_simulation_results = []

        print(f"\n📈 执行多因素误差模拟...")
        valid_simulations = 0

        for i in range(n_simulations):
            if i % 200 == 0:
                print(f"  进度: {i}/{n_simulations}, 有效模拟: {valid_simulations}")

            try:
                # 生成多因素正态分布误差
                df_error = self.df.copy()
                n_samples = len(df_error)

                # 正态分布误差
                y_errors = np.random.normal(0, y_error_sd, n_samples)
                week_errors = np.random.normal(0, week_error_sd, n_samples)
                age_errors = np.random.normal(0, age_error_sd, n_samples)
                height_errors = np.random.normal(0, height_error_sd, n_samples)
                weight_errors = np.random.normal(0, weight_error_sd, n_samples)

                # 应用误差（确保数据合理性）
                df_error['Y_Fraction'] = np.maximum(0, df_error['Y_Fraction'] + y_errors)
                df_error['Y_Compliant'] = (df_error['Y_Fraction'] >= 0.04).astype(int)
                df_error['GestationalWeek'] = np.clip(
                    df_error['GestationalWeek'] + week_errors, 10, 30
                )
                df_error['Age'] = np.maximum(18, df_error['Age'] + age_errors)
                df_error['Height'] = np.maximum(1.4, df_error['Height'] + height_errors)
                df_error['Weight'] = np.maximum(40, df_error['Weight'] + weight_errors)

                # 重新计算BMI（因为体重和身高可能有误差）
                df_error['BMI'] = df_error['Weight'] / (df_error['Height'] ** 2)

                # 重新执行PCA和聚类
                pca_features = ['Age', 'Height', 'Weight']
                pca_data = df_error[pca_features].copy()

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)

                pca_result = self.pca_model.transform(scaled_data)

                # 使用高斯混合模型进行聚类
                gmm = GaussianMixture(n_components=len(original_timing), random_state=42)
                df_error['Cluster'] = gmm.fit_predict(pca_result)

                # 重新计算最佳时点（简化版本）
                sim_timing = {}

                for cluster in original_timing.keys():
                    group_data = df_error[df_error['Cluster'] == cluster].copy()

                    if len(group_data) < 3:
                        continue

                    try:
                        # 简化计算：只计算整体达标率，使用原始最佳孕周
                        original_best_week = original_timing[cluster]['best_week']

                        # 计算该组的总体达标率
                        overall_rate = group_data['Y_Compliant'].mean()

                        # 如果该组在原始最佳孕周附近有数据，使用那个孕周的数据
                        week_range = range(max(10, original_best_week - 2),
                                           min(31, original_best_week + 3))

                        best_week = original_best_week
                        best_rate = overall_rate
                        best_score = overall_rate * (1 - self.risk_function(original_best_week))

                        # 尝试在孕周范围内找更好的选择
                        for week in week_range:
                            week_data = group_data[
                                group_data['GestationalWeek'].between(week - 0.5, week + 0.5)
                            ]
                            if len(week_data) > 0:
                                week_rate = week_data['Y_Compliant'].mean()
                                week_score = week_rate * (1 - self.risk_function(week))
                                if week_score > best_score:
                                    best_week = week
                                    best_rate = week_rate
                                    best_score = week_score

                        sim_timing[cluster] = {
                            'best_week': int(best_week),
                            'compliance_rate': float(best_rate),
                            'quick_score': float(best_score)
                        }

                    except Exception as group_error:
                        # 如果计算失败，使用原始结果的简化版本
                        try:
                            overall_rate = group_data['Y_Compliant'].mean()
                            original_week = original_timing[cluster]['best_week']
                            sim_timing[cluster] = {
                                'best_week': original_week,
                                'compliance_rate': float(overall_rate),
                                'quick_score': float(overall_rate * (1 - self.risk_function(original_week)))
                            }
                        except:
                            continue

                if sim_timing:
                    all_simulation_results.append(sim_timing)
                    valid_simulations += 1

            except Exception as e:
                print(f"  模拟{i}失败: {str(e)[:50]}...")
                continue

        print(f"\n模拟完成！总计有效模拟: {valid_simulations}/{n_simulations}")

        if valid_simulations < 100:
            print(f"警告：有效模拟次数较少 ({valid_simulations})，结果可能不够可靠")

        if not all_simulation_results:
            print("错误：没有有效的模拟结果！")
            return {}

        # 综合统计分析
        print(f"\n多因素误差影响综合分析结果:")
        print("=" * 100)

        error_impact_summary = {}

        for cluster in original_timing.keys():
            group_analysis = {
                'original_week': original_timing[cluster]['best_week'],
                'original_score': original_timing[cluster]['comprehensive_score']
            }

            print(f"\n聚类{cluster} 误差敏感性分析:")
            print(f"原始最佳孕周: 第{group_analysis['original_week']}周")
            print("-" * 60)
            print(f"{'误差类型':<12} {'平均孕周':<8} {'标准差':<8} {'95%CI':<12} {'稳定性':<8} {'评分影响':<8}")
            print("-" * 60)

            # 分析正态分布误差结果
            sim_weeks = [sim[cluster]['best_week'] for sim in all_simulation_results if cluster in sim]
            sim_scores = [sim[cluster]['quick_score'] for sim in all_simulation_results if cluster in sim]

            print(f"  调试信息: 聚类{cluster}有效模拟次数: {len(sim_weeks)}")

            if len(sim_weeks) < 50:
                print(f"聚类{cluster}: 有效模拟次数不足 (仅{len(sim_weeks)}次)，跳过分析")
                continue

            week_mean = np.mean(sim_weeks)
            week_std = np.std(sim_weeks)
            week_95ci = np.percentile(sim_weeks, [2.5, 97.5])
            stability = max(0, 1 - week_std / group_analysis['original_week'])
            score_impact = abs(np.mean(sim_scores) - group_analysis['original_score']) / group_analysis[
                'original_score']

            # 存储统计结果
            group_analysis['week_mean'] = week_mean
            group_analysis['week_std'] = week_std
            group_analysis['stability'] = stability
            group_analysis['score_impact'] = score_impact

            ci_str = f"[{week_95ci[0]:.1f},{week_95ci[1]:.1f}]"
            print(f"{'多因素':<12} {week_mean:<8.1f} {week_std:<8.2f} {ci_str:<12} "
                  f"{stability:<8.2f} {score_impact:<8.3f}")

            error_impact_summary[cluster] = group_analysis

        # 稳定性排名
        print(f"\n各组稳定性排名 (基于多因素误差):")
        stability_ranking = []
        for cluster, analysis in error_impact_summary.items():
            if 'stability' in analysis:
                stability_ranking.append((cluster, analysis['stability']))

        stability_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (cluster, stability) in enumerate(stability_ranking, 1):
            print(f"  {i}. 聚类{cluster}: 稳定性 {stability:.3f}")

        # 风险预警
        print(f"\n误差风险预警:")
        for cluster, analysis in error_impact_summary.items():
            if 'stability' in analysis:
                if analysis['stability'] < 0.8:
                    print(f"聚类{cluster}: 对检测误差敏感 (稳定性: {analysis['stability']:.3f})")
                elif analysis['stability'] > 0.95:
                    print(f"聚类{cluster}: 误差抗性良好 (稳定性: {analysis['stability']:.3f})")

        return error_impact_summary

    def create_enhanced_visualization(self):
        """结果可视化（多因素分析版）"""
        # 1. PCA降维可视化
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df['Cluster'].unique())))
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            ax1.scatter(cluster_data['PC1'], cluster_data['PC2'],
                        alpha=0.7, label=f'聚类{cluster}', color=colors[i])

        ax1.set_xlabel(f'PC1 ({self.pca_components["explained_variance_ratio"][0] * 100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({self.pca_components["explained_variance_ratio"][1] * 100:.1f}%)')
        ax1.set_title('PCA降维可视化')
        ax1.legend()
        plt.tight_layout()
        plt.savefig('1_PCA降维可视化.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 因素重要性条形图
        if self.factor_importance:
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)

            factors = list(self.factor_importance.keys())
            importance = list(self.factor_importance.values())

            colors = ['green' if x > 1 else 'red' for x in importance]
            bars = ax2.bar(factors, importance, color=colors, alpha=0.7)
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7)
            ax2.set_ylabel('风险比 (HR)')
            ax2.set_title('Cox模型因素重要性分析')
            ax2.tick_params(axis='x', rotation=45)

            for bar, imp in zip(bars, importance):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{imp:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('2_因素重要性分析.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 聚类特征雷达图
        if self.cluster_results:
            fig3 = plt.figure(figsize=(12, 8))

            n_clusters = len(self.cluster_results)
            n_cols = min(3, n_clusters)
            n_rows = (n_clusters + n_cols - 1) // n_cols

            for i, cluster in enumerate(self.cluster_results.keys()):
                ax = fig3.add_subplot(n_rows, n_cols, i + 1, projection='polar')

                stats = self.cluster_results[cluster]['stats']
                categories = ['Age', 'Height', 'Weight', 'BMI']
                values = [stats[cat]['mean'] for cat in categories]

                # 归一化
                max_vals = {cat: max([self.cluster_results[c]['stats'][cat]['mean']
                                      for c in self.cluster_results.keys()]) for cat in categories}
                min_vals = {cat: min([self.cluster_results[c]['stats'][cat]['mean']
                                      for c in self.cluster_results.keys()]) for cat in categories}

                norm_values = []
                for j, cat in enumerate(categories):
                    if max_vals[cat] - min_vals[cat] > 0:
                        norm_val = (values[j] - min_vals[cat]) / (max_vals[cat] - min_vals[cat])
                    else:
                        norm_val = 0.5
                    norm_values.append(norm_val)

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                norm_values += norm_values[:1]
                angles += angles[:1]

                ax.plot(angles, norm_values, 'o-', linewidth=2)
                ax.fill(angles, norm_values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_title(f'聚类{cluster}特征雷达图')

            plt.tight_layout()
            plt.savefig('3_聚类特征雷达图.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 各组最佳时点对比
        if self.timing_results:
            fig4 = plt.figure(figsize=(12, 8))
            ax4 = fig4.add_subplot(111)

            clusters = list(self.timing_results.keys())
            weeks = [self.timing_results[c]['best_week'] for c in clusters]
            scores = [self.timing_results[c]['comprehensive_score'] for c in clusters]

            # 根据评分设置颜色
            norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
            colors = plt.cm.RdYlGn(norm(scores))

            bars = ax4.bar([f'聚类{c}' for c in clusters], weeks, color=colors)
            ax4.set_ylabel('最佳检测孕周')
            ax4.set_title('各组最佳NIPT时点\n(颜色表示综合评分)')

            # 添加评分标签
            for bar, week, score in zip(bars, weeks, scores):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{week}周\n({score:.3f})', ha='center', va='bottom', fontsize=8)

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax4)
            cbar.set_label('综合评分')

            plt.tight_layout()
            plt.savefig('4_各组最佳时点对比.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. 多因素相关性热力图
        fig5 = plt.figure(figsize=(10, 8))
        ax5 = fig5.add_subplot(111)

        corr_cols = ['BMI', 'Age', 'Height', 'Weight', 'GestationalWeek', 'Y_Fraction']
        corr_matrix = self.df[corr_cols].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('多因素相关性热力图')
        plt.tight_layout()
        plt.savefig('5_多因素相关性热力图.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 6. 误差敏感性分析图
        if hasattr(self, 'error_impact') and self.error_impact:
            fig6 = plt.figure(figsize=(12, 6))
            ax6 = fig6.add_subplot(111)

            clusters_with_error = list(self.error_impact.keys())
            stability_scores = [self.error_impact[c].get('stability', 0) for c in clusters_with_error]

            bars = ax6.bar([f'聚类{c}' for c in clusters_with_error], stability_scores,
                           color=['green' if s > 0.9 else 'orange' if s > 0.8 else 'red' for s in stability_scores])
            ax6.set_ylabel('稳定性评分')
            ax6.set_title('各组检测误差稳定性')
            ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='稳定性阈值')
            ax6.legend()

            for bar, score in zip(bars, stability_scores):
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('6_误差敏感性分析.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("所有多因素可视化结果已保存")

    def generate_clinical_report(self):
        """生成多因素临床报告"""
        print(f"\n{'=' * 80}")
        print("临床应用报告：多因素男胎孕妇NIPT最佳检测时点")
        print(f"{'=' * 80}")

        # 创建汇总表
        summary_data = []
        for cluster in sorted(self.cluster_results.keys()):
            if cluster in self.cluster_results and cluster in self.timing_results:
                cluster_info = self.cluster_results[cluster]
                timing_info = self.timing_results[cluster]

                summary_data.append({
                    '聚类': f"聚类{cluster}",
                    'WHO分类': cluster_info['who_category'],
                    '年龄分类': cluster_info['age_category'],
                    'BMI范围': f"[{cluster_info['stats']['BMI']['min']:.1f}, {cluster_info['stats']['BMI']['max']:.1f}]",
                    'BMI均值': f"{cluster_info['stats']['BMI']['mean']:.1f}±{cluster_info['stats']['BMI']['std']:.1f}",
                    '年龄范围': f"[{cluster_info['stats']['Age']['min']:.1f}, {cluster_info['stats']['Age']['max']:.1f}]",
                    '样本量': f"{cluster_info['sample_size']}条/{cluster_info['patient_count']}人",
                    '最佳孕周': f"第{timing_info['best_week']}周",
                    'Y达标率': f"{timing_info['compliance_rate'] * 100:.1f}%",
                    '风险因子': f"{timing_info['risk_factor']:.3f}",
                    '置信度': timing_info['confidence_level']
                })

        # 按BMI均值排序
        summary_data.sort(key=lambda x: float(x['BMI均值'].split('±')[0]))

        # 打印表格
        print(f"\n多因素分组建议汇总表:")
        print("-" * 150)
        headers = ['聚类', 'WHO分类', '年龄分类', 'BMI均值', '最佳孕周', 'Y达标率', '风险因子', '置信度']
        header_line = " | ".join([f"{h:^10}" for h in headers])
        print(header_line)
        print("-" * len(header_line))

        for row in summary_data:
            data_line = " | ".join([f"{row[h]:^10}" for h in headers])
            print(data_line)

        print("-" * 150)

        # 临床建议
        print(f"\n多因素临床实施建议:")
        for i, row in enumerate(summary_data, 1):
            print(f"{i}. {row['聚类']} ({row['WHO分类']}, {row['年龄分类']})")
            print(f"   - BMI范围: {row['BMI范围']} kg/m²")
            print(f"   - 年龄范围: {row['年龄范围']} 岁")
            print(f"   - 推荐时机: {row['最佳孕周']}")
            print(f"   - 预期达标率: {row['Y达标率']}")

            risk = float(row['风险因子'])
            if risk < 0.3:
                risk_level = "低风险期"
            elif risk < 0.7:
                risk_level = "中风险期"
            else:
                risk_level = "高风险期"
            print(f"   - 风险评估: {risk_level}")

        # 因素重要性说明
        if self.factor_importance:
            print(f"\n因素重要性分析:")
            sorted_factors = sorted(self.factor_importance.items(), key=lambda x: abs(x[1] - 1), reverse=True)
            for factor, importance in sorted_factors:
                effect = "增加风险" if importance > 1 else "降低风险"
                print(f"   - {factor}: 风险比 = {importance:.3f} ({effect})")

        # 保存结果
        results_df = pd.DataFrame(summary_data)
        results_df.to_excel('多因素_NIPT最佳时点结果.xlsx', index=False)
        print(f"\n结果已保存: 多因素_NIPT最佳时点结果.xlsx")

        return summary_data

    def run_analysis(self, simulation_count=800):
        """运行多因素完整分析流程"""
        print("开始多因素NIPT最佳时点分析")
        print("=" * 80)

        # 1. 加载数据和预处理
        if not self.load_and_preprocess_data():
            return

        # 2. PCA降维和聚类分析
        print("\n阶段1: PCA降维和聚类分析...")
        self.perform_pca_clustering(n_clusters=5, n_components=2)

        # 3. 生存分析
        print("\n阶段2: 生存分析...")
        self.survival_analysis()

        # 4. 优化的最佳检测时点计算
        print("\n阶段3: 优化最佳时点计算...")
        self.calculate_optimal_timing()

        # 5. 多因素误差影响分析
        print("\n阶段4: 多因素误差影响分析...")
        self.error_impact = self.error_impact_analysis(
            n_simulations=simulation_count,
            y_error_sd=0.01,  # Y染色体浓度误差±1%
            week_error_sd=0.2,  # 孕周误差±0.2周
            age_error_sd=1.0,  # 年龄误差±1岁
            height_error_sd=0.02,  # 身高误差±0.02m(2cm)
            weight_error_sd=1.0  # 体重误差±1kg
        )

        # 6. 多因素可视化分析
        print("\n阶段5: 生成多因素可视化...")
        self.create_enhanced_visualization()

        # 7. 生成临床报告
        print("\n阶段6: 生成临床应用报告...")
        clinical_summary = self.generate_clinical_report()

        # 8. 总结分析
        print(f"\n多因素分析总结:")
        print("=" * 60)
        total_clusters = len(self.timing_results)
        high_confidence = sum(1 for r in self.timing_results.values() if r['confidence_level'] == '高置信度')
        avg_score = np.mean([r['comprehensive_score'] for r in self.timing_results.values()])

        print(f"成功分析 {total_clusters} 个多因素聚类")
        print(f"高置信度推荐: {high_confidence}/{total_clusters} 组")
        print(f"平均综合评分: {avg_score:.3f}")
        print(f"多因素误差分析: 已完成 {simulation_count} 次模拟")

        if hasattr(self, 'error_impact'):
            stable_clusters = sum(1 for analysis in self.error_impact.values()
                                  if analysis.get('stability', 0) > 0.9)
            print(f"高稳定性聚类: {stable_clusters}/{total_clusters} 组")

        # 输出最重要因素
        if self.factor_importance:
            most_important = max(self.factor_importance.items(), key=lambda x: abs(x[1] - 1))
            effect = "增加" if most_important[1] > 1 else "减少"
            print(f"最重要影响因素: {most_important[0]} (风险比: {most_important[1]:.3f}, {effect}风险)")

        print(f"\n多因素分析完成！所有结果已保存。")

        return {
            'timing_results': self.timing_results,
            'error_impact': self.error_impact if hasattr(self, 'error_impact') else None,
            'clinical_summary': clinical_summary,
            'factor_importance': self.factor_importance
        }


if __name__ == "__main__":
    analyzer = NIPTTimingAnalyzer()
    analyzer.run_analysis()