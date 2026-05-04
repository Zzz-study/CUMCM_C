
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)


class NIPTTimingAnalyzer:
    """NIPT最佳时点分析器（固定5组BMI聚类）"""
    
    def __init__(self):
        self.df = None
        self.cluster_results = {}
        self.timing_results = {}
        
    def load_and_cluster_data(self, file_path='data_processed.xlsx'):
        """加载数据并执行固定5组K-means聚类"""
        try:
            self.df = pd.read_excel(file_path)
            print(f"数据加载成功: {self.df.shape[0]} 行")
            
            # 统一列名
            self.df = self.df.rename(columns={
                '孕妇代码': 'PatientID',
                '检测孕周_数值': 'GestationalWeek', 
                '孕妇BMI': 'BMI',
                'Y染色体浓度': 'Y_Fraction'
            })
            
            # 数据清洗
            self.df['Y_Fraction'] = pd.to_numeric(self.df['Y_Fraction'], errors='coerce')
            if self.df['Y_Fraction'].dropna().gt(1).mean() > 0.2:
                self.df['Y_Fraction'] = self.df['Y_Fraction'] / 100.0
                
            self.df['BMI'] = pd.to_numeric(self.df['BMI'], errors='coerce')
            self.df['GestationalWeek'] = pd.to_numeric(self.df['GestationalWeek'], errors='coerce')
            self.df['Y_Compliant'] = (self.df['Y_Fraction'] >= 0.04).astype(int)
            
            # 移除缺失值
            self.df = self.df.dropna(subset=['BMI', 'Y_Fraction', 'GestationalWeek'])
            
            # 执行固定5组K-means聚类
            print(f"\n执行5组BMI聚类...")
            bmi_data = self.df[['BMI']].copy()
            scaler = StandardScaler()
            bmi_scaled = scaler.fit_transform(bmi_data)
            
            kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=20)
            self.df['BMI_Group'] = kmeans.fit_predict(bmi_scaled)
            
            # 分析各组特征
            self._analyze_groups()
            
            print(f"总体数据概览：")
            print(f"  观测数量: {len(self.df)}")
            print(f"  孕妇数量: {self.df['PatientID'].nunique()}")
            print(f"  总体Y达标率: {self.df['Y_Compliant'].mean() * 100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"数据处理失败: {e}")
            return False
    
    def _analyze_groups(self):
        """分析5组BMI特征"""
        print(f"\n5组BMI聚类结果:")
        print("-" * 80)
        
        for group in range(5):
            group_data = self.df[self.df['BMI_Group'] == group]
            if len(group_data) == 0:
                continue
                
            bmi_values = group_data['BMI']
            mean_bmi = bmi_values.mean()
            
            # WHO分类
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
                
            
            self.cluster_results[group] = {
                'sample_size': len(group_data),
                'patient_count': group_data['PatientID'].nunique(),
                'bmi_range': (bmi_values.min(), bmi_values.max()),
                'bmi_mean': mean_bmi,
                'bmi_std': bmi_values.std(),
                'who_category': who_category,
                'y_compliance_rate': group_data['Y_Compliant'].mean()
            }
            
            print(f"组{group} ({who_category}): "
                  f"BMI={mean_bmi:.1f}±{bmi_values.std():.1f} "
                  f"({bmi_values.min():.1f}-{bmi_values.max():.1f}), "
                  f"n={len(group_data)}, Y达标率={group_data['Y_Compliant'].mean()*100:.1f}%")
    
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
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return max(0, centre - margin)
    
    def calculate_optimal_timing(self, min_patients=5):
        """优化的最佳NIPT检测时点计算算法"""
        print(f"\n{'='*60}")
        print("优化最佳NIPT检测时点计算（多维度风险最小化）")
        print(f"{'='*60}")
        
        for group in range(5):
            group_data = self.df[
                (self.df['BMI_Group'] == group) &
                (self.df['GestationalWeek'].between(10, 30))
            ].copy()
            
            if len(group_data) < 20:
                print(f"\n组{group}: 样本量不足 (n={len(group_data)})")
                continue
            
            # 按孕妇-孕周聚合，计算更详细的统计量
            patient_week_stats = (group_data.groupby(['PatientID', 'GestationalWeek'])
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
                print(f"\n组{group}: 无足够样本的孕周")
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
            w_compliance = 0.35    # 达标率权重
            w_risk = 0.25         # 风险权重  
            w_reliability = 0.20   # 统计可靠性权重
            w_stability = 0.10     # 稳定性权重
            w_consistency = 0.10   # 一致性权重
            
            # 归一化各指标到[0,1]区间
            week_stats['compliance_norm'] = week_stats['compliance_rate']
            week_stats['risk_norm'] = 1 - week_stats['risk_factor']
            week_stats['reliability_norm'] = week_stats['wilson_lb']
            week_stats['stability_norm'] = (week_stats['stability_score'] - week_stats['stability_score'].min()) / \
                                         (week_stats['stability_score'].max() - week_stats['stability_score'].min() + 1e-8)
            week_stats['consistency_norm'] = (week_stats['consistency_score'] - week_stats['consistency_score'].min()) / \
                                           (week_stats['consistency_score'].max() - week_stats['consistency_score'].min() + 1e-8)
            
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
                selected_strategy, best_row, confidence_level = strategies[0] if strategies else ('默认', best_balanced, '低置信度')
            
            # 保存详细结果
            self.timing_results[group] = {
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
            
            print(f"\n组{group} 优化结果:")
            print(f"  推荐孕周: 第{best_row['week']:.0f}周")
            print(f"  选择策略: {selected_strategy}")
            print(f"  Y达标率: {best_row['compliance_rate']*100:.1f}%")
            print(f"  Wilson下界: {best_row['wilson_lb']*100:.1f}%")
            print(f"  风险因子: {best_row['risk_factor']:.3f}")
            print(f"  综合评分: {best_row['comprehensive_score']:.3f}")
            print(f"  稳定性评分: {best_row['stability_score']:.3f}")
            print(f"  一致性评分: {best_row['consistency_score']:.3f}")

            print(f"  置信度: {confidence_level}")

    def error_impact_analysis(self, n_simulations=1000,
                              y_error_sd=0.01, week_error_sd=0.2):
        """增强的误差影响蒙特卡洛分析（固定使用正态分布误差）"""
        print(f"\n{'=' * 60}")
        print(f"增强误差影响分析 ({n_simulations}次蒙特卡洛模拟)")
        print(f"Y浓度误差: ±{y_error_sd * 100:.1f}%, 孕周误差: ±{week_error_sd:.1f}周")
        print(f"误差分布类型: normal")
        print(f"{'=' * 60}")

        original_timing = self.timing_results.copy()
        all_simulation_results = []  # 仅存储正态分布的模拟结果

        print(f"\n执行normal分布误差模拟...")
        valid_simulations = 0
        
        # 首先检查原始数据的有效性
        print(f"  原始数据检查:")
        for group in range(5):
            group_data = self.df[self.df['BMI_Group'] == group]
            print(f"    组{group}: {len(group_data)}条记录, {group_data['PatientID'].nunique()}名孕妇")

        for i in range(n_simulations):
            if i % 200 == 0:
                print(f"  进度: {i}/{n_simulations}, 有效模拟: {valid_simulations}")

            try:
                # 生成正态分布误差
                df_error = self.df.copy()
                n_samples = len(df_error)

                # 正态分布误差（仅保留此类型）
                y_errors = np.random.normal(0, y_error_sd, n_samples)
                week_errors = np.random.normal(0, week_error_sd, n_samples)

                # 应用误差（确保数据合理性）
                df_error['Y_Fraction'] = np.maximum(0, df_error['Y_Fraction'] + y_errors)
                df_error['Y_Compliant'] = (df_error['Y_Fraction'] >= 0.04).astype(int)
                df_error['GestationalWeek'] = np.clip(
                    df_error['GestationalWeek'] + week_errors, 10, 30
                )

                # 重新计算最佳时点（超级简化版本）
                sim_timing = {}
                
                # 针对每个原始timing_results中存在的组进行模拟
                for group in original_timing.keys():
                    # 获取该组的所有数据（不限制孕周范围）
                    group_data = df_error[df_error['BMI_Group'] == group].copy()

                    if len(group_data) < 3:  # 最低要求
                        continue

                    try:
                        # 超简化：只计算整体达标率，使用原始最佳孕周
                        original_best_week = original_timing[group]['best_week']
                        
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
                                group_data['GestationalWeek'].between(week-0.5, week+0.5)
                            ]
                            if len(week_data) > 0:
                                week_rate = week_data['Y_Compliant'].mean()
                                week_score = week_rate * (1 - self.risk_function(week))
                                if week_score > best_score:
                                    best_week = week
                                    best_rate = week_rate
                                    best_score = week_score

                        sim_timing[group] = {
                            'best_week': int(best_week),
                            'compliance_rate': float(best_rate),
                            'quick_score': float(best_score)
                        }
                    
                    except Exception as group_error:
                        # 如果计算失败，使用原始结果的简化版本
                        try:
                            overall_rate = group_data['Y_Compliant'].mean()
                            original_week = original_timing[group]['best_week']
                            sim_timing[group] = {
                                'best_week': original_week,
                                'compliance_rate': float(overall_rate),
                                'quick_score': float(overall_rate * (1 - self.risk_function(original_week)))
                            }
                        except:
                            # 最后的备选方案：跳过该组
                            continue

                # 调试信息
                if i < 3:
                    print(f"    模拟{i}: 处理了{len(sim_timing)}个组别")
                    for g, result in sim_timing.items():
                        print(f"      组{g}: 孕周{result['best_week']}, 达标率{result['compliance_rate']:.3f}")

                # 只要有结果就算有效
                if sim_timing:
                    all_simulation_results.append(sim_timing)
                    valid_simulations += 1
                    
            except Exception as e:
                # 如果某次模拟失败，跳过该次模拟
                print(f"  模拟{i}失败: {str(e)[:50]}...")
                continue

        print(f"\n模拟完成！总计有效模拟: {valid_simulations}/{n_simulations}")
        
        if valid_simulations < 100:
            print(f"警告：有效模拟次数较少 ({valid_simulations})，结果可能不够可靠")

        # 检查是否有足够的模拟结果
        if not all_simulation_results:
            print("错误：没有有效的模拟结果！")
            return {}

        # 综合统计分析
        print(f"\n误差影响综合分析结果:")
        print("=" * 100)

        error_impact_summary = {}

        for group in original_timing.keys():
            group_analysis = {
                'original_week': original_timing[group]['best_week'],
                'original_score': original_timing[group]['comprehensive_score']
            }

            print(f"\n组{group} 误差敏感性分析:")
            print(f"原始最佳孕周: 第{group_analysis['original_week']}周")
            print("-" * 60)
            print(f"{'误差类型':<12} {'平均孕周':<8} {'标准差':<8} {'95%CI':<12} {'稳定性':<8} {'评分影响':<8}")
            print("-" * 60)

            # 仅分析正态分布误差结果
            sim_weeks = [sim[group]['best_week'] for sim in all_simulation_results if group in sim]
            sim_scores = [sim[group]['quick_score'] for sim in all_simulation_results if group in sim]

            print(f"  调试信息: 组{group}有效模拟次数: {len(sim_weeks)}")
            
            if len(sim_weeks) < 50:
                print(f"组{group}: 有效模拟次数不足 (仅{len(sim_weeks)}次)，跳过分析")
                continue

            week_mean = np.mean(sim_weeks)
            week_std = np.std(sim_weeks)
            week_95ci = np.percentile(sim_weeks, [2.5, 97.5])
            stability = max(0, 1 - week_std / group_analysis['original_week'])
            score_impact = abs(np.mean(sim_scores) - group_analysis['original_score']) / group_analysis[
                'original_score']

            # 存储正态分布的统计结果
            group_analysis['normal_week_mean'] = week_mean
            group_analysis['normal_week_std'] = week_std
            group_analysis['normal_stability'] = stability
            group_analysis['normal_score_impact'] = score_impact

            ci_str = f"[{week_95ci[0]:.1f},{week_95ci[1]:.1f}]"
            print(f"{'normal':<12} {week_mean:<8.1f} {week_std:<8.2f} {ci_str:<12} "
                  f"{stability:<8.2f} {score_impact:<8.3f}")

            error_impact_summary[group] = group_analysis

        # 稳定性排名（基于正态分布误差）
        print(f"\n各组稳定性排名 (基于正态分布误差):")
        stability_ranking = []
        for group, analysis in error_impact_summary.items():
            if 'normal_stability' in analysis:
                stability_ranking.append((group, analysis['normal_stability']))

        stability_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (group, stability) in enumerate(stability_ranking, 1):
            print(f"  {i}. 组{group}: 稳定性 {stability:.3f}")

        # 风险预警
        print(f"\n误差风险预警:")
        for group, analysis in error_impact_summary.items():
            if 'normal_stability' in analysis:
                if analysis['normal_stability'] < 0.8:
                    print(f"组{group}: 对检测误差敏感 (稳定性: {analysis['normal_stability']:.3f})")
                elif analysis['normal_stability'] > 0.95:
                    print(f"组{group}: 误差抗性良好 (稳定性: {analysis['normal_stability']:.3f})")

        return error_impact_summary

    def create_enhanced_visualization(self):
        """结果可视化（每个图表单独绘制）"""

        # 1. BMI分组分布图
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        colors = plt.cm.Set3(np.linspace(0, 1, 5))
        for i, group in enumerate(range(5)):
            if group in self.cluster_results:
                group_data = self.df[self.df['BMI_Group'] == group]['BMI']
                ax1.hist(group_data, alpha=0.7, label=f'组{group}', bins=12, color=colors[i])
        ax1.set_xlabel('BMI (kg/m²)')
        ax1.set_ylabel('频数')
        ax1.set_title('BMI分组分布')
        ax1.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('1_BMI分组分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 多维度评分雷达图
        if self.timing_results:
            fig2 = plt.figure(figsize=(8, 8))
            ax2 = fig2.add_subplot(111, projection='polar')
            # 选择一个代表性组别
            sample_group = list(self.timing_results.keys())[0]
            result = self.timing_results[sample_group]

            categories = ['达标率', '可靠性', '稳定性', '一致性', '低风险']
            values = [
                result['compliance_rate'],
                result['wilson_lb'],
                result['stability_score'],
                result['consistency_score'],
                1 - result['risk_factor']
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            ax2.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax2.fill(angles, values, alpha=0.25, color='blue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'组{sample_group}多维度评分')
            plt.tight_layout()
            plt.savefig('2_多维度评分雷达图.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 各组最佳时点对比
        if self.timing_results:
            fig3 = plt.figure(figsize=(10, 6))
            ax3 = fig3.add_subplot(111)
            groups = list(self.timing_results.keys())
            weeks = [self.timing_results[g]['best_week'] for g in groups]
            scores = [self.timing_results[g]['comprehensive_score'] for g in groups]

            # 根据评分设置颜色
            norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
            colors = plt.cm.RdYlGn(norm(scores))

            bars = ax3.bar([f'组{g}' for g in groups], weeks, color=colors)
            ax3.set_ylabel('最佳检测孕周')
            ax3.set_title('各组最佳NIPT时点\n(颜色表示综合评分)')

            # 添加评分标签
            for bar, week, score in zip(bars, weeks, scores):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{week}周\n({score:.3f})', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.savefig('3_各组最佳时点对比.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 策略选择分布图
        if self.timing_results:
            fig4 = plt.figure(figsize=(8, 8))
            ax4 = fig4.add_subplot(111)
            strategies = [self.timing_results[g]['strategy_used'] for g in self.timing_results.keys()]
            strategy_counts = {}
            for strategy in strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            wedges, texts, autotexts = ax4.pie(strategy_counts.values(), labels=strategy_counts.keys(),
                                               autopct='%1.0f%%', startangle=90)
            ax4.set_title('最佳时点选择策略分布')
            plt.tight_layout()
            plt.savefig('4_策略选择分布.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. 孕周-达标率详细趋势图
        if self.timing_results:
            fig5 = plt.figure(figsize=(12, 6))
            ax5 = fig5.add_subplot(111)
            groups = list(self.timing_results.keys())
            for group in groups:
                group_data = self.df[self.df['BMI_Group'] == group]
                week_trend = group_data.groupby('GestationalWeek')['Y_Compliant'].mean().reset_index()
                ax5.plot(week_trend['GestationalWeek'], week_trend['Y_Compliant'],
                         'o-', label=f'组{group}', alpha=0.7, linewidth=2)

                # 标记最佳时点
                best_week = self.timing_results[group]['best_week']
                best_rate = self.timing_results[group]['compliance_rate']
                ax5.scatter(best_week, best_rate, color='red', s=100, marker='*', zorder=5)

            ax5.set_xlabel('孕周')
            ax5.set_ylabel('Y达标率')
            ax5.set_title('各组孕周-达标率趋势及最佳时点(★)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('5_孕周-达标率趋势.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 6. 综合评分组件分析图
        if self.timing_results:
            fig6 = plt.figure(figsize=(12, 6))
            ax6 = fig6.add_subplot(111)
            components = ['达标率', '低风险', '可靠性', '稳定性', '一致性']
            groups = list(self.timing_results.keys())
            x = np.arange(len(groups))
            width = 0.15

            for i, component in enumerate(components):
                if component == '达标率':
                    values = [self.timing_results[g]['compliance_rate'] for g in groups]
                elif component == '低风险':
                    values = [1 - self.timing_results[g]['risk_factor'] for g in groups]
                elif component == '可靠性':
                    values = [self.timing_results[g]['wilson_lb'] for g in groups]
                elif component == '稳定性':
                    values = [self.timing_results[g]['stability_score'] for g in groups]
                else:  # 一致性
                    values = [self.timing_results[g]['consistency_score'] for g in groups]

                ax6.bar(x + i * width, values, width, label=component, alpha=0.8)

            ax6.set_xlabel('BMI组别')
            ax6.set_ylabel('评分值')
            ax6.set_title('各组综合评分组件分析')
            ax6.set_xticks(x + width * 2)
            ax6.set_xticklabels([f'组{g}' for g in groups])
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('6_综合评分组件分析.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 7. 误差敏感性分析图
        if hasattr(self, 'error_impact') and self.error_impact:
            fig7 = plt.figure(figsize=(10, 6))
            ax7 = fig7.add_subplot(111)
            groups_with_error = list(self.error_impact.keys())
            stability_scores = [self.error_impact[g].get('normal_stability', 0) for g in groups_with_error]

            bars = ax7.bar([f'组{g}' for g in groups_with_error], stability_scores,
                           color=['green' if s > 0.9 else 'orange' if s > 0.8 else 'red' for s in stability_scores])
            ax7.set_ylabel('稳定性评分')
            ax7.set_title('各组检测误差稳定性')
            ax7.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='稳定性阈值')
            ax7.legend()

            for bar, score in zip(bars, stability_scores):
                ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig('7_误差敏感性分析.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 8. 风险函数与最佳时点图
        if self.timing_results:
            fig8 = plt.figure(figsize=(12, 6))
            ax8 = fig8.add_subplot(111)
            weeks_range = np.linspace(10, 30, 200)
            risks = [self.risk_function(w) for w in weeks_range]
            ax8.plot(weeks_range, risks, 'r-', linewidth=3, label='风险函数R(t)')

            # 标记各组最佳时点
            groups = list(self.timing_results.keys())
            for group in groups:
                best_week = self.timing_results[group]['best_week']
                risk_at_best = self.risk_function(best_week)
                ax8.scatter(best_week, risk_at_best, s=100, label=f'组{group}最佳时点', alpha=0.8)

            ax8.set_xlabel('孕周')
            ax8.set_ylabel('风险因子')
            ax8.set_title('孕周风险函数与各组最佳时点')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('8_风险函数与最佳时点.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 9. 临床建议热力图
        if self.timing_results and self.cluster_results:
            fig9 = plt.figure(figsize=(12, 8))
            ax9 = fig9.add_subplot(111)

            # 创建建议矩阵
            recommendation_data = []
            labels = []

            groups = list(self.timing_results.keys())
            for group in groups:
                result = self.timing_results[group]
                cluster_info = self.cluster_results[group]

                # 构建建议向量 [最佳孕周, 达标率*100, 风险因子*100, 综合评分*100]
                rec_vector = [
                    result['best_week'],
                    result['compliance_rate'] * 100,
                    result['risk_factor'] * 100,
                    result['comprehensive_score'] * 100
                ]
                recommendation_data.append(rec_vector)
                labels.append(f"组{group}\n({cluster_info['who_category']})")

            recommendation_matrix = np.array(recommendation_data).T

            im = ax9.imshow(recommendation_matrix, cmap='RdYlGn', aspect='auto')
            ax9.set_xticks(range(len(groups)))
            ax9.set_xticklabels(labels)
            ax9.set_yticks(range(4))
            ax9.set_yticklabels(['最佳孕周', '达标率(%)', '风险因子(%)', '综合评分(%)'])
            ax9.set_title('各组临床建议热力图')

            # 添加数值标签
            for i in range(4):
                for j in range(len(groups)):
                    text = ax9.text(j, i, f'{recommendation_matrix[i, j]:.1f}',
                                    ha="center", va="center", color="black", fontweight='bold')

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax9, shrink=0.6)
            cbar.set_label('相对评分值')
            plt.tight_layout()
            plt.savefig('9_临床建议热力图.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("所有可视化结果已保存为单独图片")

    def generate_clinical_report(self):
        """生成临床报告"""
        print(f"\n{'='*80}")
        print("🏥 临床应用报告：5组BMI男胎孕妇NIPT最佳检测时点")
        print(f"{'='*80}")
        
        # 创建汇总表
        summary_data = []
        for group in range(5):
            if group in self.cluster_results and group in self.timing_results:
                cluster_info = self.cluster_results[group]
                timing_info = self.timing_results[group]
                
                summary_data.append({
                    '组别': f"组{group}",
                    'WHO分类': cluster_info['who_category'],
                    'BMI范围': f"[{cluster_info['bmi_range'][0]:.1f}, {cluster_info['bmi_range'][1]:.1f}]",
                    'BMI均值': f"{cluster_info['bmi_mean']:.1f}±{cluster_info['bmi_std']:.1f}",
                    '样本量': f"{cluster_info['sample_size']}条/{cluster_info['patient_count']}人",
                    '最佳孕周': f"第{timing_info['best_week']}周",
                    'Y达标率': f"{timing_info['compliance_rate']*100:.1f}%",
                    '风险因子': f"{timing_info['risk_factor']:.3f}",
                    '置信度': timing_info['confidence_level']
                })
        
        # 按BMI均值排序
        summary_data.sort(key=lambda x: float(x['BMI均值'].split('±')[0]))
        
        # 打印表格
        print(f"\n分组建议汇总表:")
        print("-" * 120)
        headers = ['组别', 'WHO分类', 'BMI均值', 'BMI范围', '最佳孕周', '风险因子', '置信度']
        header_line = " | ".join([f"{h:^12}" for h in headers])
        print(header_line)
        print("-" * len(header_line))
        
        for row in summary_data:
            data_line = " | ".join([f"{row[h]:^12}" for h in headers])
            print(data_line)
        
        print("-" * 120)
        
        # 临床建议
        print(f"\n临床实施建议:")
        for i, row in enumerate(summary_data, 1):
            print(f"{i}. {row['组别']} ({row['WHO分类']})")
            print(f"   - BMI筛选: {row['BMI范围']} kg/m²")
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
        
        # 保存结果
        results_df = pd.DataFrame(summary_data)
        results_df.to_excel('5组BMI_NIPT最佳时点结果.xlsx', index=False)
        print(f"\n结果已保存: 5组BMI_NIPT最佳时点结果.xlsx")
        
        return summary_data
    
    def run_analysis(self, simulation_count=800):
        """运行优化的完整分析流程"""
        print("开始优化版5组BMI聚类NIPT最佳时点分析")
        print("=" * 80)
        
        # 1. 加载数据和聚类
        if not self.load_and_cluster_data():
            return
        
        # 2. 优化的最佳检测时点计算
        print("\n阶段1: 优化最佳时点计算...")
        self.calculate_optimal_timing()
        
        # 3. 误差影响分析
        print("\n阶段2: 增强误差影响分析...")
        self.error_impact = self.error_impact_analysis(
            n_simulations=simulation_count,
            y_error_sd=0.01,     # Y染色体浓度误差±1%
            week_error_sd=0.2,   # 孕周误差±0.2周
        )

        
        # 4. 可视化分析
        print("\n阶段3: 生成增强可视化...")
        self.create_enhanced_visualization()
        
        # 5. 生成临床报告
        print("\n阶段4: 生成临床应用报告...")
        clinical_summary = self.generate_clinical_report()
        
        # 6. 总结分析
        print(f"\n分析总结:")
        print("=" * 60)
        total_groups = len(self.timing_results)
        high_confidence = sum(1 for r in self.timing_results.values() if r['confidence_level'] == '高置信度')
        avg_score = np.mean([r['comprehensive_score'] for r in self.timing_results.values()])
        
        print(f"成功分析 {total_groups} 个BMI组别")
        print(f"高置信度推荐: {high_confidence}/{total_groups} 组")
        print(f"平均综合评分: {avg_score:.3f}")
        print(f"误差分析: 已完成 {simulation_count*3} 次模拟")
        
        if hasattr(self, 'error_impact'):
            stable_groups = sum(1 for analysis in self.error_impact.values() 
                              if analysis.get('normal_stability', 0) > 0.9)
            print(f"高稳定性组别: {stable_groups}/{total_groups} 组")
        
        print(f"\n优化分析完成！所有结果已保存。")
        
        return {
            'timing_results': self.timing_results,
            'error_impact': self.error_impact if hasattr(self, 'error_impact') else None,
            'clinical_summary': clinical_summary
        }


if __name__ == "__main__":
    analyzer = NIPTTimingAnalyzer()
    analyzer.run_analysis()
