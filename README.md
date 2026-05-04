# CUMCM 数学建模竞赛代码

本项目包含全国大学生数学建模竞赛（CUMCM）的相关代码，主要针对NIPT（无创产前基因检测）数据进行分析，分为四个子问题（Q1-Q4）。

## 项目结构

```
CUMCM/
├── Q1/                          # 问题1：Y染色体浓度达标分析
│   ├── Q1.py                   # 贝叶斯Beta-Binomial GLMM模型
│   ├── Q1可视化.py              # GLMM模型结果可视化
│   ├── bbglmm_pred_curve.xlsx   # 预测曲线数据
│   └── bbglmm_summary.xlsx      # 模型参数摘要
├── Q2/                          # 问题2：BMI聚类与NIPT最佳时点分析
│   ├── data_process_1.py       # BMI数据核密度分布可视化
│   ├── bmi_clustering_analysis - 无PSO.py  # BMI聚类分析（无PSO优化）
│   ├── simplified_nipt_timing_analysis.py  # 简化版NIPT时点分析
│   ├── data_processed.xlsx      # 处理后数据
│   └── 5组BMI_NIPT最佳时点结果.xlsx  # 分析结果
├── Q3/                          # 问题3：多因素NIPT最佳时点分析
│   ├── complex_nipt_timing_analysis1.py  # 多因素聚类分析
│   ├── data_processed.xlsx      # 处理后数据
│   └── 多因素_NIPT最佳时点结果.xlsx  # 分析结果
└── Q4/                          # 问题4：女胎数据异常检测
    ├── test.py                  # 机器学习分类模型（SMOTE+不平衡集成）
    ├── 女胎孕妇.xlsx            # 女胎检测数据
    └── 代码运行结果.txt         # 运行结果记录
```

## 环境要求

- Python 3.8+
- 主要依赖包：
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pymc
  - arviz
  - imblearn
  - scipy
  - lifelines

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pymc arviz imbalanced-learn scipy lifelines openpyxl
```

## 各问题说明

### Q1: Y染色体浓度达标分析

**目标**: 建立贝叶斯层次模型，分析Y染色体浓度达标率与孕周（GA）的关系。

**主要功能**:
- 使用Beta-Binomial GLMM模型
- 包含随机截距（孕妇水平）
- 可选BMI协变量
- 输出预测曲线和参数后验分布

**运行方式**:
```bash
cd Q1
python Q1.py --input your_data.xlsx
```

### Q2: BMI聚类与NIPT最佳时点分析

**目标**: 基于BMI对孕妇进行聚类，确定不同BMI组别的最佳NIPT检测时点。

**主要功能**:
- BMI K-means聚类（5组固定聚类）
- Y染色体达标率时序分析
- 多维度综合评分（达标率、风险、可靠性、稳定性、一致性）
- 蒙特卡洛误差影响分析
- 生成9张可视化图表

**关键算法**:
- Wilson置信区间下界计算
- 孕周风险函数 R(t)
- 综合评分加权计算

**运行方式**:
```bash
cd Q2
python simplified_nipt_timing_analysis.py
```

### Q3: 多因素NIPT最佳时点分析

**目标**: 在Q2基础上，增加年龄、身高、体重等多因素进行PCA降维和聚类分析。

**主要功能**:
- PCA主成分分析降维
- 高斯混合模型聚类
- Cox比例风险模型生存分析
- 多因素误差敏感性分析
- 因素重要性评估

**运行方式**:
```bash
cd Q3
python complex_nipt_timing_analysis1.py
```

### Q4: 女胎数据异常检测

**目标**: 针对女胎数据，建立机器学习模型检测染色体非整倍体异常。

**主要功能**:
- 数据预处理（孕周解析、特征工程）
- 基线Z值规则模型
- 质量控制筛选
- 多种不平衡数据处理策略:
  - Logistic Regression + SMOTE
  - SVC + SMOTE
  - Gradient Boosting + SMOTE
  - Balanced Random Forest
  - Balanced Bagging
- 超参数随机搜索
- 概率校准
- 阈值优化（目标敏感度≥80%）
- 特征重要性分析
- 概率敏感性曲线生成

**运行方式**:
```bash
cd Q4
python test.py
```

## 数据格式

### 输入数据要求

各问题的输入数据应包含以下关键字段：

| 字段名 | 说明 | 问题 |
|--------|------|------|
| 孕妇代码 | 孕妇唯一标识 | Q1-Q4 |
| 检测孕周_数值 | 检测时孕周 | Q1-Q3 |
| 孕妇BMI | 孕妇BMI指数 | Q1-Q4 |
| Y染色体浓度 | 胎儿Y染色体浓度 | Q1-Q3 |
| 年龄 | 孕妇年龄 | Q3-Q4 |
| 身高 | 孕妇身高 | Q3 |
| 体重 | 孕妇体重 | Q3 |
| 染色体的非整倍体 | 异常标记（Q4） | Q4 |

## 输出结果

### Q1 输出
- `bbglmm_pred_curve.xlsx`: 孕周-Y浓度预测曲线
- `bbglmm_summary.xlsx`: 模型参数后验摘要
- `glmm_param_hdi_en.png`: 参数估计与97% HDI
- `glmm_prediction_curves_en.png`: 不同BMI组预测曲线
- `glmm_param_correlation_en.png`: 参数相关性热图
- `glmm_diagnostics_en.png`: 模型诊断图

### Q2 输出
- `5组BMI_NIPT最佳时点结果.xlsx`: 各组最佳检测时点
- `1_BMI分组分布.png`: BMI分组分布图
- `2_多维度评分雷达图.png`: 综合评分雷达图
- `3_各组最佳时点对比.png`: 各组最佳时点对比
- `4_策略选择分布.png`: 策略选择饼图
- `5_孕周-达标率趋势.png`: 孕周-达标率趋势
- `6_综合评分组件分析.png`: 评分组件分析
- `7_误差敏感性分析.png`: 误差敏感性
- `8_风险函数与最佳时点.png`: 风险函数曲线
- `9_临床建议热力图.png`: 临床建议热力图

### Q3 输出
- `多因素_NIPT最佳时点结果.xlsx`: 多因素分析结果
- `1_PCA降维可视化.png`: PCA降维散点图
- `2_因素重要性分析.png`: Cox模型因素重要性
- `3_聚类特征雷达图.png`: 聚类特征雷达图
- `4_各组最佳时点对比.png`: 各组最佳时点对比
- `5_多因素相关性热力图.png`: 因素相关性热图
- `6_误差敏感性分析.png`: 误差敏感性分析

### Q4 输出
- `female_anomaly_model.joblib`: 校准后的模型
- `female_anomaly_threshold.json`: 最优阈值
- `top10_coefficients.png`: Top10特征系数图
- `sensitivity_*.png`: 各特征概率敏感性曲线

## 作者信息

- 用途: CUMCM数学建模竞赛
- 日期: 2025

## 许可证

本项目仅供学习和竞赛使用。
