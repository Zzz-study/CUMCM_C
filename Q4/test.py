

# Now import all the libraries
import pandas as pd
import numpy as np
import imblearn
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import brier_score_loss
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, balanced_accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Set style for better visualizations
sns.set_palette("Set2")

print("All packages imported successfully!")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score, balanced_accuracy_score,
    precision_recall_curve, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

sns.set_palette("Set2")


def load_and_preprocess_data(file_path, sheet_name):
    import re  # 用于解析孕周
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    # 内嵌：把"孕周"文本统一解析成以周为单位的数值（天按 1/7 周计）
    def _parse_gestation_to_weeks(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        s = (s.replace('＋', '+').replace('﹢', '+')
             .replace('，', ',').replace('周', 'w')
             .replace('天', 'd').replace('wk', 'w'))
        # 形式：12+3
        if '+' in s:
            try:
                w, d = s.split('+', 1)
                w_nums = re.findall(r'\d+\.?\d*', w)
                d_nums = re.findall(r'\d+\.?\d*', d)
                w = float(w_nums[0]) if w_nums else 0.0
                d = float(d_nums[0]) if d_nums else 0.0
                return w + d / 7.0
            except Exception:
                pass
        # 形式：12w3d / 12w / 3d
        w_match = re.search(r'(\d+\.?\d*)\s*w', s)
        d_match = re.search(r'(\d+\.?\d*)\s*d', s)
        if w_match or d_match:
            w = float(w_match.group(1)) if w_match else 0.0
            d = float(d_match.group(1)) if d_match else 0.0
            return w + d / 7.0
        # 纯数字（按周）
        nums = re.findall(r'\d+\.?\d*', s)
        if nums:
            return float(nums[0])
        return np.nan

    # 生成标准化列：gestational weeks(week)
    gest_cols = [c for c in df.columns if any(k in str(c) for k in ['孕周', '孕期', '周数'])]
    if len(gest_cols) > 0:
        src = gest_cols[0]
        df['gestational weeks(week)'] = df[src].apply(_parse_gestation_to_weeks)

    base_features = [
        '年龄', '身高', '体重', '孕妇BMI', '原始读段数', '唯一比对的读段数',
        'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
        'X染色体的Z值', 'X染色体浓度',
        '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
        'gestational weeks(week)'
    ]
    available_features = [f for f in base_features if f in df.columns]
    for f in base_features:
        if f not in df.columns:
            print(f"警告: 特征 '{f}' 不存在")

    X = df[available_features]
    X = X.fillna(X.median(numeric_only=True))  # 中位数插补

    y = df['染色体的非整倍体'].notna()
    if y.sum() > 0 and '染色体的非整倍体' in df.columns:
        print("非整倍体类型分布:")
        print(df['染色体的非整倍体'].value_counts())

    return X, y, df


# 2. 基线规则模型
def baseline_model_z_value(X, z_threshold=3.0):
    y_pred = np.zeros(len(X), dtype=int)
    z13 = (X['13号染色体的Z值'].abs() > z_threshold) if '13号染色体的Z值' in X else False
    z18 = (X['18号染色体的Z值'].abs() > z_threshold) if '18号染色体的Z值' in X else False
    z21 = (X['21号染色体的Z值'].abs() > z_threshold) if '21号染色体的Z值' in X else False
    zx = (X['X染色体的Z值'].abs() > z_threshold) if 'X染色体的Z值' in X else False
    zy = (X['Y染色体的Z值'].abs() > z_threshold) if 'Y染色体的Z值' in X else False
    mask = z13 | z18 | z21 | zx | zy
    y_pred[mask] = 1
    return y_pred


def evaluate_baseline_model(X, y):
    y_pred = baseline_model_z_value(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


# 3. 质量控制（可按需调整阈值）
def quality_control(X, threshold_dict=None):
    if threshold_dict is None:
        threshold_dict = {
            '原始读段数': 1_000_000,
            'GC含量': (0.35, 0.45),
            '在参考基因组上比对的比例': 0.75,
        }
    low_quality = np.zeros(len(X), dtype=bool)
    if '原始读段数' in X and '原始读段数' in threshold_dict:
        low_quality |= (X['原始读段数'] < threshold_dict['原始读段数'])
    if 'GC含量' in X and 'GC含量' in threshold_dict:
        lo, hi = threshold_dict['GC含量']
        low_quality |= (X['GC含量'] < lo) | (X['GC含量'] > hi)
    if '在参考基因组上比对的比例' in X and '在参考基因组上比对的比例' in threshold_dict:
        low_quality |= (X['在参考基因组上比对的比例'] < threshold_dict['在参考基因组上比对的比例'])
    return low_quality


# 4. 阈值优化（以敏感度为目标）
def optimize_threshold_for_sensitivity(model, X_test, y_test, target_sensitivity=0.8):
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 0.5, 1000)
    best = None
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-7)
        rec = {
            'threshold': thr,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        if best is None or abs(rec['sensitivity'] - target_sensitivity) < abs(best['sensitivity'] - target_sensitivity):
            best = rec
    return pd.Series(best)


# 5. 训练：加入 SMOTE 与不平衡集成 - 修改为按孕妇分组划分
def train_and_evaluate_models_imbalance(X, y, df, test_size=0.2, random_state=42):
    # 使用GroupShuffleSplit按孕妇代码分组划分
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # 获取孕妇代码列
    woman_codes = df['孕妇代码']

    # 划分训练集和测试集
    train_idx, test_idx = next(gss.split(X, y, groups=woman_codes))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("训练集孕妇数量:", len(np.unique(woman_codes.iloc[train_idx])))
    print("测试集孕妇数量:", len(np.unique(woman_codes.iloc[test_idx])))
    print("训练集类别分布:", y_train.value_counts())
    print("测试集类别分布:", y_test.value_counts())

    models = {
        # 线性/核方法：加Scaler + SMOTE
        'LogisticRegression + SMOTE': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('clf', LogisticRegression(random_state=random_state, max_iter=5000))
        ]),
        'SVC + SMOTE': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('clf', SVC(probability=True, random_state=random_state))
        ]),
        # GBDT对尺度不敏感，但在极不平衡时也可配合SMOTE
        'GradientBoosting + SMOTE': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', SMOTE(random_state=random_state)),
            ('clf', GradientBoostingClassifier(random_state=random_state))
        ]),
        # 专门的不平衡集成
        'BalancedRandomForest': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', BalancedRandomForestClassifier(
                n_estimators=800, random_state=random_state))
        ]),
        'LogisticRegression (class_weight=balanced)': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=5000, random_state=random_state))
        ]),
        'BalancedBagging(Depth3-Tree)': ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=6, random_state=random_state),
                n_estimators=200,
                random_state=random_state,
                bootstrap=False, replacement=False
            ))
        ]),

    }

    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # 少数模型可能无 predict_proba，这里兜底成0/1概率
            y_proba = y_pred.astype(float)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        res = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'model': model,
        }
        print(
            f"  F1: {res['f1']:.3f} | Sens: {res['recall']:.3f} | Spec: {tn / (tn + fp):.3f} | AUC: {res['auc']:.3f} | PR-AUC: {res['pr_auc']:.3f}")
        results[name] = res

    return results, X_test, y_test


def main():
    print("=== 只处理女胎数据 ===")
    X, y, df = load_and_preprocess_data('女胎孕妇.xlsx', '女胎检测数据')

    print(f"\n女胎数据形状: {X.shape} | 正类比例: {y.mean():.2%}")
    print(f"孕妇数量: {len(df['孕妇代码'].unique())}")
    print(f"每个孕妇平均检测次数: {len(df) / len(df['孕妇代码'].unique()):.2f}")

    print("\n=== 基线模型（Z值规则） ===")
    baseline = evaluate_baseline_model(X, y)
    print(
        f"准确率: {baseline['accuracy']:.3f}  精确率: {baseline['precision']:.3f}  召回率: {baseline['recall']:.3f}  F1: {baseline['f1']:.3f}  "
        f"敏感度: {baseline['sensitivity']:.3f}  特异度: {baseline['specificity']:.3f}")
    print(f"TP: {baseline['tp']}, FP: {baseline['fp']}, FN: {baseline['fn']}, TN: {baseline['tn']}")

    low_q = quality_control(X)
    print(f"\n低质量样本数量: {low_q.sum()} ({low_q.mean():.2%})")
    if low_q.sum() > 0:
        print("移除低质量样本...")
        # 注意：需要同时更新X, y和df
        X = X[~low_q]
        y = y[~low_q]
        df = df[~low_q]
        print(f"移除后数据形状: {X.shape} | 正类比例: {y.mean():.2%}")

    print("\n=== 训练含重采样/不平衡集成的模型 ===")
    # 修改：传入df参数以便获取孕妇代码
    results, X_test, y_test = train_and_evaluate_models_imbalance(X, y, df, test_size=0.2, random_state=42)

    # 选最佳模型（以F1为主，也可换成 PR-AUC）
    best_name, best = max(results.items(), key=lambda kv: kv[1]['f1'])
    best_model = best['model']
    print(f"\n🎯 最佳模型: {best_name} | F1: {best['f1']:.3f} | AUC: {best['auc']:.3f} | PR-AUC: {best['pr_auc']:.3f}")

    if hasattr(best_model, 'predict_proba'):
        opt = optimize_threshold_for_sensitivity(best_model, X_test, y_test, target_sensitivity=0.80)
        print(f"\n📊 阈值优化(目标敏感度=0.80):")
        print(
            f"  最佳阈值: {opt['threshold']:.3f} | 敏感度: {opt['sensitivity']:.3f} | 精确率: {opt['precision']:.3f} | F1: {opt['f1']:.3f}")
        print(f"  TP: {int(opt['tp'])}, FP: {int(opt['fp'])}, FN: {int(opt['fn'])}, TN: {int(opt['tn'])}")

    # 如需特征重要性（仅树模型）
    if 'BalancedRandomForest' in best_name:
        clf = best_model.named_steps['clf']
        importances = clf.feature_importances_
        cols = X.columns
        order = np.argsort(importances)[::-1]
        topk = min(10, len(cols))
        print("\n🔍 特征重要性 (Top 10):")
        for i in range(topk):
            print(f"  {i + 1}. {cols[order[i]]}: {importances[order[i]]:.4f}")


if __name__ == "__main__":
    main()

# 1) 超参随机搜索：仅在训练集上调参（主指标：PR-AUC）
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split, GroupShuffleSplit
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def tune_logreg_with_smote_on_train(X_train, y_train, groups, random_state=42, n_iter=30):
    pipe = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state)),
        ('clf', LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    param_dist = {
        'smote__k_neighbors': [3, 5, 7],
        'clf__C': loguniform(1e-3, 1e2),
        'clf__solver': ['lbfgs', 'saga'],
        'clf__class_weight': [None, 'balanced'],
    }

    # 使用GroupKFold进行交叉验证，确保同一孕妇不会同时出现在训练和验证折中
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=n_iter,
        scoring='average_precision', cv=gkf, n_jobs=-1, random_state=random_state, verbose=1
    )
    search.fit(X_train, y_train, groups=groups)
    print("最佳PR-AUC(交叉验证):", search.best_score_)
    print("最佳参数:", search.best_params_)
    return search.best_estimator_


def threshold_at_min_recall(y_true, y_proba, min_recall=0.80):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    mask = recalls[:-1] >= min_recall
    if mask.any():
        idx_in_mask = precisions[:-1][mask].argmax()
        thr = thresholds[mask][idx_in_mask]
    else:
        idx = np.abs(recalls[:-1] - min_recall).argmin()
        thr = thresholds[idx]
    return float(thr)


# 2) 训练/校准/阈值：在召回≥0.80前提下最大化精确率
# 说明：重新分一次train/test，用于调参与最终评估的一致性
# 若 X,y,df 未定义则自动加载并做质控
try:
    X, y, df
except NameError:
    X, y, df = load_and_preprocess_data('女胎孕妇.xlsx', '女胎检测数据')
    low_q = quality_control(X)
    if low_q.sum() > 0:
        X = X[~low_q]
        y = y[~low_q]
        df = df[~low_q]
    print(f"数据已加载: {X.shape}, 正类比例: {y.mean():.2%}")

random_state = 42
woman_codes = df['孕妇代码']

# 使用GroupShuffleSplit按孕妇代码分组划分
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
train_idx, test_idx = next(gss.split(X, y, groups=woman_codes))
X_train_t, X_test_t = X.iloc[train_idx], X.iloc[test_idx]
y_train_t, y_test_t = y.iloc[train_idx], y.iloc[test_idx]
groups_train = woman_codes.iloc[train_idx]

# 调参得到最优 LR+SMOTE 管道（只在训练集）
best_lr_smote = tune_logreg_with_smote_on_train(X_train_t, y_train_t, groups_train, random_state=random_state,
                                                n_iter=30)

# 概率校准（更稳的概率，用于阈值移动）；小样本建议 'sigmoid'
cal = CalibratedClassifierCV(best_lr_smote, cv=5, method='sigmoid')
cal.fit(X_train_t, y_train_t)
y_proba_cal = cal.predict_proba(X_test_t)[:, 1]

# 在召回≥目标下，选择让精确率最大的阈值
target_recall = 0.80
thr = threshold_at_min_recall(y_test_t, y_proba_cal, min_recall=target_recall)
y_pred_cal = (y_proba_cal >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test_t, y_pred_cal).ravel()
print("\n[LogReg+SMOTE 调参+校准+受约束阈值]")
print(f"阈值: {thr:.3f}")
print(f"召回: {tp / (tp + fn):.3f} | 精确率: {tp / (tp + fp):.3f} | F1: {f1_score(y_test_t, y_pred_cal):.3f}")
print(
    f"Spec: {tn / (tn + fp):.3f} | PR-AUC: {average_precision_score(y_test_t, y_proba_cal):.3f} | AUC: {roc_auc_score(y_test_t, y_proba_cal):.3f}")



# 可视化：系数条形图 + 一维敏感性曲线（PDP）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import json
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = True


# 1) Load calibrated model and threshold if not in memory
try:
    cal, thr
except NameError:
    cal = load('female_anomaly_model.joblib')
    with open('female_anomaly_threshold.json', 'r', encoding='utf-8') as f:
        thr = json.load(f)['threshold']

# 2) Load X if missing (for reference median row)
try:
    X
except NameError:
    X, y, df = load_and_preprocess_data('女胎孕妇.xlsx', '女胎检测数据')
    low_q = quality_control(X)
    if low_q.sum() > 0:
        X = X[~low_q]; y = y[~low_q]
    print(f"Data loaded: {X.shape}, Positive rate: {y.mean():.2%}")

# Mapping for display (model still uses original column names)
display_map = {
    '年龄': 'Age',
    '身高': 'Height (cm)',
    '体重': 'Weight (kg)',
    '孕妇BMI': 'Maternal BMI',
    '原始读段数': 'Raw reads',
    '唯一比对的读段数': 'Uniquely mapped reads',
    'GC含量': 'GC content',
    '13号染色体的Z值': 'Chr13 Z-score',
    '18号染色体的Z值': 'Chr18 Z-score',
    '21号染色体的Z值': 'Chr21 Z-score',
    'X染色体的Z值': 'ChrX Z-score',
    'X染色体浓度': 'ChrX fraction',
    '在参考基因组上比对的比例': 'Alignment ratio',
    '重复读段的比例': 'Duplicate ratio',
    '被过滤掉读段数的比例': 'Filtered reads ratio',
    '唯一比对/原始读段数': 'Unique/Raw reads ratio',
}


def disp(name: str) -> str:
    return display_map.get(name, name)

# 3) Extract per-fold fitted base estimators and average coefficients
calibrators = getattr(cal, 'calibrated_classifiers_', None)
if calibrators is None:
    raise RuntimeError("cal has no calibrated_classifiers_. Ensure it's a fitted CalibratedClassifierCV.")
coefs = []
names = None
for cc in calibrators:
    est = cc.estimator  # fitted ImbPipeline per CV fold
    if names is None:
        if hasattr(est.named_steps['imputer'], 'feature_names_in_'):
            names = list(est.named_steps['imputer'].feature_names_in_)
        else:
            names = list(X.columns)
    lr = est.named_steps['clf']
    coefs.append(lr.coef_.ravel())

coefs = np.vstack(coefs)
coef_mean = coefs.mean(axis=0)
coef_df = pd.DataFrame({'feature': names, 'coef': coef_mean})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)

# 4) Report BMI coefficient and Top-10
bmi_row = coef_df.loc[coef_df['feature'] == '孕妇BMI']
bmi_coef = float(bmi_row['coef'].iloc[0]) if not bmi_row.empty else np.nan
print(f"Maternal BMI average coefficient: {bmi_coef:.6f} "
      f"(>0 increases risk; <0 decreases; coefficients are in standardized space)")

print("\nTop-10 coefficients by absolute value:")
tmp = coef_df.head(10).copy()
tmp['feature'] = tmp['feature'].map(disp)
print(tmp[['feature','coef']].to_string(index=False))

# 5) Plot Top-10 coefficient bar chart (English)
top10 = coef_df.head(10).iloc[::-1].copy()
top10['feature_disp'] = top10['feature'].map(disp)

plt.figure(figsize=(7,5))
plt.barh(top10['feature_disp'], top10['coef'])
plt.axvline(0, color='k', linewidth=1)
plt.title('Logistic Regression Coefficients (Top-10, standardized features)')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.savefig('top10_coefficients.png', bbox_inches='tight', dpi=300)
plt.close()  # 关闭图表，避免内存占用
print("✅ Top10系数图已保存为: top10_coefficients.png")

# 6) Probability sensitivity curves for key features
expected_cols = names  # model input order

def build_reference_row(X_df: pd.DataFrame) -> pd.DataFrame:
    ref = X_df.median(numeric_only=True).to_dict()
    for c in expected_cols:
        if c not in ref:
            ref[c] = np.nan
    ref_df = pd.DataFrame([ref])[expected_cols].copy()
    if ('原始读段数' in ref_df.columns) and ('唯一比对的读段数' in ref_df.columns):
        if '唯一比对/原始读段数' in expected_cols:
            with np.errstate(divide='ignore', invalid='ignore'):
                ref_df['唯一比对/原始读段数'] = np.where(
                    (ref_df['原始读段数'] > 0) & pd.notna(ref_df['原始读段数']) & pd.notna(ref_df['唯一比对的读段数']),
                    ref_df['唯一比对的读段数'] / ref_df['原始读段数'],
                    np.nan
                )
    return ref_df[expected_cols]

ref_row = build_reference_row(X)

def prob_curve_for_feature(feature: str, grid: np.ndarray):
    X_grid = pd.concat([ref_row]*len(grid), ignore_index=True)
    if feature not in X_grid.columns:
        raise ValueError(f"Feature not in training columns: {feature}")
    X_grid.loc[:, feature] = grid
    if '唯一比对/原始读段数' in X_grid.columns and {'原始读段数','唯一比对的读段数'}.issubset(X_grid.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            X_grid['唯一比对/原始读段数'] = np.where(
                (X_grid['原始读段数'] > 0) & pd.notna(X_grid['原始读段数']) & pd.notna(X_grid['唯一比对的读段数']),
                X_grid['唯一比对的读段数'] / X_grid['原始读段数'],
                np.nan
            )
    probs = cal.predict_proba(X_grid[expected_cols])[:, 1]
    return probs

def plot_curve(name_cn, grid):
    probs = prob_curve_for_feature(name_cn, grid)
    plt.figure(figsize=(6,4))
    plt.plot(grid, probs, lw=2,color='b')
    plt.axhline(thr, color='r', ls='--', label=f'Threshold={thr:.3f}')
    plt.xlabel(disp(name_cn)); plt.ylabel('Predicted probability')
    plt.title(f'Probability sensitivity: {disp(name_cn)}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()
    filename = f'sensitivity_{name_cn.replace("号染色体的Z值", "").replace("染色体浓度", "").replace(" ", "_")}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()  # 关闭图表
    print(f"✅ 敏感性曲线已保存为: {filename}")

grids = {
    '21号染色体的Z值': np.linspace(-4, 5, 200),
    '18号染色体的Z值': np.linspace(-4, 5, 200),
    '13号染色体的Z值': np.linspace(-4, 5, 200),
    'X染色体浓度': np.linspace(0.0, max(0.3, np.nanpercentile(X.get('X染色体浓度', pd.Series([0.2])), 99)), 200),
    '在参考基因组上比对的比例': np.linspace(0.60, 0.95, 200),
    'GC含量': np.linspace(0.30, 0.50, 200),
    '孕妇BMI': np.linspace(16, 35, 200),
}

for feat_cn, grid in grids.items():
    if feat_cn in expected_cols:
        plot_curve(feat_cn, grid)