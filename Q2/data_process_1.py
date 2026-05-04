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
df = pd.read_excel('data_processed.xlsx')
# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.figure(figsize=(10, 6))
sns.kdeplot(df['孕妇BMI'], fill=True, bw_adjust=0.8)  # bw_adjust调整平滑度
plt.axvline(df['孕妇BMI'].mean(), color='red', linestyle='--', label='均值')
plt.title('BMI数据核密度分布')
plt.xlabel('BMI(kg/m2)')
plt.ylabel('密度')
plt.legend()
plt.show()