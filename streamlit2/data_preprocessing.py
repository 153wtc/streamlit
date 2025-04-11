import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 1. 加载数据
df = pd.read_csv("bearing_data.csv")
X = df[["振动信号1", "振动信号2", "温度", "转速"]].values  # 输入特征（4个维度）
y = df["寿命百分比"].values.reshape(-1, 1)           # 标签

# 2. 数据标准化
# 输入特征标准化
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
# 标签标准化
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# 3. 构建时间序列数据（每个样本为连续的10个时间步）
def create_sequences(data, seq_length=10):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X_sequences = create_sequences(X_scaled, seq_length=10)  # 形状 [样本数, 时间步长=10, 特征数=4]
y_sequences = y_scaled[9:]  # 对齐标签（每个样本的标签对应最后一个时间步）

# 4. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, shuffle=False
)

# 5. 保存处理后的数据
dump(X_train, "train_set.joblib")
dump(y_train, "train_label.joblib")
dump(X_test, "test_set.joblib")
dump(y_test, "test_label.joblib")
dump(y_scaler, "y_scaler.joblib")  # 保存标签的标准化器

print("数据预处理完成！生成以下文件：")
print("train_set.joblib, train_label.joblib, test_set.joblib, test_label.joblib, y_scaler.joblib")