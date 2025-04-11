import pandas as pd
import numpy as np

# 生成1000个样本的模拟数据
timestamps = np.arange(0, 10000, 10)  # 时间戳（0~10000秒，间隔10秒）
vibration1 = np.random.randn(1000) * 0.1 + 0.5  # 振动信号1
vibration2 = np.random.randn(1000) * 0.1 + 0.6  # 振动信号2
temperature = np.linspace(25, 80, 1000)         # 温度
rpm = np.linspace(1500, 1800, 1000)             # 转速
lifespan = np.linspace(100, 0, 1000)            # 寿命从100%线性降至0%

# 创建数据框并保存为CSV文件
df = pd.DataFrame({
    "时间戳": timestamps,
    "振动信号1": vibration1,
    "振动信号2": vibration2,
    "温度": temperature,
    "转速": rpm,
    "寿命百分比": lifespan
})
df.to_csv("bearing_data.csv", index=False)  # 确保文件名正确
