import streamlit as st
import torch
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from cnntransformer import CNNTransformer, dataloader

# --------------------------
# 页面基础配置
# --------------------------
st.set_page_config(layout="wide")
st.title("材料寿命预测系统")

# --------------------------
# 模型参数 (必须与训练时完全一致)
# --------------------------
MODEL_CONFIG = {
    "input_dim": 4,
    "conv_archs": ((1, 32),),
    "hidden_dim": 64,
    "num_layers": 2,
    "num_heads": 4,
    "output_dim": 1
}

# --------------------------
# 侧边栏参数设置
# --------------------------
with st.sidebar:
    st.header("⚙️ 模型参数")
    batch_size = st.slider("批量大小 (Batch Size)", 1, 32, 16)
    num_heads = st.slider("注意力头数 (Num Heads)", 1, 8, 4)
    num_layers = st.slider("Transformer层数 (Num Layers)", 1, 4, 2)
    epochs = st.slider("训练轮数 (Epochs)", 10, 100, 50)

# --------------------------
# 加载模型和数据 (关键修复部分)
# --------------------------
@st.cache_resource
def load_model_and_data():
    try:
        # 初始化模型结构
        model = CNNTransformer(**MODEL_CONFIG)
        
        # 加载模型参数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load('best_model_cnn_transformer.pt', map_location=device)
        )
        model.eval()  # ✅ 现在可以安全调用eval()
        model.to(device)

        # 加载测试数据
        _, test_loader = dataloader(batch_size=16)
        
        return model, test_loader, device
        
    except FileNotFoundError as e:
        st.error(f"❌ 关键文件缺失: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"🚨 初始化失败: {str(e)}")
        st.stop()

# --------------------------
# 主界面操作
# --------------------------
model, test_loader, device = load_model_and_data()

if st.button("🚀 运行预测"):
    test_origin_data = []
    test_pre_data = []
    
    with torch.no_grad():
        for data, label in test_loader:
            # 数据格式转换
            data = data.to(device)
            label = label.to(device)
            
            # 执行预测
            pred = model(data)
            
            # 收集数据
            test_origin_data.extend(label.cpu().numpy().flatten())
            test_pre_data.extend(pred.cpu().numpy().flatten())
    
    try:
        # 反标准化
        scaler = load('./y_scaler.joblib')
        test_origin_data = scaler.inverse_transform(np.array(test_origin_data).reshape(-1, 1))
        test_pre_data = scaler.inverse_transform(np.array(test_pre_data).reshape(-1, 1))
    except Exception as e:
        st.error(f"🔧 数据后处理失败: {str(e)}")
        st.stop()
    
    # 计算指标
    try:
        metrics = {
            "R²": r2_score(test_origin_data, test_pre_data),
            "MSE": mean_squared_error(test_origin_data, test_pre_data),
            "RMSE": np.sqrt(mean_squared_error(test_origin_data, test_pre_data)),
            "MAE": mean_absolute_error(test_origin_data, test_pre_data)
        }
    except ValueError as e:
        st.error(f"📊 指标计算错误: {str(e)}")
        st.stop()
    
    # 显示结果
    st.subheader("📈 预测结果")
    cols = st.columns(4)
    metric_info = [
        ("R²分数", "R²", "越接近1表示预测越准确"),
        ("均方误差 (MSE)", "MSE", "数值越小越好"),
        ("均方根误差 (RMSE)", "RMSE", "与目标变量同单位"),
        ("平均绝对误差 (MAE)", "MAE", "绝对误差平均值")
    ]
    
    for col, (title, key, desc) in zip(cols, metric_info):
        with col:
            st.metric(label=title, value=f"{metrics[key]:.4f}", help=desc)
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_origin_data, 
            label="真实寿命",
            color="#2c3e50",
            linewidth=1.5,
            alpha=0.8)
    ax.plot(test_pre_data,
            label="预测值",
            color="#e74c3c",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8)
    
    ax.set_xlabel("运行周期/10s", fontsize=12)
    ax.set_ylabel("寿命百分比", fontsize=12)
    ax.set_title("CNN-Transformer 预测结果对比", 
                fontsize=16, 
                pad=20,
                fontweight="bold")
    ax.grid(True, 
           color="#bdc3c7",
           linestyle="--", 
           linewidth=0.5,
           alpha=0.4)
    ax.legend(loc="upper right")
    
    st.pyplot(fig)