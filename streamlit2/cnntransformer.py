import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import matplotlib.pyplot as plt

# 固定随机种子保证可重复性
torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader(batch_size, workers=2):
    """加载预处理好的数据集"""
    try:
        train_set = load('./train_set.joblib')
        train_label = load('./train_label.joblib')
        test_set = load('./test_set.joblib')
        test_label = load('./test_label.joblib')
    except FileNotFoundError as e:
        print(f"❌ 数据文件缺失: {str(e)}")
        raise

    # 转换为Tensor格式
    train_set = torch.tensor(train_set, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)
    test_set = torch.tensor(test_set, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_set, train_label),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_set, test_label),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True
    )
    return train_loader, test_loader

class CNNTransformer(nn.Module):
    def __init__(self, input_dim, conv_archs, hidden_dim, num_layers, num_heads, output_dim, dropout_rate=0.55):
        super().__init__()
        # CNN参数
        self.conv_arch = conv_archs
        self.input_channels = input_dim
        self.cnn_features = self.make_layers()

        # Transformer编码器
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=conv_archs[-1][-1],
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_rate,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 池化和输出层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(conv_archs[-1][-1], output_dim)

    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.extend([
                    nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ])
                self.input_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        x = self.cnn_features(x)
        x = x.permute(0, 2, 1)  # [B, D, L] -> [B, L, D]
        
        x = self.transformer(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.reshape(x.size(0), -1)
        return self.linear(x)

# 模型参数
input_dim = 4
conv_archs = ((1, 32),)
output_dim = 1
hidden_dim = 64
num_layers = 2
num_heads = 4

model = CNNTransformer(input_dim, conv_archs, hidden_dim, num_layers, num_heads, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

def model_train(epochs, model, optimizer, loss_function, train_loader, device):
    """增强的训练函数"""
    model = model.to(device)
    best_loss = float('inf')
    
    train_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seq)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model_cnn_transformer.pt')
            print(f"💾 保存最佳模型，当前loss: {best_loss:.4f}")
        
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}')
    
    print(f'\n训练完成，耗时: {time.time()-start_time:.1f}秒')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == '__main__':
    # 数据加载
    train_loader, test_loader = dataloader(batch_size=16)
    
    # 开始训练（取消注释下一行）
    model_train(
        epochs=50,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
        device=device
    )
    
    # 测试代码
    model.load_state_dict(torch.load('best_model_cnn_transformer.pt'))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(label.cpu().numpy())
    
    # 反标准化
    scaler = load('./y_scaler.joblib')
    test_preds = scaler.inverse_transform(np.array(test_preds))
    test_labels = scaler.inverse_transform(np.array(test_labels))
    
    # 计算指标
    print(f"\n{'指标':<15}{'值':>10}")
    print(f"{'R² Score':<15}{r2_score(test_labels, test_preds):>10.4f}")
    print(f"{'MSE':<15}{mean_squared_error(test_labels, test_preds):>10.4f}")
    print(f"{'RMSE':<15}{np.sqrt(mean_squared_error(test_labels, test_preds)):>10.4f}")
    print(f"{'MAE':<15}{mean_absolute_error(test_labels, test_preds):>10.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(test_labels, label='True Values', alpha=0.7)
    plt.plot(test_preds, label='Predictions', linestyle='--', alpha=0.7)
    plt.title('Prediction Results Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Remaining Life (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()