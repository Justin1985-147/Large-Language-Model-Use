# PyTorch Project accelerated by GPU
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # 添加这行代码解决OpenMP运行时冲突问题

"""
项目过程：
1、 数据准备：均匀分布构建样本特征数据X,定义线性回归函数生成标签Y.
2、 模型定义: 采用简单的全连接网络模型
3、 建模、优化函数设置、损失函数设置、训练模型函数：将模型和数据移到GPU上，设置优化方法和损失函数开始模拟特征data和label之间的规律 练学习过程。
4、测试评估、
我们首先生成了一些随机的输入数据 
"""

import torch
from torch import nn

# 1. 数据准备
# 均匀分布构建样本特征数据X,定义线性回归函数生成标签Y.
sample_num = 1000000 # 样本数量
sample_test = 1000 # 测试样本数量
X = 10 * torch.rand(sample_num, 2) - 5.0 # 均匀分布构建样本特征数据X
X_test = 10 * torch.rand(sample_test, 2) - 5.0 # 均匀分布构建测试样本特征数据X
w0 = torch.tensor([[2.0, -3.0]]) # 定义线性回归函数的系数
b0 = torch.tensor([[10.0]]) # 定义线性回归函数的偏置
Y = X @ w0.t() + b0 + torch.normal(0, 2, size=(sample_num, 1)) # 定义线性回归函数的标签Y
Y_test = X_test @ w0.t() + b0 + torch.normal(0, 2, size=(sample_test, 1)) # 定义线性回归函数的测试标签Y
# X@w0.t() 表示矩阵乘法，X的每一行与w0的每一行进行点积运算，得到一个结果，然后将这些结果组合成一个向量。
# w0.t() 表示矩阵的转置，将w0的行向量转换为列向量。
print("torch.cuda.is_available():", torch.cuda.is_available())
print("X.shape:", X.shape, "Y.shape:", Y.shape, "X_test.shape:", X_test.shape, "Y_test.shape:", Y_test.shape)

# 将数据移到GPU上
data = X.cuda()
label = Y.cuda()
# 检查数据是否在GPU上
print("data is on GPU:", data.is_cuda)
print("label is on GPU:", label.is_cuda)

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand_like(w0))
        self.b = nn.Parameter(torch.rand_like(b0))        

    def forward(self, x):
        return x @ self.w.t() + self.b

# 训练过程
epochs = 2000  # 训练轮数
losses = [] # 损失列表

# 定义训练函数
def train():
    import time
    tic = time.time()
    # 建模
    linear = LinearRegression()
    # 将模型移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)
    # 定义损失函数
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pre = linear(data)
        loss = loss_fn(Y_pre, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    # 保存模型时只保存参数
    torch.save(linear.state_dict(), "./linear_parameters.pth")
    toc = time.time()
    print(f"训练时间: {toc - tic:.4f}秒")

train()

# 测试模型效果
data_test = X_test.cuda()
label_test = Y_test.cuda()

def test():
    import time
    tic = time.time()
    loss_fn = nn.MSELoss()
    # 创建新模型实例并加载参数
    linear_test = LinearRegression()
    linear_test.load_state_dict(torch.load("./linear_parameters.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear_test.to(device)
    linear_test.eval()# 设置为评估模式
    Y_pre = linear_test(data_test)
    loss_test = loss_fn(Y_pre, label_test)
    toc = time.time()
    print(f"测试时间: {toc - tic:.4f}秒", f"测试损失: {loss_test.item():.4f}")
    return Y_pre

test()

# 可视化损失函数
import matplotlib.pyplot as plt
plt.plot(range(epochs), losses, label="Training loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.show()

# 可视化模型预测结果
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 创建3D图形对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 获取数据
Y_pre = test().cpu().detach().numpy()
X_test_np = data_test.cpu().detach().numpy()
Y_test_np = label_test.cpu().detach().numpy()

# 绘制真实数据点
ax.scatter(X_test_np[:, 0], X_test_np[:, 1], Y_test_np, 
          c='blue', marker='o', label='True data')

# 绘制预测数据点
ax.scatter(X_test_np[:, 0], X_test_np[:, 1], Y_pre, 
          c='red', marker='^', label='Predicted data')

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 添加标题和图例
plt.title('Linear Regression Prediction Result Comparison')
plt.legend()
plt.show()