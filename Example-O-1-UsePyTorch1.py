# some simple operations of tensor and use GPU to accelerate the training

# install pytorch with cuda
# https://pytorch.org/get-started/locally
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

import torch

# create tensor with random values
x = torch.rand(5, 3)
print(x)

# create tensor with zeros
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# create tensor with ones
x = torch.ones(5, 3, dtype=torch.long)
print(x)

# create tensor with a specific value
x = torch.tensor([5.5, 3])
print(x)

# create tensor with a existing tensor and shape
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x.size())

# some comments on the common operations of tensor
# 1. tensor(sizes) 创建一个指定形状的张量。
# 2. tensor(data) 使用已有的数据创建张量。
# 3. ones(sizes) 创建一个指定形状的全1张量。
# 4. zeros(sizes) 创建一个指定形状的全0张量。
# 5. eye(sizes) 创建一个指定形状的单位矩阵。
# 6. arange(start, end, step) 创建一个指定范围和步长的等差数列。
# 7. linspace(start, end, steps) 创建一个指定范围和步长的等差数列。
# 8. randn(sizes) 创建一个指定形状的正态分布随机张量。
# 9. rand(sizes) 创建一个指定形状的均匀分布随机张量。
# 10. normal(mean, std, sizes) 创建一个指定均值和标准差的正态分布随机张量。
# 11. randperm(n) 创建一个指定长度的随机整数张量。

# tensor add
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
print(y.add(x))# 返回一个新的张量，不改变y
print(y)
print(y.add_(x))# 返回一个新的张量，改变y
print(y)

# tensor index
x = torch.rand(5, 3)
print(x[:, 1])

# share memory
y = x[0, :]
print(x[0, :])
y += 1
print(x[0, :])

# transform tensor
x = torch.rand(5, 3)
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())

# data type
x = torch.randn(1)
print(type(x))
print(type(x.item()))

# check the GPU
cudaMsgs = torch.cuda.is_available()
gpuCount = torch.cuda.device_count()
print("1.是否存在GPU：{}".format(cudaMsgs), "2.如果存在有：{}个".format(gpuCount))

# let tensor move between CPU and GPU
test_tensor = torch.rand(100, 100)
test_tensor_gpu = test_tensor.cuda() # 将张量移动到GPU or test_tensor_gpu = test_tensor.to("cuda")
test_tensor_cpu = test_tensor_gpu.cpu() # 将张量移动到CPU or test_tensor_cpu = test_tensor_gpu.to("cpu")
print("test_tensor的设备：{}".format(test_tensor.device), "\ntest_tensor_gpu的设备：{}".format(test_tensor_gpu.device), "\ntest_tensor_cpu的设备：{}".format(test_tensor_cpu.device))

# use GPU to accelerate the training
from torch import nn
model = nn.Linear(10, 1)
print("The device is gpu begin?:",next(model.parameters()).is_cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("The device is gpu later?:",next(model.parameters()).is_cuda)
print("The device is gpu,",next(model.parameters()).device)

# create a model with multi-GPUs
model = nn.Linear(10, 1)
print("The device is gpu begin?:",next(model.parameters()).is_cuda)
model = nn.DataParallel(model)# 将模型分配到多个GPU
print("The device is gpu begin?:",next(model.module.parameters()).device)


