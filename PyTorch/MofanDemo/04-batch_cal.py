import torch
import torch.utils.data as Data

import matplotlib.pyplot as plt

def drawResult(x, y, prediction, title="result"):
    # 出图
    plt.title(title)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

BATCH_SIZE = 25      # 批训练的数据个数

torch.manual_seed(1)    # reproducible 随机种子
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据(报错！原因是：cuda 虚拟环境的共享内存不足。)
)


# 建网络
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()


for epoch in range(30):   # 训练所有!整套!数据 30 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打出来进度数据
        print('Epoch: ', epoch, '| Step: ', step)
        # print('| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

prediction = net(x) # 使用神经网络
drawResult(x, y, prediction, 'net')

"""
Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
"""