import time, os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

work_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(work_dir)

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


"""1. 准备训练数据"""
# Mnist 手写数字————【1.训练数据】
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

"""2. 准备测试数据"""
# 我们除了训练数据, 还给一些【2.测试数据】, 测试看看它有没有训练好.
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)


# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]

def main():
    """3.搭建神经网络"""
    cnn = CNN()
    print(cnn)  # net architecture
    """
    CNN (
    (conv1): Sequential (
        (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU ()
        (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )
    (conv2): Sequential (
        (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU ()
        (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )
    (out): Linear (1568 -> 10)
    )
    """

    """4.训练数据"""
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        print('epoch: {}'.format(epoch))
        for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
            output = cnn(b_x)               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 12 == 0:
                print('step: [{}/{}]'.format(step, len(train_loader)))

    """
    ...
    Epoch:  0 | train loss: 0.0306 | test accuracy: 0.97
    Epoch:  0 | train loss: 0.0147 | test accuracy: 0.98
    Epoch:  0 | train loss: 0.0427 | test accuracy: 0.98
    Epoch:  0 | train loss: 0.0078 | test accuracy: 0.98
    """

    """5.测试训练结果"""
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

    """
    [7 2 1 0 4 1 4 9 5 9] prediction number
    [7 2 1 0 4 1 4 9 5 9] real number
    """

    """6.导出模型"""
    time.sleep(1)
    params_pkl_path = 'mnist.pkl'
    # torch.save(cnn, pkl_path)  # 保存整个网络
    torch.save(cnn.state_dict(), params_pkl_path)

def load_pkl(params_pkl_path = 'mnist.pkl'):
    """7.加载模型"""
    cnn = CNN()
    cnn.load_state_dict(torch.load(params_pkl_path))

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

if __name__ == '__main__':
    main()
    load_pkl()
