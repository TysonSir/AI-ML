import torch
import matplotlib.pyplot as plt

def drawResult(x, y, prediction, title="result"):
    # 出图
    plt.title(title)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

def createNet(x, y):
    # 建网络
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net

def save(net, pkl_path, params_pkl_path):
    torch.save(net, pkl_path)  # 保存整个网络
    torch.save(net.state_dict(), params_pkl_path)   # 只保存网络中的参数 (速度快, 占内存少)

def restore_net(pkl_path):
    # restore entire
    net = torch.load(pkl_path)
    return net

def restore_params(net, params_pkl_path):
    # 将保存的参数复制到 net
    net.load_state_dict(torch.load(params_pkl_path))

if __name__ == '__main__':
    # 训练数据
    torch.manual_seed(1)    # reproducible 随机种子
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    # 训练神经网络
    net = createNet(x, y)
    prediction = net(x)
    drawResult(x, y, prediction, 'src net')

    # 1 保存：网络对象 和 网络参数
    save(net, 'huigui.pkl', 'huigui_params.pkl')

    # 2.1 提取整个网络对象
    net1 = restore_net('huigui.pkl')
    prediction = net1(x) # 使用神经网络
    drawResult(x, y, prediction, 'load net')

    # 2.2 提取网络参数, 复制到新网络
    net_new = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    restore_params(net_new, 'huigui_params.pkl')
    prediction = net_new(x) # 使用神经网络
    drawResult(x, y, prediction, 'param load net')
