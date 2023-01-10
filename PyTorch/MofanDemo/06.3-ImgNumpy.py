import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

work_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(work_dir)

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

def get_numpy_data(id):
    return train_data.train_data[id].numpy()

def get_label(id):
    return train_data.train_labels[id]

def save_images(dir='mnist_images/'):
    for id in range(10):
        image_array = get_numpy_data(id)
        label = get_label(id)
        img_path = os.path.join(dir, f'{id}-{label}.png')
        numpy2image(image_array, img_path)

def show_data(id):
    plt.imshow(get_numpy_data(id), cmap='gray')
    plt.title('%i' % get_label(id))
    plt.show()

def show_result(image_array, pred_out):
    plt.imshow(image_array, cmap='gray')
    plt.title('predict: %i' % pred_out)
    plt.show()

def numpy2image(image_array, image_path):
    im = Image.fromarray(image_array)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(image_path)

def image2numpy(image_path):
    im = Image.open(image_path)
    image_array = np.array(im) # 转换成np.ndarray格式
    return image_array

def numpy2tensor(image_array):
    # print(image_array[None].shape) # (1, 28, 28), 提升维度
    return torch.from_numpy(image_array[None][None])

def load_image(image_path):
    return image2numpy(image_path)

def load_net(params_pkl_path):
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

    cnn = CNN()
    cnn.load_state_dict(torch.load(params_pkl_path))
    return cnn

def predict(net, image_array):
    image_tensor = numpy2tensor(image_array) /255 # 转成0~1之间的数
    # print(image_tensor)

    out = net(image_tensor)
    idx_max = torch.max(out, 1)[1].data.numpy()[0]
    return idx_max


if __name__ == '__main__':
    # save_images()
    image_path = 'num_images/7-3.png'

    net = load_net('mnist.pkl') # 执行06.2生成
    image_array = load_image(image_path)
    out = predict(net, image_array)
    show_result(image_array, out)

    # image_array = get_numpy_data(1)    
    # image_tensor = numpy2tensor(image_array)
    # print(image_tensor)
    # show_data(1)
    # numpy2image(image_array, image_path)
    # image_array = image2numpy(image_path)
    # print(image_array)