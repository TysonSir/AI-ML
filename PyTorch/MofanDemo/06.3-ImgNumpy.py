import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
    return train_data.train_labels[0]

def show_data(id):
    plt.imshow(get_numpy_data(id), cmap='gray')
    plt.title('%i' % get_label(id))
    plt.show()

def numpy2image(image_array, image_path):
    im = Image.fromarray(image_array)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(image_path)

def image2numpy(image_path):
    im = Image.open(image_path)
    image_array = np.array(im) # 转换成np.ndarray格式
    return image_array

if __name__ == '__main__':
    image_path = 'num.png'
    # image_array = get_numpy_data(1)
    show_data(1)
    # numpy2image(image_array, image_path)
    # image_array = image2numpy(image_path)
    # print(image_array)