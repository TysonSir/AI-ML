import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

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

def show_mnist(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.title('%i' % train_data.train_labels[0])
    plt.show()

def numpy2image(image_array, image_path):
    im = Image.fromarray(image_array)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(image_path)

if __name__ == '__main__':
    image_array = get_numpy_data(1)
    show_mnist(image_array)
    numpy2image(image_array, 'num.png')