import numpy as np

# 一维拼接
cars1 = np.array([5, 10, 12, 6])
cars2 = np.array([5.2, 4.2])
cars = np.concatenate([cars1, cars2])
print('一维拼接：', cars)

print('-' * 20)


# 二维拼接
a = np.array([
[1,2],
[3,4]
])
b = np.array([
[5,6],
[7,8]
])
print("竖直合并\n", np.vstack([a, b]))
print("水平合并\n", np.hstack([a, b]))

print('-' * 20)


# 创建数据，观察大小
cars = np.array([
    [5, 10, 12, 6],
    [5.1, 8.2, 11, 6.3],
    [4.4, 9.1, 10, 6.6]
])

print('数据：\n', cars)
print('维度数：', cars.ndim)
print("总共多少测试数据：", cars.size)
print("第一个维度：", cars.shape[0])
print("第二个维度：", cars.shape[1])
print("所有维度：", cars.shape)
