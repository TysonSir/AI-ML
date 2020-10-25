
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/Income1.csv')

# 显示原始数据
# plt.scatter(data.Education, data.Income)
# plt.show()

x = data.Education
y = data.Income

model = tf.keras.Sequential() # 创建顺序模型
model.add(tf.keras.layers.Dense(1, input_shape=(1, ))) # 输出维度1（y），输入维度1 （x）
model.summary() # 显示，Dense层 -> y=ax+b方程

# 配置+编译模型
# optimizer 优化方法( adam 梯度下降算法 Adaptive momentum ）
# loss 损失函数( mse 均方差 Mean Square Error )
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=5000) # 训练15000次

print(model.predict(x)) # 使用模型出输出
print('20 year income:', model.predict(pd.Series([20]))) # 使用模型出输出
