
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/Advertising.csv')

# plt.scatter(data.TV, data.sales)
# plt.show()

x = data.iloc[:, 1:-1] # TV,radio,newspaper三列数据
y = data.iloc[:, -1] # sales列数据

"""
Dense:
参数1：输出10个数据
参数2: input_shape= 输入数据3个
参数3: activation= 激活函数
"""
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'), # 隐藏层
                             tf.keras.layers.Dense(1)]) # 输出层
model.summary() # 显示 层

model.compile(optimizer='adam', loss='mse') # 输出只有1个数，用mse

# 训练
history = model.fit(x, y, epochs=500) # 训练100次

# 预测
print(model.predict(x[:10])) # 使用模型出输出
print(model.predict([[230.1, 37.8, 69.2], [67.8,36.6,114]])) # 使用模型出输出
