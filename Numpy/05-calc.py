import numpy as np

'''
加减乘除
+-*/
np.dot()
数据统计分析
np.max() np.min() np.sum() np.prod() np.count()
np.std() np.mean() np.median()
特殊运算符号
np.argmax() np.argmin()
np.ceil() np.floor() np.clip()
'''

a = np.array([150, 166, 183, 170])
print("a + 3:", a + 3)
print("a - 3:", a - 3)
print("a * 3:", a * 3)
print("a / 3:", a / 3)

# 矩阵相乘
a = np.array([
[1, 2],
[3, 4]
])
b = np.array([
[5, 6],
[7, 8]
])

print(a.dot(b))
print(np.dot(a, b))

# 最大值最小值
a = np.array([150, 166, 183, 170])
print("最大：", np.max(a))
print("最小：", a.min())
print("求和：", a.sum())
print("累乘：", a.prod())
print("总数：", a.size)  
a = np.array([0, 1, 2, 3])
print("非零总数：", np.count_nonzero(a))

month_salary = [1.2, 20, 0.5, 0.3, 2.1]
print("平均工资：", np.mean(month_salary))
print("工资中位数：", np.median(month_salary))

month_salary = [1.2, 20, 0.5, 0.3, 2.1]
print("标准差：", np.std(month_salary))

# 找到最大最小的索引
a = np.array([150, 166, 183, 170])
name = ["小米", "OPPO", "Huawei", "诺基亚"]
high_idx = np.argmax(a)
low_idx = np.argmin(a)
print("{} 最高".format(name[high_idx]))
print("{} 最矮".format(name[low_idx]))


a = np.array([150.1, 166.4, 183.7, 170.8])
print("向上取整:", np.ceil(a))
print("向下取整:", np.floor(a))

a = np.array([150.1, 166.4, 183.7, 170.8])
print("clip:", a.clip(160, 180)) # [160.  166.4 180.  170.8]

condit = (a > 160) & (a < 180)
print("clip:", a[condit]) # [166.4 170.8]