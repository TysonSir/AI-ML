import numpy as np


'''
单个选取
array[1]
array[1,2,3]
array[1][1]
切片划分
array[:3]
array[2:4, 1:3]
条件筛选
array[array<0]
np.where(array, array < 0)
'''

a = np.array([1, 2, 3])
print("a[0]:", a[0])
print("a[1]:", a[1])

print("a[[0,1]]:\n", a[[0,1]])
print("a[[1,1,0]]:\n", a[[1,1,0]])


b = np.array([
[1,2,3,4],
[5,6,7,8],
[9,10,11,12]
])

# 选第 2 行所有数
print("b[1]:\n", b[1])   

# 选第 2 行，第 1 列的数
print("b[1,0]:\n", b[1,0])   

# 这个看着有点纠结，如果对应到数据，
# 第一个拿的是数据位是 [1,2]
# 第二个拿的是 [0,3]
print("b[[1,0],[2,3]]:\n", 
b[[1,0],
[2,3]]) #  [7 4]

print("b[[1,0],[2,3]]:\n", 
b[[1,0,2], # 所有行号
[2,3,1]]) # 所有列号
#  [7 4 10]


# 切片
a = np.array([1, 2, 3])
print("a[0:2]：\n", a[0:2])
print("a[1:]：\n", a[1:])
print("a[-2:]：\n", a[-2:])

b = np.array([
[1,2,3,4],
[5,6,7,8],
[9,10,11,12]
])

print("b[:2]:\n", b[:2])
print("b[:2, :3]:\n", b[:2, :3])
print("b[1:3, -2:]:\n", b[1:3, -2:])

# 条件筛选
a = np.array([
[1,2,3,4],
[5,6,7,8],
[9,10,11,12]
])
print(a[a>7]) # [ 8  9 10 11 12]

condition = (a > 7) & (a != 10)
print(a[condition])

print(np.where(condition, -1, a))

condition = a > 7
print(np.where(condition, -1, 2))

condition = a > 7
b = -a - 1
print(np.where(condition, a, b))