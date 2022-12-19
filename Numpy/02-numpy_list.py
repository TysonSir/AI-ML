'''numpy数据类型都一样，速度更快于list'''
import numpy as np

my_array = np.array([1, 2, 3.5])
print(my_array[0])

my_array[0] = -1
print(my_array) # [-1.   2.   3.5]

my_array = np.array([1, 2, 3.5, '999'])
print(my_array) # ['1' '2' '3.5' '999']