import pandas as pd
import numpy as np

data = np.arange(-12, 12).reshape((6, 4))
df = pd.DataFrame(
  data, 
  index=list("abcdef"), 
  columns=list("ABCD"))
print(df)

# 选Column
print(df["B"])
print("numpy:\n", data[:, [2,1]])
print("\ndf:\n", df[["C", "B"]])

# loc
print(data[2:3, 1:3]) # 3不算
print(df.loc["c":"d", "B":"D"])#D算进去
print(df.loc["c":"d", :])

print("numpy:\n", data[[3,1], :])
print("\ndf:\n", df.loc[["d", "b"], :])

df2 = pd.DataFrame(
  data, 
  index=list("beacdf"), 
  columns=list("ABCD"))
print(df2)
print(df2.loc["e":"c"])

# iloc
print("numpy:\n", data[2:3, 1:3])
print("\ndf:\n", df.iloc[2:3, 1:3])

print("numpy:\n", data[[3,1], :])
print("\ndf:\n", df.iloc[[3, 1], :])

print("\n---------------------------------df:\n", df.iloc[0]) # 第一行

# loc和iloc混搭
row_labels = df.index[2:4] # 索引转换的方式
print("row_labels:\n", row_labels)
print("\ndf:\n", df.loc[row_labels, ["A", "C"]])

col_labels = df.columns[[0, 3]]
print("col_labels:\n", col_labels)
print("\ndf:\n", df.loc[row_labels, col_labels])

# 我想要找 A B 两个特征的 前两个数据
col_index = df.columns.get_indexer(["A", "B"])
print("col_index:\n", col_index)
print("\ndf:\n", df.iloc[:2, col_index])
# df.index.get_indexer(["a", "b"]) 也可以这样获取到 label 对应的 index 信息

# 条件过滤筛选
print(df[df["A"] < 0])

# 选在第一行数据不小于 -10 的数据
print("~:\n", df.loc[:, ~(df.iloc[0] < -10)])
print("\n>=:\n", df.loc[:, df.iloc[0] >= -10])
# 选在第一行数据不小于 -10 或小于 -11 的数据
i0 = df.iloc[0]
df.loc[:, ~(i0 < -10) | (i0 < -11)]

print(~(df.iloc[0] < -10)) # True False信息
# df.iloc[:, ~(df.iloc[0] < -10)] # 错误，iloc不接受bool信息

# Series和DataFrame类似
# Series于list的区别，就是可以把数字索引改成 自定义索引（dict）
