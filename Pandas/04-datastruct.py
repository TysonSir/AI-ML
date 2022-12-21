import pandas as pd
import numpy as np

# 数据序列Series
l = [11,22,33]
s = pd.Series(l)
print("list:", l)
print("series:", s)

s = pd.Series(l, index=["a", "b", "c"])
print("series:", s)

s = pd.Series({"a": 11, "b": 22, "c": 33})
print("series:", s)

s = pd.Series(np.random.rand(3), index=["a", "b", "c"])

print("array:", s.to_numpy())
print("list:", s.values.tolist())

# 数据表DataFrame
df = pd.DataFrame([
  [1,2],
  [3,4]
])
print(df)
# 第 0 行，第 1 列
# 或 第一个维度中的第 0 号，第二个维度中的第 1 号
print(df.at[0, 1])  

df = pd.DataFrame({"col1": [1,3], "col2": [2, 4]})
print(df)


print('-'*10)
print(df["col1"], "\n")
print("取出来之后的 type：", type(df["col1"])) # Series

df = pd.DataFrame({"col1": pd.Series([1,3]), "col2": pd.Series([2, 4])})
print(df)

# 构建特殊索引
s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
df = pd.DataFrame({"col1": [1,3], "col2": [2, 4]}, index=["a", "b"])
print(s, "\n")
print(df)
print(df.index)
print(df.columns)


my_json_data = [
  {"age": 12, "height": 111},
  {"age": 13, "height": 123}
]
df = pd.DataFrame(my_json_data, index=["jack", "rose"])
print(df)
print(df.to_numpy())

df = df.append(
    pd.Series(
        [0, 0],
        index=df.columns,
        name='nick',
    )
)
print(df)

df = pd.concat([df,
    pd.Series(
        [0, 0],
        index=df.columns,
        name='tim',
    ).to_frame().T
])
print(df)