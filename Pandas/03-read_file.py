import pandas as pd
df = pd.read_excel("体检数据.xlsx", index_col=0)
print(df)

df = pd.read_excel("体检数据.xlsx")
print(df)

df.loc[2, "体重"] = 1
print(df)
df.to_excel("体检数据_修改.xlsx")

