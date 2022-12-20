import pandas as pd
import numpy as np

a_array = np.array([
    [1,2],
    [3,4]
])
a_df = pd.DataFrame(
    {"a": [1,3], 
     "b": [2,4]}
)

print("numpy array:\n", a_array)
print("\npandas df:\n", a_df)