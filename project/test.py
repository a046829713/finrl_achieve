import pandas as pd
import numpy as np


# Sample data provided by the user
data = {
    'date': ['2013-01-02', '2013-01-03', '2013-01-04', '2013-01-07'],
    'tic': ['1301.TW', '1301.TW', '1301.TW', '1301.TW'],
    'close': [79.000000, 79.800003, 80.300003, 80.000000],
    'high': [79.400002, 79.800003, 80.300003, 81.000000],
    'low': [78.199997, 79.099998, 78.599998, 79.199997]
}

# Convert the sample data to a DataFrame
df = pd.DataFrame(data)


# Assume the normalization window size n is equal to the number of samples we have, which is 4 in this case.
n = len(df)


# Create the normalized price matrices




print(V_t)
print(V_t_hi)
print(V_t_lo)
# Stack the normalized price matrices to form X_t
X_t = np.hstack((V_t, V_t_hi, V_t_lo))


