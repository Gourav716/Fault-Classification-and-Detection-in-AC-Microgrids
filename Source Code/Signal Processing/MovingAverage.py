import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If00"]
time = df["Time0"].values

window_size = 500

def simple_moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def cumulative_moving_average(signal):
    cma = []
    cma.append(signal[0]) # data point at t=0

    for i, x in enumerate(signal[1: ], start=1):
        cma.append(((x+(i+1)*cma[i-1])/ (i+1)))
 
    return cma

def weighted_moving_average(Data, period):
    weighted = []
    for i in range(len(Data)):
            try:
                total = np.arange(1, period + 1, 1) # weight matrix
                matrix = Data[i - period + 1: i + 1, 3:4]
                matrix = np.ndarray.flatten(matrix)
                matrix = total * matrix # multiplication
                wma = (matrix.sum()) / (total.sum()) # WMA
                weighted = np.append(weighted, wma) # add to array
            except ValueError:
                pass
    return weighted

#filtered_df = df[(df["Noise"] > 0.01) & (df["Noise"] < 0.012)]

#filtered_df["Moving_Average"] = filtered_df["Noise"].rolling(window_size).mean()
"""
new_data = np.empty(len(data))
for i in range(0, len(new_data)):
    if ((data[i] > 0.01)  & (data[i] < 0.012)):
        new_data[i] = data[i]

new_list = []
new_data = np.empty(len(data))
for i in range(0, len(new_data)):
    if len(new_list) % 2 == 0:
        if (data[i] > 0.015):
            new_list.append(i)
    if len(new_list) % 2 != 0:
        if (data[i] < 0.005):
            new_list.append(i)
    if len(new_list) % 2 == 0:
        if (data[i] < -0.025):
            new_list.append(i)
    if len(new_list) % 2 != 0:
        if (data[i] > 0.005):
            new_list.append(i)
       
print(new_list)
for i in range(0, len(new_list)):
    print(data[new_list[i]])
"""
"""
df2 = pd.Series(data[new_list])

df2["Moving_Average"] = df2.rolling(window_size).mean()
"""
"""
incr = data.diff().ge(0)

# shifted trend (local minima)
shifted = incr.ne(incr.shift())

# local max
local_max = shifted & (~incr)

# thresholding function
def thresh(x, threshold=0.03, step=1):
    ret = pd.Series([0]*len(x), index=x.index)
    t = x.min() + threshold
    ret.loc[x.gt(t)] = step
    return ret

signal = data.groupby(local_max.cumsum(), group_keys=True).apply(thresh)
signal += data.min()

out = np.array(signal)
"""

new_data = simple_moving_average(data, window_size)
plt.figure(figsize=(12,6))
#plt.plot(data, label='Filtered Signal', color='b')
plt.plot(time, new_data, label="Filtered Signal", color='r')
#plt.plot(new_data, label="Filtered Signal", color='r')
#plt.plot(data[new_list], label='New Signal', color='r', marker='o')
plt.grid()

