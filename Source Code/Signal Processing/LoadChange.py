import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If00"]
time = df["Time0"].values

thres = 0.06

index = []
def DetectWave(signal):
    wave = np.empty(len(signal))

    for i in range(1, len(signal)):
        if (signal[i] - signal[i - 1]) > thres:
            wave[i] = signal[i]
            index.append(i)
        elif (signal[i] - signal[i - 1]) < -thres:
            wave[i] = signal[i]
            index.append(i)
        else:
            wave[i] = 0

    return wave

wave = DetectWave(data)

new_data = np.empty(len(data))
for i in range(1, len(index)):
    if ((i % 2 != 0) & (index[i] + 1 <= len(data))):
        for j in range(index[i - 1], index[i] + 1):
            new_data[j] = data[j]

plt.figure(figsize=(12,6))
"""
plt.subplot(2, 1, 1)
plt.plot(time, data, label='Original Signal', color='blue')
plt.title('Original Signal')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()
plt.subplot(2, 1, 2)
"""
plt.plot(time, new_data, label='Filtered Signal', color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()
plt.legend()
plt.tight_layout()