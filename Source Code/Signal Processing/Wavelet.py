import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If31"].values
time = df["Time3"].values

cA, cD = pywt.dwt(data, 'db4')

time1 = np.zeros(len(cD))

for i in range(1, len(time1)):
    time1[i] = i / 4000

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(time, data, color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.title("Original Signal")
plt.grid()

plt.subplot(2,1,2)
plt.plot(time1, cD, color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.title("R-G Fault PyWavelet db4")
plt.grid()
plt.tight_layout()