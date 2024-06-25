import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

df = pd.read_excel("FAULT CURRENT LOAD 80MW IF.xlsx")

data = df["If00"].values
time = df["Time0"].values

target_snr_db = 20
signal_avg = np.mean(data)
signal_avg_db = 10 * np.log10(signal_avg)
# Calculate noise
noise_avg_db = signal_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(data))
# Noise up the original signal
noisy_data = data + noise

plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
plt.plot(time, noisy_data, color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()

emd = EMD()
Imfs = emd(noisy_data)

plt.subplot(2, 1, 2)
plt.plot(time, Imfs[0], color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()