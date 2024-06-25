import pandas as pd
import matplotlib.pyplot as plt
import pywt
from ssqueezepy import cwt, imshow, Wavelet
from ssqueezepy.experimental import scale_to_freq

df = pd.read_excel("FAULT CURRENT LOAD 80MW IF.xlsx")

data = df["If00"].values
time = df["Time0"].values

length = len(data)

'''
coeffs, freq = pywt.cwt(data, length, 'gaus1')
cwtmatr, freqs = pywt.cwt(data, length, 'gaus1')
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
'''

freq = 1000;
Wx, scales = cwt(data, fs=freq)

plt.figure(figsize=(12,6))
wavelet = Wavelet()
freqs_cwt = scale_to_freq(scales, wavelet, length)
ikw = dict(abs=1, xticks=time, xlabel="Time [sec]", ylabel="Frequency [Hz]")
imshow(Wx, **ikw, yticks=freqs_cwt)

'''
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()
'''