from PyEMD import EEMD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If00"].values
time = df["Time0"].values

eemd = EEMD(1, 0)
eImfs = eemd(data, max_imf=1)

num_imfs = len(eImfs)
plt.figure(figsize=(12, 2*num_imfs))
for i, imf in enumerate(eImfs):
    plt.subplot(num_imfs, 1, i+1)
    plt.plot(time, imf, label=f'EIMF {i+1}')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title(f'EIMF {i+1}')
plt.tight_layout()