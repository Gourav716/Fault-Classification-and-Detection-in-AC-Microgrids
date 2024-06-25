from PyEMD import EMD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If00"].values
time = df["Time0"].values

emd = EMD()
Imfs = emd(data, max_imf=1)

num_imfs = len(Imfs)
plt.figure(figsize=(12, 2*num_imfs))
#plt.title("Y-G Fault IMF", y=1.03)
for i, imf in enumerate(Imfs):
    plt.subplot(num_imfs, 1, i+1)
    plt.plot(time, imf, label=f'IMF {i+1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.title(f'IMF {i+1}')
    plt.grid()
    """
    output_file = f"YG_IMF {i + 1}.xlsx"
    df = pd.DataFrame(New_Imfs[i].T)
    df.to_excel(output_file, index=False)
    """
plt.tight_layout()