import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("FAULT CURRENT LOAD 80MW IF.xlsx")

data = df["If01"].values
time = df["Time0"].values

plt.figure(figsize=(12,6))
plt.plot(time, data)
plt.xlabel('time (s)')
plt.ylabel('current (A)')
#plt.title("R-G Fault")
plt.grid()