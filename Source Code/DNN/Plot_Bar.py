import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

df = pd.read_excel("Model_Accuracy.xlsx") 
x = df["Model"] 
y = df['Accuracy']

plt.figure(figsize=(12,6))
plt.rcParams.update({'font.size': 12})
plt.bar(x, y, width=0.5)

for i in range(len(x)):
    plt.text(i, y[i] + 2, str(y[i]) + '%', ha = 'center')

plt.xlabel("DNN Model", fontsize='12') 
plt.ylabel("Accuracy (in %)")
plt.ylim(0, 110)
plt.yticks(np.arange(0, 110, 20))