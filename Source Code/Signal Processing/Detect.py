import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the signal data from an Excel file (replace with your file path)
input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)

# Define parameters
window_size = 10  # Window size for the moving average filter

# Apply the moving average filter
df['Filtered_Value'] = (df['If00'].rolling(window=window_size).mean())

#change = abs(df['Noise'] - df['Filtered_Value'])
# Detect load changes
#df['Load_Change'] = np.logical_and(change > 0.02, change < 0.3)
"""
result = np.empty(len(df['Filtered_Value']))

for i in range(len(df['Filtered_Value'])):
    if (np.logical_and(df['Filtered_Value'][i] < 0.012, df['Filtered_Value'][i] > 0.001)):
        result[i] = 0.05
    else:
        result[i] = 0
"""
# Plot the original signal and filtered signal
plt.figure(figsize=(10, 6))
#plt.plot(df['If00'], label='Original Signal', color='b')
plt.plot(df['Filtered_Value'], label='Filtered Signal', color='r')
#plt.plot(result, color='g')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Original vs. Filtered Signal')
plt.legend()
plt.grid(True)

# Identify load change points
#load_change_points = df[df['Load_Change']]
#if not load_change_points.empty:
#    print("Load changes detected at the following time points:")
#    print(load_change_points)
#else:
#    print("No load changes detected.")

plt.show()
