import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt

def butterworth_filter(data, cutoff_freq, sampling_freq, order=5):
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

excel_file = 'noisy.xlsx'
sheet_name = 'Sheet1'
column_name = 'Noise(2)'

df = pd.read_excel(excel_file, sheet_name=sheet_name)
data = df[column_name].values

# Filter parameters
sampling_freq = 1000  # Sample frequency in Hz
cutoff_freq = 2      # Cutoff frequency in Hz

filtered_data = butterworth_filter(data, cutoff_freq, sampling_freq)

# Plot original and filtered signals
time = np.arange(len(data)) / sampling_freq
plt.figure(figsize=(12, 6))
    
plt.subplot(2, 1, 1)
plt.plot(time, data, label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.legend()
plt.grid()
    
plt.subplot(2, 1, 2)
plt.plot(time, filtered_data, label='Filtered Signal', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Signal')
plt.legend()
plt.grid()
plt.tight_layout()
"""
output_file = "Filter(2).xlsx"

a = np.array(time)
b = np.array(filtered_data)

df = pd.DataFrame({"Time": a, "Filtered_Data": b})

df.to_excel(output_file, sheet_name="Butterworth", index=False)
"""