import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wavelet_transform(signal, wavelet='db4', level=1):
    
    coeffs = []

    # Define the low-pass (scaling) and high-pass (wavelet) filters
    if wavelet == 'db4':
        scaling_filter = [1, 1]  # Haar scaling filter
        wavelet_filter = [1, -1]  # Haar wavelet filter
    # Add more wavelet filters as needed
    
    for _ in range(level):
        # Pad the filter with zeros for convolution
        padding = [0] * (_ * (len(scaling_filter) - 1))
        scaling_filter_padded = padding + scaling_filter
        wavelet_filter_padded = padding + wavelet_filter

        # Convolve the signal with the filters
        cA = np.convolve(signal, scaling_filter_padded, mode='valid')
        cD = np.convolve(signal, wavelet_filter_padded, mode='valid')

        # Downsample by a factor of 2 (decimation)
        cA = cA[::2]
        cD = cD[::2]

        # Store the approximation and detail coefficients
        coeffs.append(cD)

        # Update the signal for the next level
        signal = cA

    # Add the final approximation coefficients to the list
    coeffs.append(signal)

    return coeffs

wavelet_name = 'db4'
level = 4

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"
df = pd.read_excel(input_excel_file)
data = df["If00"].values
time = df["Time0"].values
#output_excel_file = "Output1.xlsx"

coeffs = wavelet_transform(data, wavelet_name, level)
"""
# Create a DataFrame with the coefficients, padding the shorter arrays with NaNs
max_len = max(len(c) for c in coeffs)
padded_coeffs = [np.pad(c, (0, max_len - len(c)), 'constant', constant_values=np.nan) for c in coeffs]

output_data = pd.DataFrame({f"Approximation_Level_{i+1}": padded_coeffs[i] for i in range(level)},
                           columns=[f"Approximation_Level_{i+1}" for i in range(level)])

# Save the DataFrame to the output Excel file
output_data.to_excel(output_excel_file, sheet_name="Processed_Data", index=False)
"""

plt.figure(figsize=(12, 6))
plt.plot(coeffs[0], color='black')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.title("R-G Fault")
plt.grid()