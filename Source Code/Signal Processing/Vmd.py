import pandas as pd
import matplotlib.pyplot as plt  
from vmdpy import VMD
import time
import numpy as np

input_excel_file = "FAULT CURRENT LOAD 80MW IF.xlsx"

df = pd.read_excel(input_excel_file, sheet_name='Sheet1')

data = df["If00"].values
time_val = df["Time0"].values

alpha = 100000       # moderate bandwidth constraint  
tau = 0            # noise-tolerance (no strict fidelity enforcement)  
K = 1              # 1 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  

start_time = time.time()
#. Run VMD 
u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

end_time = time.time()
run_time = end_time - start_time
print(f"Elapsed Time = {run_time:0.4f} sec")

#. Visualize decomposed modes
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(time_val, data, 'black')
plt.title('Original signal 1')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.grid()

time_val = np.delete(time_val, len(time_val) - 1)

plt.subplot(2,1,2)
plt.plot(time_val, u.T, 'black')
plt.title(f'Decomposed mode, alpha = 1e5, Elapsed Time = {run_time:0.4f} sec')
plt.xlabel('time (s)')
plt.ylabel('current (A)')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()
plt.grid()
"""
output_file = "Imf1.xlsx"

df = pd.DataFrame(u.T)

df.to_excel(output_file, sheet_name="Imf_8", index=False)
"""