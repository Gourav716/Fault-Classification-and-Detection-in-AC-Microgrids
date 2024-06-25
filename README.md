A deep learning algorithm using Tensorflow that achieved a 99% accuracy in fault detection and 97% accuracy in fault classification.
The algorithm is able to detect faults even in the presence of noise signals with a signal-to-noise ratio of 30 dB. 

A 200 MW Solar PV plant is designed in PSCAD to generate a variety of fault scenarios and fault signals, which were used to train and test the algorithm. 

Signal processing techniques including Pywavelets, VMD, EMD, and Butterworth filtering are implemented to effectively filter noise signals.
The filtering techniques were able to reduce the noise level by an average of 20 dB, which improved the response time by 15%. 

NVIDIA Jetson Nano development kit was used to acquire and process signals via Python code.
The Jetson Nano was able to acquire and process signals at a rate of 100 kHz, which is sufficient for real-time fault detection. 

Solar PV Plant system designed in PSCAD :-
![PSCAD PV Plant](https://github.com/Gourav716/Fault-Classification-and-Detection-in-AC-Microgrids/assets/58388637/56e8a176-ab59-4ed3-83fe-088b2a938384)


Fault current waveform obtained by simulation of Solar PV Plant :-
![RG Fault](https://github.com/Gourav716/Fault-Classification-and-Detection-in-AC-Microgrids/assets/58388637/b982f35a-5792-456b-9488-f6f169f87c14)


Scalogram of above fault current waveform obtained by applying Continuous Wavelet Transform(CWT) :-
![RG Scalogram](https://github.com/Gourav716/Fault-Classification-and-Detection-in-AC-Microgrids/assets/58388637/933b527e-2712-40b8-934e-c7afebd7ae63)


Confusion matrix of DNN model for fault classification :-
![Confusion Matrix](https://github.com/Gourav716/Fault-Classification-and-Detection-in-AC-Microgrids/assets/58388637/8de8762e-daab-4379-86f4-f0283e0a28ef)


Hardware setup and simulation done in Jetson Nano and signal displayed in Oscilloscope :-
![IMG_20240328_153655](https://github.com/Gourav716/Fault-Classification-and-Detection-in-AC-Microgrids/assets/58388637/21c8fed1-56c2-4ee6-9125-277e41d9f23a)
