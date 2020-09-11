
import numpy as np
import matplotlib.pyplot as plt

def csv_load(path):
    data = np.genfromtxt(path, delimiter=',')
    return data

def fft(data):
    window = np.hamming(data.size)
    N = window.size
    return np.square(np.absolute(2.0/N * np.fft.fft(window * data)[:N/2]))


def eeg2img(data):
    step = 250
    time = 20
    time_step = 2
    freq = 2000
    freq_step = time_step * freq
    eeg1 = []
    eeg2 = []
    emg = []
    for i in range(freq * time / step):
        print("step {}".format(i+1))
        eeg1_row = []
        eeg2_row = []
        for j in range(24):
            eeg1_row.append(fft(data[i*step : freq_step + i*step, 0])[j+1])
            eeg2_row.append(fft(data[i*step : freq_step + i*step, 1])[j+1])
        eeg1.append(eeg1_row)
        eeg2.append(eeg2_row)
        emg_sum = sum(fft(data[i*step : freq_step + i*step, 2])[1:30])
        emg.append([emg_sum for x in range(24)])
    arr = np.dstack((eeg1, eeg2, emg))
    std = (arr - np.mean(arr)) / np.std(arr)
    return std

data = csv_load("EDF/190605C1_5min.csv")
# freq = np.fft.fftfreq(4000, 0.0005)
# y = fft(data[0:4000, 2])
# plt.plot(freq[:2000], y)
# print(freq)
# plt.show()

print(eeg2img(data).shape)
