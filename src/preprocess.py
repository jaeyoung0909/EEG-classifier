
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import csv
from collections import deque

def csv_load(path):
    data = np.genfromtxt(path, delimiter=',')
    return data

def label2np(path):
    f = open(path, 'r')
    f = f.read().split('\n')
    data = []
    for line in f:
        tmp = line.split(',')
        if len(tmp) == 11:
            data.append(tmp[2])
    data = data[4:-2]
    def criteria(char):
        if char == 'W':
            return 0
        elif char == 'R':
            return 1
        elif char == 'NR':
            return 2
        assert False
    data = [criteria(i) for i in data]
    return np.array(data)

def edf2csv(path):
    edf = mne.io.read_raw_edf(path)
    header = ','.join(edf.ch_names)
    np.savetxt(path.split('.')[0]+'.csv', edf.get_data().T, delimiter=',', header=header)

def fft(data):
    window = np.hamming(data.size)
    N = window.size
    return np.square(np.absolute(2.0/N * np.fft.fft(window * data)[:N//2]))


def eeg2img(data):
    step = 250
    time = 20
    time_step = 2
    freq = 2000
    freq_step = time_step * freq
    eeg1 = []
    eeg2 = []
    emg = []
    for i in range(freq * time // step):
        eeg1_row = []
        eeg2_row = []
        for j in range(24):
            eeg1_row.append(fft(data[i*step : freq_step + i*step, 0])[2*j+1])
            eeg2_row.append(fft(data[i*step : freq_step + i*step, 1])[2*j+1])
        eeg1.append(eeg1_row)
        eeg2.append(eeg2_row)
        emg_sum = sum(fft(data[i*step : freq_step + i*step, 2])[100:200:2])
        emg.append([emg_sum for x in range(24)])
    eeg1 = np.array(eeg1)
    eeg1 = (eeg1 - np.mean(eeg1)) / np.std(eeg1)
    eeg2 = np.array(eeg2)
    eeg2 = (eeg2 - np.mean(eeg2)) / np.std(eeg2)
    emg = np.array(emg)
    emg = (emg - np.mean(emg)) / np.std(emg)
    
    arr = np.dstack((eeg1, eeg2, emg))
    return arr

def eeg2csv(path_eeg):
    batch = 20 * 2000 + 4000 - 250
    step = 4 * 2000
    f = open(path_eeg)
    data = csv.reader(f)
    arr = []
    target = deque([])
    for i, x in enumerate(data):
        if i == 0:
            continue
        if i-1 < batch:
            target.append([float(i) for i in x])
            continue 
        if (i-1-batch)%step == 0:
            print("step {}".format((i-1-batch)//step + 1))
            arr.append(eeg2img(np.array(target)))
        target.popleft()
        target.append([float(i) for i in x])
    save_file = open(path_eeg.split('.')[0] + '.npy', 'wb')
    np.save(save_file, np.array(arr))
    save_file.close()


    # arr = []
    # for i, chunk in enumerate(pd.read_csv(path_eeg, chunksize=step)):
    #     if i+1 > 10796:
    #         continue
    #     print("step {}".format(i+1))
    #     data = chunk.to_numpy()
    #     arr.append(eeg2img(data))
    
    # f = open(path_eeg.split('.')[0] + '_chunk.npy', 'wb')
    # np.save(f, np.array(arr))
    # f.close()
# eeg2csv("EDF/180812_CH1_46_SD.csv")

# edf2csv("EDF/180812_CH1_46_SD.edf")

# data = csv_load("EDF/190605C1_5min.csv")
# freq = np.fft.fftfreq(4000, 0.0005)
# y = fft(data[0:4000, 2])
# plt.plot(freq[:2000], y)
# print(freq)
# plt.show()
# print(eeg2img(data[:2000*20]).shape)

# labeltxt2csv("EDF/190605C1_5min_Result.txt")
# print(len(labeltxt2csv("EDF/190605C1_5min_Result.txt")))

# eeg2csv("EDF/190605C1.csv")
b = np.load("EDF/180812_CH1_46_SD.npy")[:10796]
print(b.shape)
# print(len(csv_load("EDF/190605C1.csv"))/20*2000)
print(label2np("EDF/180812_CH1_46_SD_1.txt").shape)
