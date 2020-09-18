# EEG-classifier
This project is EEG data classifier for researcher obtaining EEG data from mice. It classifies whether given EEG data indicate which state; W(wake), R(ram), NR(non-ram). Implemented model is based on [this paper](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1006968&type=printable).

## How to use

We need EEG dataset, which consist of EEG1, EEG2, EMG. Labeled data is also needed, if you need training for more precise results.  

### 1. Data Preprocessing

- Convert EEG dataset with .edf format file to .csv format.

`python src/main.py --edf2csv edf_file_path`

- Convert EEG dataset with .csv to 3 channel numpy array calculated by fast fourier transform with hamming window.

`python src/main.py --eeg2np csv_file_path`

### 2. Training

Default labeled data format is extracted from "SleepSign for Animal".

`python src/main.py --train EEG.npy_file_path Labeled_txt` 

#### Reference
- SPINDLE:End-to-end learning from EEG/EMG to extrapolate animal sleep scoring across experimental settings, labs and species
