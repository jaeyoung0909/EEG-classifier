import sys, getopt
import argparse

from train import EEG_Model
from preprocess import edf2csv, eeg2np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--edf2csv')
    parser.add_argument('-n', '--eeg2np')
    parser.add_argument('-t', '--train', nargs='+')
    args = parser.parse_args()
    if args.edf2csv:
        edf2csv(args.edf2csv)
    elif args.eeg2np:
        eeg2np(args.eeg2np)
    elif args.train:
        model = EEG_Model()
        model.fit(args.train[0], args.train[1])
        model.train()

if __name__ == "__main__":
    main()