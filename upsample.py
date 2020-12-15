import numpy as np
from scipy.signal import find_peaks
import random
from numba import jit

# 0:4倍, 1:1, 2:20倍, 3:4倍
@jit
def upsampling(label, inter):
    new_data = []
    new_label = []

    for i in range(len(label)):
        if label[i,1] == 1:
            continue
        elif label[i,0] == 1 or label[i, 3] == 1:
            for _ in range(3):
                new_PPG = []
                for j in range(100*5):
                    if inter[i,j] != 0:
                        temp = inter[i, j] + 8*random.randint(-20, 20)
                        new_PPG.append(temp)
                    else:
                        new_PPG.append(inter[i, j])
                new_data.append(new_PPG)
                new_label.append(label[i, :])
                #new_PPG.clear()
                #PPG_raw[i]
        elif label[i,2] == 1:
            for _ in range(20):
                new_PPG = []
                for j in range(100*5):
                    if inter[i,j] != 0:
                        temp = inter[i, j] + 8*random.randint(-20, 20)
                        new_PPG.append(temp)
                    else:
                        new_PPG.append(inter[i,j])
                new_data.append(new_PPG)
                new_label.append(label[i,:])
                #new_PPG.clear()
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    return new_data, new_label

def raw_to_up_inter(PPG_raw, label):

    stages = [0]*5

    inter = np.zeros((PPG_raw.shape[0], 100*5))
    for i in range(PPG_raw.shape[0]):
        peaks, _ = find_peaks(PPG_raw[i,:], distance=40, height=1, width=20)
        #peaks = np.array(peaks)
        distance = peaks[1:] - peaks[0:-1]
        inter[i, :len(distance)] = distance*8
    #print (inter)

    for i in range(len(label[0])):
        stages[i] = sum(label[:,i])
    print (stages)
    #exit()
    new_data, new_label = upsampling(label, inter)

    print (new_data.shape)
    print (new_label.shape)

    new_PPGs = np.concatenate((inter, new_data), axis=0)
    new_labels = np.concatenate((label, new_label), axis=0)

    for i in range(len(new_labels[0])):
        stages[i] = sum(new_labels[:,i])
    print (stages)

    #np.save("./datas/inter-4-150s-upsampled", new_PPGs)
    #np.save("./datas/label-4-150s-upsampled", new_labels)

def raw_to_inter(raw_path, label_path):
    PPG_raw = np.load(raw_path)
    label = np.load(label_path)

    inter = np.zeros((PPG_raw.shape[0], 100*5))
    for i in range(PPG_raw.shape[0]):
        peaks, _ = find_peaks(PPG_raw[i,:], distance=40, height=1, width=20)
        #peaks = np.array(peaks)
        distance = peaks[1:] - peaks[0:-1]
        inter[i, :len(distance)] = distance*8
    return inter, label

if __name__ == "__main__":
    PPG_raw = np.load("./datas/PPG-4-150s-mid-all.npy")
    label = np.load("./datas/label-4-150s-mid-all.npy")
    raw_to_up_inter(PPG_raw, label)