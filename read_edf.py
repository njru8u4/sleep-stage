import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import re
import mne
from scipy.signal import find_peaks

def read_stage(path):
    fp = open(path, 'r')
    lines = fp.readlines()
    fp.close()
    counter = 0
    anns = []
    for i in range(len(lines)-4):
        if lines[i].split('\t')[0] == '#':
            counter = counter + 1
        if counter >= 3:
            j = i + 2
            lines[j] = lines[j].split('\n')[0]
            temp = (re.split(',\t\t|,\s\t|,\t|,\t', lines[j]))
            anns.append(temp)
            
    anns = np.array(anns)
    anns = anns[:,[3,6]].astype(float).astype(int)
    return anns

def cut_epoch(raw, ann, rate):
    raw_data = []
    label = []
    counts = [0,0,0,0,0,0,0,0]
    S2_counter = 0
    W_counter = 0
    REM_counter = 0
    S1_counter = 0
    for i in range(3, len(ann[:,0])-4):
        start = ann[i-2,0]*rate
        end = (ann[i+2,0] + 30)*rate
        temp = raw[start:end]
        if len(temp) != rate*30*5 or ann[i,1] > 5:
            continue
        peaks, _ = find_peaks(temp, distance=40, height=1, width=20)
        #print (peaks.size)
        if peaks.size < 20:
            continue
        if ann[i,1] == 2:
            #ann[i, 1] = 1
            S2_counter = S2_counter + 1
            #if S2_counter > 70:
            #    continue
        elif ann[i,1] == 0:
            W_counter = W_counter + 1
            #if W_counter > 100:
            #    continue
        elif ann[i, 1] == 5:
            ann[i, 1] = 4
            REM_counter = REM_counter + 1
            #if REM_counter > 100:
            #    continue
        elif ann[i,1] == 1:
            S1_counter = S1_counter + 1
            #if S1_counter > 70:
            #    continue
        elif ann[i,1] == 4:
            ann[i,1] = 3
        elif ann[i, 1] == 3:
            ann[i, 1] = 3
        
        raw_data.append(temp)
        counts[ann[i,1]] = counts[ann[i,1]] + 1
        
        
        label.append(ann[i,1])
    return [raw_data, label, counts]


PPG = []
label = []
stage_counts = []
rate = 125
for root, dir, f in os.walk('./psg-datavase01'):
    for files in f:
        [name, extension] = files.split('.')
        if extension == 'edf':
            path = os.path.join(root, files)
            data = mne.io.read_raw_edf(path, preload = True)
            data = data.resample(rate)

            raw = data.to_data_frame()
            if 'Plesmo' in raw.columns:
                PPG_temp = np.array(raw.loc[:, 'Plesmo'])
            elif 'Plethysmo' in raw.columns:
                PPG_temp = np.array(raw.loc[:, 'Plethysmo'])
            else:
                continue
            
            txts = name + '.TXT'
            if txts not in f:
                txts = name + '.txt'
            txt_path = os.path.join(root, txts)
            ann = read_stage(txt_path)
            
            [temp1, temp2, temp3] = cut_epoch(PPG_temp, ann, rate)
            
            PPG = PPG + temp1
            label = label + temp2
            stage_counts.append(temp3)
            
            del raw
            del data
            gc.collect()


PPG = np.array(PPG).reshape(len(PPG), len(PPG[0]), 1)
label = np.array(label)
stage_counts = np.array(stage_counts)

#np.savetxt('./datas/all_stages.csv', stage_counts, delimiter=",")

from sklearn.preprocessing import OneHotEncoder
for i in range(len(PPG)):
    if len(PPG[i]) != rate*30*5:
        print (len(PPG[i]))
print (len(label))
print (len(PPG))
PPG = np.array(PPG).reshape(len(PPG), len(PPG[0]))
label = np.array(label)

label = label.reshape(len(label),1)
onehotencoder = OneHotEncoder()
#np.save('label-5-30s-class', label)
label = onehotencoder.fit_transform(label).toarray()

print (label.shape)
np.save('./datas/PPG-5-150s-mid-all', PPG)
np.save('./datas/label-5-150s-mid-all', label)
#np.savetxt('PPG-5.csv', PPG, delimiter=',')
#np.savetxt('label-5.csv', label, fmt="%d",delimiter=',')

"""
from scipy.signal import find_peaks
from scipy import interpolate
import matplotlib.pyplot as plt

interval = []
sx = np.arange(0, 3750)
#print (PPG_.shape)
for i in range(PPG.shape[0]):
    peaks, loca = find_peaks(PPG[i,:], distance=40, height=1, width=20)
    distance = peaks[1:-1] - peaks[0:-2]
    distance = np.array(distance)
    #func1 = interpolate.UnivariateSpline(peaks[1:-1], distance, s=0)
    #sy = func1(sx)
    distance = np.concatenate((label[i], distance))
    interval.append(distance)
#inter = np.array(interval)
#np.savetxt('inter.csv', inter, fmt="%d", delimiter=',')
import csv
with open('inter1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(interval)
"""