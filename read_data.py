import numpy as np
from numba import jit
import scipy
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal import welch
import time
from multiprocessing import Pool

def tt(inter, time_stamp):
    sx = np.arange(0, 1000*30*time_stamp)
    x = [0]
    temp = []
    for j in range(100*time_stamp):
        if inter[j] == 0:
            break
        temp.append(inter[j])
        x.append(x[j] + inter[j])

    func1 = interpolate.UnivariateSpline(x[1:], temp, s=0)
    sy = func1(sx)
    sy = sy - np.mean(sy)
    _, p = welch(x = sy, fs = 1000, window="hann")
    return p


def read_frequency(inter_path, time_stamp):
    print ("in read start")
    inter = np.load(inter_path)
    pool = Pool(4)
    now = time.time()
    #f_domain = pool.starmap(tt, ((inter[i,:], time_stamp) for i in range(100)))
    f_domain = pool.starmap(tt, ((inter[i,:], time_stamp) for i in range(inter.shape[0])))
    print (time.time() - now)
    print ("pool end")
    
    #interval = np.array(interval)
    #interval = interval.reshape(len(interval), len(interval[0]), 1)
    f_domain = np.array(f_domain)
    f_domain = f_domain.reshape(len(f_domain), len(f_domain[0]), 1)

    return f_domain


def read_inter(inter_path, time_stamp):
    inter = np.load(inter_path)
    inter = inter.reshape(len(inter), len(inter[0]), 1).astype(int)
    
    return inter

def read_label(label_path):
    label = np.load(label_path)
    return label

def from_raw(raw_data_path, label_path, time_stamp):
    PPG_ = np.load(raw_data_path)
    label = np.load(label_path)

    interval = []
    f_domain = []
    sx = np.arange(0, 125*30*time_stamp)
    inter = np.zeros((PPG_.shape[0], 100*time_stamp))
    for i in range(PPG_.shape[0]):
        peaks, _ = find_peaks(PPG_[i,:], distance=40, height=1, width=20)
        distance = peaks[1:-1] - peaks[0:-2]
        inter[i, :len(distance)] = distance
        func1 = interpolate.UnivariateSpline(peaks[1:-1], distance, s=0)
        sy = func1(sx)
        _, p = welch(sy, 125)
        f_domain.append(p)
        interval.append(sy)


    interval = np.array(interval)
    interval = interval.reshape(len(interval), len(interval[0]), 1)
    inter = inter.reshape(len(inter), len(inter[0]), 1)
    f_domain = np.array(f_domain)
    f_domain = f_domain.reshape(len(f_domain), len(f_domain[0]), 1)
    #label_class = label_class.reshape(len(label_class))
    PPG_ = PPG_.reshape(len(PPG_), len(PPG_[0]), 1)
    return PPG_, inter, f_domain, label
    




from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    inter = read_inter("./datas/inter-4-150s-upsampled.npy", 5)
    f_domain = np.load("./datas/f_domain-4-150s-upsampled.npy")
    label = read_label("./datas/label-4-150s-upsampled.npy")
    print (inter.shape)
    
    x_train, x_test, y_train, y_test = train_test_split([inter, f_domain], label, test_size = 0.2, shuffle = False)
    print (x_train.shape)
    
