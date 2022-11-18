import numpy as np
from LVBF import LV_fd
import matplotlib.pyplot as plt
import MieFitting
from tqdm import tqdm

def vis(params, x):
    N = len(params)
    plt.figure(figsize=(40, 40))
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, i * N + j + 1)
            plt.plot(x[:,i], x[:,j], '*')
            plt.xlabel(params[i])
            plt.ylabel(params[j])
    plt.show()

def label_hist(x, _labels):
    for i in range(len(np.unique(_labels))):
        plt.hist(np.sum(x[_labels == i], axis=1), bins=100, label='label%d' %(i))
    plt.yscale('log')
    plt.legend()
    plt.show()

def label_plot(x, _labels):
    for i in range(len(np.unique(_labels))):
        plt.plot(x[_labels == i].T)
        plt.xlabel('N = %d' % (len(x[_labels == i])))
        plt.show()
