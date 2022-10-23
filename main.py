import numpy as np
from LVBF import LV_fd
import matplotlib.pyplot as plt
import MieFitting
from tqdm import tqdm

reader = LV_fd(endian='>', encoding='cp1252')
with open(r'3D_pockets_uint16.bin') as reader.fobj:
    data = reader.read_array(reader.read_numeric, reader.LVuint16, ndims=3)
    data = np.array(data, dtype='float')


ss = MieFitting.SFC_MieFitting()
trace_exp = ss.trace_zero_norm(data[:, 1, ])
trigger_exp = ss.trace_zero_norm(data[:, 3, ])

trace_exp, time_exp = ss.time_cutter(trace_exp, trigger_exp)
trace_exp, time_exp = ss.cluster_filter(trace_exp, time_exp)

solves = []

for i in tqdm(range(len(trace_exp))):
    res = ss.fit(time_exp[i], trace_exp[i])
    solves.append(res[0])
    plt.plot(trace_exp[i])
    plt.plot(res[1])
    plt.show()


solves = np.array(solves)
solves[:, 1] = solves[:, 1] * 1000
stat = [np.mean(solves[:, 1]), np.std(solves[:, 1]), np.mean(solves[:, 0]), np.std(solves[:, 0])]

plt.plot(solves[:, 0], solves[:, 1], '*', label = 'X0_mean = %f, \n X0_std = %f, \n v_mean = %f \n, v_std = %f' %(stat[0], stat[1], stat[2], stat[3]))
plt.ylabel('X0, m')
plt.xlabel('v, m/s')
plt.legend()
plt.show()
