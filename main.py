import numpy as np
from LVBF import LV_fd
import matplotlib.pyplot as plt
import MieFitting
from tqdm import tqdm
import visualisator

reader = LV_fd(endian='>', encoding='cp1252')
#with open(r'/home/laser/20220906/l3.7_laser on_Vel 30/3D_pockets_uint16.bin') as reader.fobj:
#with open(r'/media/laser/Ubuntu 20_04_4 LTS amd641/20220908/pl1/3D_pockets_uint16.bin') as reader.fobj:

with open(r'C:\Users\Kamov\Desktop\20220906/l3.7_laser on_Vel 30\3D_pockets_uint16.bin') as reader.fobj:
    data = reader.read_array(reader.read_numeric, reader.LVuint16, ndims=3)
    data = np.array(data, dtype='float')


ss = MieFitting.SFC_MieFitting()
ss.tf_hf_init()
trace_exp = ss.trace_zero_norm(data[:, 1, ])
trigger_exp = ss.trace_zero_norm(data[:, 3, ])

#
# plt.plot(trace_exp[2:200].T)
# plt.show()



trace_exp, time_exp = ss.time_cutter(trace_exp, trigger_exp)
#trace_exp, time_exp, labels = ss.cluster_int_filter(trace_exp, time_exp)
trace_exp, time_exp, labels = ss.cluster_cm_filter(trace_exp, time_exp)

#visualisator.label_hist(trace_exp, labels)
visualisator.label_plot(trace_exp, labels)

trace_exp, time_exp = trace_exp[labels == 1], time_exp[labels == 1]
trace_exp, time_exp, labels = ss.cluster_cm_filter(trace_exp, time_exp)
visualisator.label_plot(trace_exp, labels)

solve = ss.fit_serial(time_exp, trace_exp)

visualisator.vis(['d', 'n', 'v', 'l0','I0'], solve)


