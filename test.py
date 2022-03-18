
from sqlite3 import Timestamp
from matplotlib import markers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parso import parse

import torch
import torch.nn as nn
import pylab

import math
from random import randrange
from statistics import mean


from itertools import groupby



from autoencoders_and_more import thresholding_algo, ordered_cluster


import wntr

# sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
#                         'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
#                         'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

# X = pd.read_csv('p232/p232/Levels.csv', index_col=0, parse_dates=[0])
# pressure = pd.read_csv('p232/p232/Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)

# pump_flows = pd.read_csv('p232/p232/Flows.csv', index_col=0, parse_dates=[0],  squeeze= True)

# X = X.reset_index()
# pressure = pressure.reset_index()
# X = pd.concat([X, pressure], axis = 1)
# X = X.set_index('index')

# print(type(X.index))

# # X = X.loc[~X.index.duplicated(keep='first')]
# #X = pd.concat([timeindex, pressure], axis = 1)

# #X = X.reset_index()
# #X = X.iloc[:, 2:]

# # y = X.iloc[:, 1:]
# # print(y)

# # a = np.NaN
# # print(a)
# # b = []
# # b = b.append(a)
# # print(b)

# # new_res = []
# # i = 0
# # for j in range(len(res)*batch_size):
# #     if j % batch_size==0:
# #         print(j)
# #         new_res.append(res[i])
# #         i = i+1
# #     else:
# #         print(f'gkagmkagma{j}')
# #         new_res.append(np.NaN)

# # print(f'lengde på new_res: {len(res)}')
# # print(f'new_res er : {res}')

# file_path = 'p232/p232/'
# print(file_path[:4])

# # print(X.shape) #works, but is really slow, figure out how to fix later, too  bored with trying now
# # for col in X.columns:
# #     if col not in sensor_columns:
# #         X = X.drop(col, axis = 1)
# # print(X.shape)

# # #for i in range(0, X.shape[0], batch_size):
# #     #print(X[i])

# data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';', parse_dates= True, index_col = 0)


# max_leak = {}
# for col in data.columns:
#     max_leak[col] = data[col].idxmax()

# print(max_leak)

# for leak in max_leak:
#     if leak == 'p866':
#         a = max_leak[leak]
#         print(type(a))
#         a = a.strftime('%Y-%m-%d %X')
#         a = pd.to_datetime(a)
#         plt.plot(a, 0, marker = 'o', color = 'red', label = 'burst & repair', alpha = 0.8)
#         print(f'red dateis: {a}')    
#     if leak == 'p654':
#         a = max_leak[leak]
#         print(type(a))
#         a = a.strftime('%Y-%m-%d %X')
#         a = pd.to_datetime(a)
#         plt.plot(a, 0, marker = 'o', color = 'green', label = 'burst & repair', alpha = 0.8)
#         print(f'green dateis: {a}')    
#     if leak == 'p232':
#         a = max_leak[leak]
#         plt.plot(a, 0, marker = 'o', color = 'blue', label = 'burst & repair', alpha = 0.8)
#         print(f'blue dateis: {a}')    
# plt.show()
#print(a)


# burst_and_repaired_leaks = ['p158','p183','p232','p369','p461','p538','p628','p673','p866']
# slow_increase_leaks = ['p31','p461']
# neverending_leaks = ['p257', 'p427', 'p654', 'p810']


# from random import randrange

# print(randrange(4))



#define training (<time1), validation (between time1 and time2) and testing period (from time 2 to 3)
    # time1 = '2018-1-19'
    # time2 = '2018-1-25' # times for the p232 dataset
#   # time3 = '2018-12-31'


# p232 = ['2018-1-19' , '2018-1-25', '2018-12-31'] 
# p369 = ['2018-03-01' , '2018-05-01', '2018-12-31']
# p810 = ['2018-03-01', '2018-05-01', '2018-12-31']

# possible_datasets = ['p232', 'p369', 'p810']

# possible_datasets_times = pd.DataFrame(columns = ['time1','time2', 'time3'], index = possible_datasets, data = ([p232, p369, p810]))
# print(possible_datasets_times)

# #dataset = possible_datasets[randrange(len(possible_datasets))]
# dataset = 'p232'
# # print(dataset)
# # dataset_times = possible_datasets_times.loc[dataset]
# # print(dataset_times)
# # time1 = dataset_times['time1']
# # time2 = dataset_times['time2']
# # time3 = dataset_times['time3']

# sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
#                 'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
#                 'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

# file_path = f'data/{dataset}/'

# X = pd.read_csv(file_path + 'Levels.csv', index_col=0, parse_dates=[0])
# pressure = pd.read_csv(file_path +'Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
# pump_flows = pd.read_csv(file_path + 'Flows.csv', index_col=0, parse_dates=[0]).squeeze()

# X = X.reset_index()
# pressure = pressure.reset_index()
# X = pd.concat([X, pressure], axis = 1)
# X = X.set_index('index')

#X = X.drop(['T1'], axis = 1)

#X = pd.concat([X, demands], axis = 1)
#X['p227'] = pump_flows['p227']
#X['p235'] = pump_flows['p235']
# X['PUMP_1'] = pump_flows['PUMP_1']
# X['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']

# X['Demand'] = X['Demand'].rolling(3000, min_periods=1).mean()

# X = X[:time3] 
# print(len(X['Demand']))
# print(f'Before: {X}')

# plt.plot(X['Demand'])

# plt.show()
# #X = max_abs_scaling(X)
# X = (X - X.mean())/X.std() 
# #print(f'After: {X}')

# # define training, testing and validation datasets
# X_train = X[:time1]
# X_test = X[time1:time2]
# # X_val = X[time2:]

# a = X.head(5)
# fig = plt.figure(1, figsize = (14,6))
# ax = plt.subplot(111)


# for i, col in enumerate(a.columns):
#     if i >= 9 and i < 18:   
#         ax.plot(a.index, a[col], label = col, marker = 11)
#     elif i >= 18 and i < 27: 
#         ax.plot(a.index, a[col],label = col, marker = "D")
#     elif i >= 27:
#         ax.plot(a.index, a[col],label = col, marker = "x")
#     else:
#         ax.plot(a.index, a[col], label = col)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# #ax = plt.gca()
# #ax.set_ylim([, a.max()+a.max()*0.1])
# plt.legend(ncol=2, loc = 'best', bbox_to_anchor = (1, 0.8))
# plt.show()

#a = [[5.1, 5.2, 5.4, 5.4453],[4.4,3.4,5.3,2.1]]
#print(np.shape(a))

#print(n)

# test = torch.tensor([5,5], dtype = float)
# more_test = torch.tensor([[2,2],[2,2]], dtype =float)

# c = test+more_test
# print(c)

# string = "autoencoder_test_figures_1"

# for s in string.split('_'):
#     if s.isdigit():
#         number = int(s)

# # print(number)'

# print(X)

# for col in X.columns:
#     X[col].values[:] = 0

# print(X)
# #vilkårlig output tensor fra y_pred = model(V(x))
# b = torch.tensor([[ 5.8967e-01,  1.0000e+00, -9.9999e-01,  9.9897e-01,  3.8110e-01,
#           1.0000e+00,  2.9572e-02,  9.9956e-01,  1.6019e-01,  1.0000e+00,
#           1.2554e-01,  1.0000e+00, -3.6826e-02,  5.6512e-02,  2.1804e-01,
#           9.8146e-01,  1.0000e+00,  1.7352e-01,  7.2970e-02,  9.9989e-01,
#           2.0437e-04,  4.1310e-02,  3.0938e-02,  2.7296e-01, -1.4981e-02,
#           1.5442e-01, -2.9899e-02, -4.0141e-02,  9.7974e-03, -4.7208e-02,
#          -2.3310e-01, -1.1833e-01, -1.7525e-01,  1.0000e+00, -1.0000e+00,
#           4.3980e-01],
#         [ 7.7119e-01,  1.0000e+00, -9.9999e-01,  9.9940e-01,  5.2673e-01,
#           1.0000e+00,  1.0344e-01,  9.9957e-01,  4.1982e-01,  1.0000e+00,
#           5.2833e-01,  1.0000e+00,  2.7974e-01,  1.5930e-01,  5.9010e-01,
#           9.8615e-01,  1.0000e+00,  8.7885e-02,  1.5018e-01,  9.9990e-01,
#           3.2657e-01,  2.0126e-01,  2.3468e-01,  5.1424e-01,  1.4865e-01,
#           4.8160e-01, -5.7452e-02, -3.2103e-02,  3.5377e-01,  3.9541e-01,
#          -6.3482e-02,  1.3888e-01, -1.4520e-01,  1.0000e+00, -1.0000e+00,
#           1.9613e-01],
#         [ 6.6279e-01,  1.0000e+00, -9.9999e-01,  9.9920e-01,  4.6409e-01,
#           1.0000e+00,  5.8161e-02,  9.9958e-01,  2.5426e-01,  1.0000e+00,
#           3.1079e-01,  1.0000e+00,  1.2442e-01,  7.6438e-02,  4.0538e-01,
#           9.8411e-01,  1.0000e+00,  1.4547e-01,  9.5630e-02,  9.9990e-01,
#           1.4862e-01,  8.0043e-02,  1.1165e-01,  4.1319e-01,  4.6202e-02,
#           3.1425e-01, -6.4309e-02, -5.7565e-02,  1.7431e-01,  1.6171e-01,
#          -2.0102e-01,  1.5477e-02, -1.6971e-01,  1.0000e+00, -1.0000e+00,
#           3.4052e-01],
#         [ 8.3626e-01,  1.0000e+00, -9.9999e-01,  9.9946e-01,  5.6185e-01,
#           1.0000e+00,  2.0655e-01,  9.9957e-01,  6.1200e-01,  1.0000e+00,
#           6.4900e-01,  1.0000e+00,  3.6393e-01,  3.0671e-01,  6.9674e-01,
#           9.8787e-01,  1.0000e+00,  7.3283e-02,  2.3361e-01,  9.9990e-01,
#           4.3984e-01,  3.5169e-01,  3.2133e-01,  5.6631e-01,  2.9225e-01,
#           5.6758e-01, -8.5227e-03,  5.6792e-02,  4.3771e-01,  5.5198e-01,
#           1.1606e-01,  2.2041e-01, -7.2658e-02,  1.0000e+00, -1.0000e+00,
#           9.8850e-02],
#         [ 7.6263e-01,  1.0000e+00, -9.9999e-01,  9.9939e-01,  5.2254e-01,
#           1.0000e+00,  9.6566e-02,  9.9958e-01,  4.0125e-01,  1.0000e+00,
#           5.1274e-01,  1.0000e+00,  2.6904e-01,  1.4731e-01,  5.7693e-01,
#           9.8600e-01,  1.0000e+00,  9.2620e-02,  1.4329e-01,  9.9990e-01,
#           3.1240e-01,  1.8706e-01,  2.2386e-01,  5.0859e-01,  1.3634e-01,
#           4.7026e-01, -6.0631e-02, -3.8280e-02,  3.4209e-01,  3.7744e-01,
#          -8.0927e-02,  1.2989e-01, -1.4965e-01,  1.0000e+00, -1.0000e+00,
#           2.0773e-01],
#         [ 7.9011e-01,  1.0000e+00, -9.9999e-01,  9.9942e-01,  5.3615e-01,
#           1.0000e+00,  1.2243e-01,  9.9957e-01,  4.6573e-01,  1.0000e+00,
#           5.6210e-01,  1.0000e+00,  3.0291e-01,  1.9083e-01,  6.1920e-01,
#           9.8651e-01,  1.0000e+00,  7.7699e-02,  1.6809e-01,  9.9990e-01,
#           3.5866e-01,  2.3650e-01,  2.5980e-01,  5.2602e-01,  1.8034e-01,
#           5.0590e-01, -4.8369e-02, -1.4538e-02,  3.7815e-01,  4.3595e-01,
#          -1.9058e-02,  1.5951e-01, -1.3265e-01,  1.0000e+00, -1.0000e+00,
#           1.7012e-01],
#         [ 7.7820e-01,  1.0000e+00, -9.9999e-01,  9.9941e-01,  5.3013e-01,
#           1.0000e+00,  1.0974e-01,  9.9957e-01,  4.3591e-01,  1.0000e+00,
#           5.4089e-01,  1.0000e+00,  2.8834e-01,  1.7007e-01,  6.0078e-01,
#           9.8628e-01,  1.0000e+00,  8.4001e-02,  1.5629e-01,  9.9990e-01,
#           3.3821e-01,  2.1354e-01,  2.4368e-01,  5.1858e-01,  1.5953e-01,
#           4.9063e-01, -5.4439e-02, -2.6252e-02,  3.6294e-01,  4.1014e-01,
#          -4.8065e-02,  1.4625e-01, -1.4110e-01,  1.0000e+00, -1.0000e+00,
#           1.8661e-01],
#         [ 8.0307e-01,  1.0000e+00, -9.9999e-01,  9.9944e-01,  5.4251e-01,
#           1.0000e+00,  1.4106e-01,  9.9957e-01,  5.0256e-01,  1.0000e+00,
#           5.8653e-01,  1.0000e+00,  3.1963e-01,  2.1800e-01,  6.4010e-01,
#           9.8685e-01,  1.0000e+00,  7.4763e-02,  1.8288e-01,  9.9990e-01,
#           3.7979e-01,  2.6418e-01,  2.7514e-01,  5.3642e-01,  2.0628e-01,
#           5.2291e-01, -3.9592e-02,  1.2641e-03,  3.9527e-01,  4.6643e-01,
#           1.2919e-02,  1.7418e-01, -1.1988e-01,  1.0000e+00, -1.0000e+00,
#           1.5142e-01]])

# c = b + 1
# print(c)

# print(b.shape)
# t1 = (b[:,0])
# print(t1)

# avg_t1 = torch.mean(b[:,0])
# print(avg_t1)


# X_head = X.head(3)
# # for ind in X_head.index:
# #     for i, col in enumerate(X_head):
# #         X_head[col] = torch.mean(b[:,i])

# print(X_head)

# hh = torch.Tensor([ 0.4016,  0.6108,  0.5016,  0.5302, -0.1252,  0.4534,  0.3090,  0.3427,
#          0.1692,  0.5987,  0.3155,  0.4639,  0.3343,  0.3187, -0.1799, -0.0129,
#          0.2989,  0.1255,  0.1110,  0.3878,  0.2961,  0.3300,  0.2048,  0.4412,
#          0.3381,  0.0138,  0.2756,  0.6472,  0.2545,  1.0000,  0.2978,  0.4092,
#          1.0000,  0.2671,  0.6143, -1.0000])

# bb = hh +1 

# for jj, ind in enumerate(X_head.index):
#     for ii, col in enumerate(X_head.columns):
#       #X_head[col][0] = hh[ii]
#       X_head[col][jj] = bb[ii]

# print(X_head)



# import sys
# import pickle


# test = pickle.load(sys.stdin)
# print(test)

# a = [40, 80, 120, 160, 200]
# j = 0 
# for h in range(5):
#   print(a[j]+h)
  
  
#   print(a[j+1]+h)


#   y = X.index[:103000]
#   print(y)

per_neuron = pd.read_csv('per_neuron_all_sensors.csv', parse_dates=True)
random_data = pd.read_csv('test-res_test_v5.csv', parse_dates = True)

print(per_neuron)
# print(per_neuron)
# res = per_neuron.drop(columns=['index'])

# #res = pd.read_csv("test-res_test_v3.csv")

# n = 2000
# #rol = res.rolling(n).mean()
# #res = np.array(res)

# print(res)

# # plt.plot(res)
# # plt.show()
# results = pd.DataFrame(index = [':-)'], columns = ['n', 'idx', 'a', 'Recall', 'F1_score', 'sensor','True positive', 'False positive'], data = 0)

# for kk, col in enumerate(res.columns):
#     lag = n
#     threshold = 2.75 #3.5 for two 1's
#     influence = 0.80

#     test = thresholding_algo(res[col], lag = lag, threshold= threshold, influence= influence)

#     # pylab.subplot(211)
#     # pylab.plot(np.arange(1, len(res)+1), res)
#     # pylab.plot(np.arange(1, len(res)+1),
#     #            test["avgFilter"], color="cyan", lw=2)
#     # pylab.plot(np.arange(1, len(res)+1),
#     #            test["avgFilter"] + threshold * test["stdFilter"], color="green", lw=2)
#     # pylab.plot(np.arange(1, len(res)+1),
#     #            test["avgFilter"] - threshold * test["stdFilter"], color="green", lw=2)
#     # pylab.subplot(212)
#     # pylab.step(np.arange(1, len(res)+1), test["signals"], color="red", lw=2)
#     # pylab.ylim(-1.5, 1.5)

#     #pylab.show()
#     #results
#     results_holder = pd.DataFrame(index = [':-)'], columns = ['n', 'idx', 'a', 'Recall', 'F1_score', 'sensor','True positive', 'False positive'], data = 0)

#     idx = np.where(test["signals"] == 1)
#     number = idx[0][:]

#     #print(f'number is this: {number}')
#     abs_tol = 288

#     a = list(ordered_cluster(number, abs_tol*4))
#     container = []
#     for aa in a:
#         container.append(mean(aa))

#     b = list(ordered_cluster(container, abs_tol))
#     containerv2 = []
#     for bb in b:
#         containerv2.append(mean(bb))

#     #based on when max value of leak occurs in 2018_leakages.csv-file
#     #transferred from random_plotting.py
#     #true_leak_idx_burst_and_repair = {'p158': 80688, 'p183': 64560, 'p232': 10134,'p369': 86448,'p461': 25106, 'p538': 42963,'p628': 39225,'p673': 18783, 'p866': 44125}
#     true_leak_idx_burst_and_repair = {'p232': 10134}
#     #p232 er egt 10134, skfitet til 8000 for testeformål, 
#     #problem: max_idx from earlier, do not match what the evaluation algorithm detects very closely, need to find instance of when it first happens. 
#     #values above might need to be adjusted somewhat.
#     #max peak occurs too late in the leak's lifespan, so find other metric. i.e. first time reaches 80% of max peak or something like that

#     #results.loc['test 1'] = 0
#     holderv2 = []
#     for leak_nr in true_leak_idx_burst_and_repair:
#         holderv2.append(true_leak_idx_burst_and_repair[leak_nr])

#     for c in containerv2:
#         a = any(math.isclose(c, item, abs_tol=abs_tol*2) for item in holderv2)
#         #print(c, a)
#         if a: #seems to be a bit much, 1 step is 5 min here, 288 per day    
#             current_value = results_holder['True positive'].iloc[-1]
#             results_holder.loc[results_holder.index[-1], 'True positive'] = current_value + 1
#             results_holder.loc[results_holder.index[-1], 'idx'] = c
#             results_holder.loc[results_holder.index[-1], 'a'] = a
#         else:
#             current_value = results_holder['False positive'].iloc[-1]
#             results_holder.loc[results_holder.index[-1], 'False positive'] = current_value + 1 
#             results_holder.loc[results_holder.index[-1], 'idx'] = c
#             results_holder.loc[results_holder.index[-1], 'a'] = a

#     results_holder['n'] = kk
#     results_holder['sensor'] = col

#     results = pd.concat([results, results_holder])

# print(results)
wn = wntr.network.WaterNetworkModel('temp.inp') 
ax = wntr.graphics.plot_network(wn, node_attribute='elevation', node_colorbar_label='Elevation (m)')

nodes = wn.nodes()

y = wn.get_node('n216')
#print(type(y))

#only resulted in zeros, dunno why, fixed later with get_coordinates()
# for col in per_neuron.columns:
#     print(wntr.network.Junction(col, wn).coordinates)

#from statsmodels.tsa.seasonal import STL

# stl = STL(random_data, period=7, robust = True)
# res = stl.fit()
# seasonal = res.seasonal
# print(f'this is seasonal: {seasonal}')
# trend = res.trend


# fig = res.plot()

# plt.show()
# plt.clf()

# n458 = per_neuron['n458']

# n458_stl = STL(n458, period = 288, robust = True)
# n458_res = n458_stl.fit() #denne tar utrolig lang tid, lage til numpy array? problem at er dataframe?
# n458_seasonal = n458_res.seasonal
# n458 = n458 - n458_seasonal
# fig = n458.rolling(288*5).mean().plot()
# plt.show()

def get_coordinates():
    nodes = []
    x_coord = []
    y_coord = []
    with open('coordinates.txt') as f:
        line = f.readline()
        while line:
            line = f.readline()
            a = line.split()
            if len(a)> 0: 
                nodes.append(a[0])
                x_coord.append((float(a[1])))
                y_coord.append((float(a[2])))

    f.close()

    coordinates = pd.DataFrame()
    coordinates['nodes'] = nodes
    coordinates['x'] = x_coord
    coordinates['y'] = y_coord
    #print(coordinates)
    return coordinates


a = get_coordinates()
b = a

b = b['y'].rolling(7, min_periods = 1).mean()

print('\n\n\n',a)

print(b)