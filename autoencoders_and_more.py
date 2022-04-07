import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import collections

import torch
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

from statistics import mean


#original model from 22_02:


class AutoEncoder_1(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        #self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        #x = torch.sigmoid(self.lin2(x))
        #x = torch.tanh(self.drop2(self.lin2(x)))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))
        #x = torch.tanh(self.lin8(x))
        #x = torch.sigmoid(self.drop2(self.lin8(x)))
        return x

        #x = torch.tanh(self.lin2(x))
        #x = self.drop2(torch.tanh(self.lin2(x)))
        #print(x.shape)
        #x = self.lin3_bn(x)
        #x = torch.tanh(self.lin3(x))
        #x = self.drop2(torch.tanh(self.lin3(x)))
        #x = self.drop2(torch.tanh(self.lin4(x)))
        #x = torch.tanh(self.lin6(x))
        #x = torch.tanh(self.lin7(x))
        #x = self.lin8(x)
        #return x


class AutoEncoder_2(nn.Module): #similar to one above, slight adjustment to neurons per layer
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length,20)
        self.lin2_bn = nn.BatchNorm1d(20)
        self.lin2 = nn.Linear(20, 8)
        self.lin3_bn = nn.BatchNorm1d(8)
        self.lin3 = nn.Linear(8, 4)

        self.lin6 = nn.Linear(4, 8)
        self.lin7_bn = nn.BatchNorm1d(8)
        self.lin7 = nn.Linear(8, 20)
        self.lin8_bn = nn.BatchNorm1d(20)
        self.lin8 = nn.Linear(20, length)


        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))

        return x


class AutoEncoder_3(nn.Module): #at encoder: batch norm -> drop at inner layer -> decoder: drop inner layer, batch norm outer
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        #self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        #self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = self.drop2(torch.tanh(self.lin3(x)))
        #x = torch.tanh(self.lin3(x))

        #x = torch.tanh(self.lin6(x))
        x = self.drop2(torch.tanh(self.lin6(x)))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        #x = torch.tanh(self.lin8(self.lin8_bn(x)))
        x = torch.tanh(self.lin8(x))

        return x


class AutoEncoder_4(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        #self.lin2_bn = nn.BatchNorm1d(24)
        self.lin2 = nn.Linear(24, 12)
        self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 24)
        self.lin8_bn = nn.BatchNorm1d(24)
        self.lin8 = nn.Linear(24, length)

        self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):
        x = torch.tanh(self.lin1(data))
        x = self.drop2(torch.tanh(self.lin2(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))

        x = torch.tanh(self.lin6(x)) #mulig til helvete pga denne? problem med dropout -> videre batch_norm og utvidelse, tror dropout bare ok på vei inn?
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))


        return x

#Autoencoder_5 -> no batch normalization for 1 to 1 output. 

class AutoEncoder_5(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 24)
        self.lin2 = nn.Linear(24, 12)
        self.lin3 = nn.Linear(12, 6)

        self.lin6 = nn.Linear(6, 12)
        self.lin7 = nn.Linear(12, 24)
        self.lin8 = nn.Linear(24, length)

        #self.drop2 = nn.Dropout(0.2)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)

        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))

        x = torch.tanh(self.lin6(x))
        x = torch.tanh(self.lin7(x))
        x = torch.tanh(self.lin8(x))

        return x


class AutoEncoder_6(nn.Module): #similar to AE_2, but added another layer
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length, 26)
        self.lin2_bn = nn.BatchNorm1d(26)
        self.lin2 = nn.Linear(26, 12)
        self.lin3_bn = nn.BatchNorm1d(12)
        self.lin3 = nn.Linear(12, 8)
        self.lin4_bn = nn.BatchNorm1d(8)
        self.lin4 = nn.Linear(8, 4)


        self.lin5 = nn.Linear(4, 8)
        self.lin6_bn = nn.BatchNorm1d(8)
        self.lin6 = nn.Linear(8, 12)
        self.lin7_bn = nn.BatchNorm1d(12)
        self.lin7 = nn.Linear(12, 26)
        self.lin8_bn = nn.BatchNorm1d(26)
        self.lin8 = nn.Linear(26, length)

        self.lin1.weight.data.uniform_(-2, 2)
        self.lin2.weight.data.uniform_(-2, 2)
        self.lin3.weight.data.uniform_(-2, 2)
        self.lin4.weight.data.uniform_(-2, 2)

        self.lin5.weight.data.uniform_(-2, 2)
        self.lin6.weight.data.uniform_(-2, 2)
        self.lin7.weight.data.uniform_(-2, 2)
        self.lin8.weight.data.uniform_(-2, 2)

    def forward(self, data):

        x = torch.tanh(self.lin1(data))
        x = torch.tanh(self.lin2(self.lin2_bn(x)))
        x = torch.tanh(self.lin3(self.lin3_bn(x)))
        x = torch.tanh(self.lin4(self.lin4_bn(x)))

        x = torch.tanh(self.lin5(x))
        x = torch.tanh(self.lin6(self.lin6_bn(x)))
        x = torch.tanh(self.lin7(self.lin7_bn(x)))
        x = torch.tanh(self.lin8(self.lin8_bn(x)))

        return x


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


def ordered_cluster(data, max_diff):
    current_group = ()
    for item in data:
        test_group = current_group + (item, )
        test_group_mean = mean(test_group)
        if all((abs(test_group_mean - test_item) < max_diff for test_item in test_group)):
            current_group = test_group
        else:
            yield current_group
            current_group = (item, )
    if current_group:
        yield current_group

def get_jenks_breaks(data_list, number_class):
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass


def get_timestep(x, time_df):
    return time_df.loc[x]