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

#from torch import autograd
#torch.autograd.set_detect_anomaly(True)

import gc
gc.enable()

class AutoEncoder(nn.Module):
    # 17_02 https://prnt.sc/26xqcfi
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


def train(epochs, model, model_loss):
    try:
        c = model_loss.epoch[-1]
    except:
        c = 0
    #for epoch in tqdm(range(epochs), position=0, total=epochs):
    for epoch in range(epochs):
        losses = []
        dl = iter(xdl)
        for t in range(len(dl)):

            # Forward pass: compute predicted y and loss by passing x to the model.
            xt = next(dl)
            y_pred = model(V(xt))
            l = loss(y_pred, V(xt)) 
            losses.append(l)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            l.backward()
            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        val_dl = iter(tdl)
        val_scores = [score(next(val_dl)) for i in range(len(val_dl))]

        model_loss.epoch.append(c + epoch)
        model_loss.loss.append(l.item())
        model_loss.val_loss.append(np.mean(val_scores))

        print(f'Epoch: {epoch}   Loss: {l.item():.5f}    Val_Loss: {np.mean(val_scores):.5f}')

def score(x):
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred, x1).item()


def max_abs_scaling(x):
    x_new = x.copy()
    for column in x.columns:
        min_x = x[column].min()
        max_x = x[column].max()
        if (min_x==max_x):
            x_new[column] = x[column]
        else:
            if column == 'p227':
                print(x)
                print(f'abs.max.column is this value: {x[column].abs().max()}')
 
            x_new[column] = x[column]/x[column].abs().max()
    return x_new    

if __name__ == '__main__':

    #define training (<time1), validation (between time1 and time2) and testing period (from time 2 to 3)
    time1 = '2018-1-19'
    time2 = '2018-1-25'
    time3 = '2018-12-31'

    sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
                    'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
                    'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

    X = pd.read_csv('data/p232/Levels.csv', index_col=0, parse_dates=[0])
    pressure = pd.read_csv('data/p232/Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
    pump_flows = pd.read_csv('data/p232/Flows.csv', index_col=0, parse_dates=[0]).squeeze()

    X = X.reset_index()
    pressure = pressure.reset_index()
    X = pd.concat([X, pressure], axis = 1)
    X = X.set_index('index')

    #X = X.drop(['T1'], axis = 1)
    
    #X = pd.concat([X, demands], axis = 1)
    #X['p227'] = pump_flows['p227']
    #X['p235'] = pump_flows['p235']
    X['PUMP_1'] = pump_flows['PUMP_1']
    X['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']
    
    X = X[:time3] 

    print(f'Before: {X}')
    #X = max_abs_scaling(X)
    X = (X - X.mean())/X.std() 
    print(f'After: {X}')

    # define training, testing and validation datasets
    X_train = X[:time1]
    X_test = X[time1:time2]
    X_val = X[time2:]

    #transform dataframes to tensors
    xtr = torch.FloatTensor(X_train.values)
    xts = torch.FloatTensor(X_test.values)
    xtv = torch.FloatTensor(X_val.values)

    #dataloader for creating batches 17_02
    batch_size = 32
    xdl = DataLoader(xtr, batch_size=batch_size)
    tdl = DataLoader(xts, batch_size=batch_size)

    #initialize the autoencoder (argument is sensor number for first and last AE layer)
    model = AutoEncoder(len(X_train.columns))
    loss = nn.MSELoss()

    epochs = 200 #normally use 200

    learning_rate = 1e-3
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #SGD needs a lot of epochs, may be increased learning rate
    print(model)

    #Utilize a named tuple to keep track of scores at each epoch
    model_hist = collections.namedtuple('Model', 'epoch loss val_loss')
    model_loss = model_hist(epoch=[], loss=[], val_loss=[])

    #training the AE
    train(model=model, epochs=epochs, model_loss=model_loss)

    #Plotting losses over epocs in the training phase
    x = np.linspace(0, epochs - 1, epochs)

    # print(model_loss.loss)
    # print(model_loss.val_loss)
    # print(x)

    #IMPORTING LEAK-DATA TO BE PLOTTED
    data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';', parse_dates= True, index_col = 0)
    max_leak = {}
    for col in data.columns:
        max_leak[col] = data[col].idxmax()

    burst_and_repaired_leaks = ['p158','p183','p232','p369','p461','p538','p628','p673','p866']
    slow_increase_leaks = ['p31','p461']
    neverending_leaks = ['p257', 'p427', 'p654', 'p810']

    #PLOTTING
    fig = plt.figure(1, figsize = (14,6))
    mpl.rcParams['font.size']=14
    gs = gridspec.GridSpec(6,8)
    gs.update(wspace = 0.25,  hspace = 1.6)
    colors = sns.color_palette("rocket", 6)
    sns.set_style('darkgrid')

    tl_subsplot = fig.add_subplot(gs[0:2, 0:4])
    plt.plot(x, model_loss.loss, color = colors[1])
    plt.ylabel(r'Loss')
    plt.xlabel(r'Epochs')
    plt.title('Model loss')

    tr_subsplot = fig.add_subplot(gs[0:2, 4:8])
    plt.plot(x, model_loss.val_loss, color = colors[1])
    plt.tick_params(length = 5, bottom = True, left = False, right = False, top = False)
    plt.title('Validation loss')
    plt.xlabel(r'Epochs')

    #plotting average reconstruction loss over the whole dataset
    res = [] 
    xt = torch.FloatTensor(X.values)

    for ii in range(0, xt.shape[0], batch_size):
        a = xt[ii:(ii+batch_size)]
        #if ii == 0:
        #    print(a)
        #    print(a.shape)
        res.append(score(a))
        
    #print(f'lengde på res: {len(res)}')
    #print(f'res er : {res}')

    new_res = []
    i = 0
    for j in range(len(res)*batch_size):
        if j % batch_size==0:
            new_res.append(res[i])
            i = i+1
        else:
            new_res.append(0)

    new_res = np.array(new_res, dtype = float)

    #an_array = np.where(an_array > 20, 0, an_array)
    #new_res = np.where(new_res == 0, np.nan, new_res) #alt forsvinner, forstår ikke helt hvorfor    
    # print(f'lengde på new_res: {len(new_res)}')
    # print(f'new_res er : {new_res}')
    
    bottom_subplot = fig.add_subplot(gs[2:6, 0:8])
    plt.plot(X.index, new_res[:len(X)], color = colors[1], label = '_nolegend_')
    
    a, b, c = 0, 0, 0
    for leak in max_leak:
        if leak in burst_and_repaired_leaks:
            plt.plot(max_leak[leak], 0, marker = 'o', color = 'blue', label = 'burst & repair' if a == 0 else "", alpha = 0.8)
            a = a+1
        if leak in slow_increase_leaks:
            plt.plot(max_leak[leak], 0, marker = 'o', color = 'black', label = 'slow increase & repair' if b == 0 else "", alpha = 0.8)
            b = b+1
        if leak in neverending_leaks:
            plt.plot(max_leak[leak], 0, marker = 'o', color = 'red', label = 'neverending' if c == 0 else "",  alpha = 0.8)
            c = c+1

    plt.title('Reconstruction')
    plt.ylabel('Error')

    # including the times, based on p232
    #plt.axvline(pd.to_datetime(time1), color='k', linestyle='--')
    #plt.axvline(pd.to_datetime(time2), color='k', linestyle='--')
    plt.legend()
    plt.show()

    gc.collect()