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

from random import randrange

from autoencoders import AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4


#from torch import autograd
#torch.autograd.set_detect_anomaly(True)

import gc
gc.enable()

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
    # time1 = '2018-1-19'
    # time2 = '2018-1-25' # times for the p232 dataset
    # time3 = '2018-12-31'
    # time1 = '2018-2-19'
    # time2 = '2018-4-25'
    # time3 = '2018-12-31'

    p232 = ['2018-1-19' , '2018-1-25' , '2018-12-31'] 
    p369 = ['2018-03-01', '2018-05-01', '2018-12-31']
    # p810 = ['2018-03-01', '2018-05-01', '2018-12-31']
    # p31 =  ['2018-02-01', '2018-04-01', '2018-12-31']
    # #p158 = ['2018-02-01', '2018-04-01', '2018-12-31']
    # p183 = ['2018-02-01', '2018-04-01', '2018-12-31']
    # p257 = ['2018-01-14', '2018-01-25', '2018-12-31']
    # p427 = ['2018-02-01', '2018-04-01', '2018-12-31']
    # p461 = ['2018-01-30', '2018-03-01', '2018-12-31']
    # p538 = ['2018-03-01', '2018-05-01', '2018-12-31']
    # p628 = ['2018-02-01', '2018-04-01', '2018-12-31']
    # p654 = ['2018-02-01', '2018-04-01', '2018-12-31']
    # p673 = ['2018-02-01', '2018-02-01', '2018-12-31']
    # p866 = ['2018-02-01', '2018-04-01', '2018-12-31']

    #possible_datasets = ['p232', 'p369', 'p810','p31','p183','p257','p427','p461','p538','p628','p654','p673','p866'] #remember to add back p158 later
    possible_datasets = ['p232']
    #possible_datasets_times = pd.DataFrame(columns = ['time1','time2', 'time3'], index = possible_datasets, data = ([p232, p369, p810, p31, p183, p257, p427, p461, p538, p628, p654, p673, p866]))
    possible_datasets_times = pd.DataFrame(columns = ['time1','time2', 'time3'], index = possible_datasets, data = [p232])

    sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
                    'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
                    'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

    possible_models = [AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4]

    mse = nn.MSELoss()
    mae = nn.L1Loss() #read somewhere mae works well for anomaly detection i think
    nlll = nn.NLLLoss()
    cel = nn.CrossEntropyLoss()
    kld = nn.KLDivLoss()

    possible_loss_functions = [mse, mae, nlll, cel, kld]
    
    #loss = nn.MSELoss()

    data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';', parse_dates= True, index_col = 0) #for plotting dots


    #initialize the autoencoder (argument is sensor number for first and last AE layer)
    for kk in range(50):

        dataset = possible_datasets[randrange(len(possible_datasets))]
        #dataset = possible_datasets[13]
        #print(dataset)
        dataset_times = possible_datasets_times.loc[dataset]

        time1 = dataset_times['time1']
        time2 = dataset_times['time2']
        time3 = dataset_times['time3']

        file_path = f'data/{dataset}/'

        #may be a lot quicker if load-in all the datasets outside of the for-loop.
        X = pd.read_csv(file_path + 'Levels.csv', index_col=0, parse_dates=[0])
        pressure = pd.read_csv(file_path +'Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
        pump_flows = pd.read_csv(file_path + 'Flows.csv', index_col=0, parse_dates=[0]).squeeze()

        X = X.reset_index()
        pressure = pressure.reset_index()
        X = pd.concat([X, pressure], axis = 1)
        X = X.set_index('index')

        X['PUMP_1'] = pump_flows['PUMP_1']
        X['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']

        #small attempt to reduce seasonality
        #X['Demand'] = X['Demand'].apply(lambda x: np.log(x))
        #X['Demand'] = X['Demand'].rolling(3000, min_periods=1).mean()

        X = X[:time3] 

        #print(f'Before: {X}')
        #X = max_abs_scaling(X)
        X = (X - X.mean())/X.std() 
        #print(f'After: {X}')

        # define training, testing and validation datasets
        X_train = X[:time1]
        X_test = X[time1:time2]
        X_val = X[time2:]

        xtr = torch.FloatTensor(X_train.values)
        xts = torch.FloatTensor(X_test.values)
        xtv = torch.FloatTensor(X_val.values)

        model = possible_models[randrange(len(possible_models))]
        model = model(len(X_train.columns))

        #loss = possible_loss_functions[randrange(len(possible_loss_functions))]
        loss = nn.MSELoss()

        #dataloader for creating batches
        possible_batch_sizes = [4, 8, 16, 24, 32, 40, 48]
        batch_size = possible_batch_sizes[randrange(len(possible_batch_sizes))]
        #batch_size = 8

        xdl = DataLoader(xtr, batch_size=batch_size)
        tdl = DataLoader(xts, batch_size=batch_size)

        possible_epochs = [50, 75, 100, 125, 150, 175, 200, 225]
        epochs = possible_epochs[randrange(len(possible_epochs))]
        #epochs = 10

        possible_learning_rates = [0.01, 0.001, 0.005, 0.0001, 0.00005, 0.1]
        learning_rate = possible_learning_rates[randrange(len(possible_learning_rates))]

        sgd  = torch.optim.SGD(model.parameters(), lr = learning_rate)
        adam = torch.optim.Adam(model.parameters(), lr = learning_rate)
        adamW = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        ndam = torch.optim.NAdam(model.parameters(), lr = learning_rate)
        adamax = torch.optim.Adamax(model.parameters(), lr = learning_rate)

        possible_optimizers = [sgd, adam, adamW, ndam, adamax]
        optimizer = possible_optimizers[randrange(len(possible_optimizers))] #SGD needs a lot of epochs, may be increased learning rate
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

        #plotting average reconstruction loss over the whole dataset
        res = [] 
        xt = torch.FloatTensor(X.values)
        #24_02: if i remember correctly, there was a problem with the batch normalization initially; had to reformat input for it to work. the section below does that with new_res
        for ii in range(0, xt.shape[0], batch_size):
            a = xt[ii:(ii+batch_size)]
            res.append(score(a))

        new_res = []
        i = 0
        for j in range(len(res)*batch_size):
            if j % batch_size==0:
                new_res.append(res[i])
                i = i+1
            else:
                new_res.append(0)

        new_res = np.array(new_res, dtype = float)

        bottom_subplot = fig.add_subplot(gs[2:6, 0:8])
        plt.plot(X.index, new_res[:len(X)], color = colors[1], label = '_nolegend_')

        a, b, c = 0, 0, 0
        for leak in max_leak:
            if leak in burst_and_repaired_leaks:
                if leak == dataset:
                    #if leak == 'p232' or 'p158' or 'p673' or 'p866' or 'p461' or 'p183':
                    if leak == 'p232': 
                        leak_date = max_leak[leak]
                        leak_date = leak_date.strftime('%Y-%m-%d %X')
                        leak_date = pd.to_datetime(leak_date)
                        plt.plot(leak_date, 0, marker = 'o', color = 'blue', label = 'burst & repair', alpha = 0.8)
                    else:
                        leak_date = max_leak[leak]
                        plt.plot(leak_date, 0, marker = 'o', color = 'blue', label = 'burst & repair', alpha = 0.8)

            if leak in slow_increase_leaks:
                if leak == dataset:
                    if leak == 'p31':
                        leak_date = max_leak[leak]
                        leak_date = leak_date.strftime('%Y-%m-%d %X')
                        leak_date = pd.to_datetime(leak_date)
                        plt.plot(leak_date, 0, marker = 'o', color = 'black', label = 'burst & repair', alpha = 0.8)
                    else: 
                        leak_date = max_leak[leak]
                        plt.plot(leak_date, 0, marker = 'o', color = 'black', label = 'slow increase & repair' if b == 0 else "", alpha = 0.8)

            if leak in neverending_leaks:
                if leak == dataset:
                    leak_date = max_leak[leak]
                    #leak_date = leak_date.strftime('%Y-%d-%m %X')
                    #leak_date = pd.to_datetime(leak_date)
                    plt.plot(leak_date, 0, marker = 'o', color = 'red', label = 'neverending' if c == 0 else "",  alpha = 0.8)
                #c = c+1

        param_text = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Dataset = {file_path} \n\n Model: {model}') 
        param_text2 = (f'optimzer: {optimizer},\nloss function : {loss},\nnormalization method: standard \ntraining time: SOY until {time1},\ntest time: between {time1} and {time2},\nvalidation time: {time3} until EOY' )
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)

        plt.text(0.02, 0.70, param_text, bbox= props, fontsize=8, transform=plt.gcf().transFigure)
        plt.text(0.4, 0.72, param_text2, bbox= props, fontsize=8, transform=plt.gcf().transFigure)

        X_train = X[:time1]
        X_test = X[time1:time2]
        X_val = X[time2:]


        plt.title('Reconstruction')
        plt.ylabel('Error')

        plt.legend()
        #plt.show()

        gc.collect()
        folder_path = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\figures_23_02_dataset_test\\"
        fig.savefig(folder_path+str(kk))
        plt.clf()
