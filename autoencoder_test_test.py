import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import collections
import os
import os.path
from os import path
from sklearn import cluster

import torch
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

from random import randrange
from statistics import mean
import math
from statsmodels.tsa.seasonal import STL

from autoencoders_and_more import AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4, AutoEncoder_5
from autoencoders_and_more import thresholding_algo, ordered_cluster

import pylab

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
        if epoch%50==0 or epoch == epochs-1:
            print(f'Epoch: {epoch}   Loss: {l.item():.5f}    Val_Loss: {np.mean(val_scores):.5f}') 

def score(x):
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred, x1).item()

def score_v2(xt, df):
    j = 0
    for ii in range(0, xt.shape[0], batch_size):
        aa = (xt[ii:(ii+batch_size)])
        y_pred = model(V(aa))
        for i, col in enumerate(df):
            df[col][j] = (torch.mean(y_pred[:,i]))
            #df.at[col, j] = torch.mean(y_pred[:,i])
            #df.loc[col][j] = torch.mean(y_pred[:,i])
        j = j+1
    return df

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
    # p158 = ['2018-02-01', '2018-04-01', '2018-12-31']
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

    #note to self: this is not all the sensors for whatever reason, missing the ones surrouding p232 leak
    sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
                    'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
                    'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

    possible_models = [AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4]

    mse = nn.MSELoss()
    mae = nn.L1Loss() #read somewhere mae works well for anomaly detection i think
    #nlll = nn.NLLLoss()
    #cel = nn.CrossEntropyLoss()
    #kld = nn.KLDivLoss()

    possible_loss_functions = [mse, mae] #cel, kld]
    
    #loss = nn.MSELoss()

    data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';', parse_dates= True, index_col = 0) #for plotting dots
    
    directory_reconstruction_figures = "reconstruction_plots\\autoencoder_test_figures_1"
    directory_evaluation_plots = "evaluation_plots\\evaluation_plots_test_1"
    parent_directory = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\results\\"

    path_reconstruction_figures = os.path.join(parent_directory, directory_reconstruction_figures)

    while os.path.isdir(path_reconstruction_figures):
        old_directory = directory_reconstruction_figures
        if os.path.isdir(os.path.join(parent_directory, directory_reconstruction_figures)) is False:
            break
        
        for s in directory_reconstruction_figures.split('_'):
            if s.isdigit():
                number = int(s)

        directory_reconstruction_figures = f"reconstruction_plots\\autoencoder_test_figures_{str(int(number)+1)}"
        directory_evaluation_plots = f"evaluation_plots\\evaluation_plots_test_{str(int(number+1))}"
        directory_result_files = f"model_performance_csv_files\\model_performance_test_{str(int(number+1))}"
        directory_model_config_files = f"model_configuration_files\\model_config_test_{str(int(number+1))}"
        directory_per_neuron_csv = f"per_neuron_all_sensors_csv\\all_sensors_csv_files_{str(int(number+1))}"

        path_reconstruction_figures = os.path.join(parent_directory, directory_reconstruction_figures)
        path_evaluation_plots = os.path.join(parent_directory, directory_evaluation_plots)
        path_result_files = os.path.join(parent_directory, directory_result_files)
        path_model_config_files = os.path.join(parent_directory, directory_model_config_files)
        path_all_neurons_csv_files = os.path.join(parent_directory, directory_per_neuron_csv)

    os.mkdir(path_reconstruction_figures)
    os.mkdir(path_evaluation_plots)
    os.mkdir(path_result_files)
    os.mkdir(path_model_config_files)
    os.mkdir(path_all_neurons_csv_files)

    results = pd.DataFrame(index = [':)'], columns = ['n', 'Accuracy', 'Precision', 'Recall', 'F1_score','True positive', 'False positive'], data = 0)

    true_leak_idx = {'p31': 62540, 'p158': 80688, 'p183': 64560, 'p232': 10134, 'p257': 7229, 'p369': 86448, 'p427': 42962, 'p461': 25106, 'p538': 42963, 'p628': 39225, 'p654': 74634, 'p673': 18783, 'p810': 101424, 'p866': 44125}
    #true_leak_idx_burst_and_repair = {'p158': 80688, 'p183': 64560, 'p232': 8000,'p369': 86448,'p461': 25106, 'p538': 42963,'p628': 39225,'p673': 18783, 'p866': 44125} #p232 to 8000 due to other numbers are peak of leak,a bit too late time-wise for algorithm to detect
    #true_leak_idx_burst_and_repair = {'p232':8600}
    #'p461','p538','p628','p673','p866'

    true_leak_idx_burst_and_repair = {'p461': 23106,'p538': 40963,'p628': 37225,'p673': 17783,'p866': 42125, 'p232': 8134}

    holderv2 = []
    for leak_nr in true_leak_idx_burst_and_repair:
        holderv2.append(true_leak_idx_burst_and_repair[leak_nr])

    
    for kk in range(200):

        dataset = possible_datasets[randrange(len(possible_datasets))]
        #dataset = possible_datasets[13]
        #print(dataset)
        dataset_times = possible_datasets_times.loc[dataset]

        time1 = dataset_times['time1']
        time2 = dataset_times['time2']
        time3 = dataset_times['time3']

        file_path = f'data/{dataset}/'

        alternative_file_path = f'custom_data/halfabruptleaksv2.csv'
        X = pd.read_csv(alternative_file_path, index_col=0, parse_dates=[0])

        #X.rolling(288*7, min_periods=1).mean()

        # pressure = pd.read_csv(alternative_file_path, index_col=0, parse_dates=[0], usecols=sensor_columns)
        # pump_flows = pd.read_csv(alternative_file_path, index_col=0, parse_dates=[0]).squeeze()

        # print(np.shape(X))
        # print(np.shape(pressure))
        # print(np.shape(pump_flows))


        # X = pd.read_csv(file_path + 'Levels.csv', index_col=0, parse_dates=[0])
        # pressure = pd.read_csv(file_path +'Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
        # pump_flows = pd.read_csv(file_path + 'Flows.csv', index_col=0, parse_dates=[0]).squeeze()

        # X = X.reset_index()
        # pressure = pressure.reset_index()
        # X = pd.concat([X, pressure], axis = 1)
        # X = X.set_index('index')

        #X['PUMP_1'] = pump_flows['PUMP_1']
        #X['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']

        #small attempt to reduce seasonality
        #X['Demand'] = X['Demand'].apply(lambda x: np.log(x))
        #X['Demand'] = X['Demand'].rolling(3000, min_periods=1).mean()

        #removing seasonality from all the columns, super slow <-- can do this outside of for-loop and save to .csv to just read csv later
        #assuming using the same dataset all the time nowadays
        #also add rolling mean at
        # for col in X.columns:
        #     a = STL(X[col], period = 288, robust= True)
        #     a_res = a.fit()
        #     a_seasonal = a_res.seasonal
        #     X[col] = X[col] - a_seasonal

        X = X[:time3] 

        X = (X - X[:time1].mean())/X[:time1].std() 

        # define training, testing and validation datasets
        X_train = X[:time1]
        X_test = X[time1:time2]
        X_val = X[time2:]

        xtr = torch.FloatTensor(X_train.values)
        xts = torch.FloatTensor(X_test.values)
        xtv = torch.FloatTensor(X_val.values)

        #model = possible_models[randrange(len(possible_models))]
        model = AutoEncoder_2(len(X_train.columns))
        #model = model(len(X_train.columns))

        #loss = possible_loss_functions[randrange(len(possible_loss_functions))]
        #loss = mse
        loss = mae

        possible_batch_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500]
        #batch_size = possible_batch_sizes[randrange(len(possible_batch_sizes))]
        batch_size = 1500

        xdl = DataLoader(xtr, batch_size=batch_size)
        tdl = DataLoader(xts, batch_size=batch_size)

        possible_epochs = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 750]
        #epochs = possible_epochs[randrange(len(possible_epochs))]
        epochs = 1000

        possible_learning_rates = [0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.0025, 0.00025]
        #learning_rate = possible_learning_rates[randrange(len(possible_learning_rates))]
        learning_rate = 0.0005

        #sgd  = torch.optim.SGD(model.parameters(), lr = learning_rate)
        adam = torch.optim.Adam(model.parameters(), lr = learning_rate)
        adamW = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        nadam = torch.optim.NAdam(model.parameters(), lr = learning_rate)
        adamax = torch.optim.Adamax(model.parameters(), lr = learning_rate)
        #adagrad = torch.optim.Adagrad(model.parameters(), lr = learning_rate)
        #rprop = torch.optim.Rprop(model.parameters(), lr = learning_rate)

        possible_optimizers = [adam, adamW, nadam, adamax]
        #optimizer = possible_optimizers[randrange(len(possible_optimizers))] 
        optimizer = adamW
        print(model)

        #Utilize a named tuple to keep track of scores at each epoch
        model_hist = collections.namedtuple('Model', 'epoch loss val_loss')
        model_loss = model_hist(epoch=[], loss=[], val_loss=[])

        #averaging weights of model https://arxiv.org/abs/1803.05407 <- try someday
        #swa_model = AveragedModel(model)

        #training the AE
        train(model=model, epochs=epochs, model_loss=model_loss)

        #Plotting losses over epocs in the training phase
        x = np.linspace(0, epochs - 1, epochs)

        #IMPORTING LEAK-DATA
        data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';', parse_dates= True, index_col = 0)

        max_leak = {}
        for col in data.columns:
            max_leak[col] = data[col].idxmax()

        burst_and_repaired_leaks = ['p158','p183','p232','p369','p461','p538','p628','p673','p866']
        slow_increase_leaks = ['p31','p461']
        neverending_leaks = ['p257', 'p427', 'p654', 'p810']

        #reconstruction loss over the whole dataset
        res = [] 
        xt = torch.FloatTensor(X.values)
        #24_02: if i remember correctly, there was a problem with the batch normalization initially; 
        # had to reformat input for it to work. the section below does that with new_res
        
        per_neuron = X.copy()
        for col in per_neuron.columns:
            per_neuron[col].values[:] = 0
        
        # for ii in range(xt.shape[0]):
        #     res.append(score(xt[ii, :]))

        per_neuron = score_v2(xt, per_neuron)
        per_neuron = per_neuron-X
        
        for jjj in range(len(xt)-batch_size):
            b = xt[(jjj):(jjj+batch_size)]
            res.append(score(b))

        # for ii in range(0, xt.shape[0], batch_size): #with batch_normalization <-- original version
        #     a = xt[ii:(ii+batch_size)]         
        #     res.append(score(a))     
                    
        fig = plt.figure(1, figsize = (14,8))
        ax = plt.subplot(111)

        #leak for p232 occurs at 5th of february i think -> correspond to
        #from excel sheet: starts at 8688, peaked between 9500-11500
    
        #columns/pressure sensors closest to the p232 leak:
        interesting_sensors = ['n163', 'n458', 'n613', 'n342','Demand', 'n216', 'n458', 'n740']
        #interesting_sensors = [sensor_columns]
        all_sensors = interesting_sensors + sensor_columns
        #https://prnt.sc/oEG4GIibABjh

        nper_neuron = pd.DataFrame()
        print(per_neuron)

        for col in per_neuron:
            if col in all_sensors:
                nper_neuron[col] =  per_neuron[col]

        per_neuron = nper_neuron
        #careful whether nper or per
        #per_neuron = per_neuron.drop(columns = ['Demand'], axis = 1)

        #leak for p232 occurs at 5th of february i think -> correspond to
        #from excel sheet: starts at 8688, peaked between 9500-11500

        per_neuron.to_csv(path_all_neurons_csv_files+f'\_{kk}.csv')

        per_neuron = per_neuron.rolling(500).mean()
        for i, col in enumerate(per_neuron.columns):
            if i >= 9 and i < 18:   
                ax.plot(per_neuron[col][8000:13000], label = col, marker = 11)  #careful whether nper or per
                #ax.plot(per_neuron[col], label = col, marker = 11)  #careful whether nper or per
            elif i >= 18 and i < 27: 
                ax.plot(per_neuron[col][8000:13000],label = col, marker = "D")
            elif i >= 27:
                ax.plot(per_neuron[col][8000:13000],label = col, marker = "x")
            else:
               ax.plot(per_neuron[col][8000:13000], label = col)
             #ax.plot(per_neuron[col], label = col)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax = plt.gca()
        ax.set_ylim([-1.5, 1.5])
        plt.legend(ncol=3, loc = 'best', bbox_to_anchor = (1, 1))
        #plt.show()
        plt.clf()

        #plotting for saved file
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

        # new_res = []
        # i = 0
        # for j in range(len(res)*batch_size):
        #     if j % batch_size==0:
        #         new_res.append(res[i])
        #         i = i+1
        #     else:
        #         new_res.append(0)

        #new_res = np.array(new_res, dtype = float)
        res = np.array(res, dtype =float)
        #np.savetxt("test-res_test_v5.csv", res, delimiter=",")


        bottom_subplot = fig.add_subplot(gs[2:6, 0:8])
        #res har 102620 values, missing [batch_size] datapoints ideal would be 105120 i think, think it is 105120-batch_size
        plt.plot(X.index[:len(res)], res, color = colors[1], label = '_nolegend_')

        a, b, c = 0, 0, 0
        #need to fix this at some point
        for leak in max_leak:
                        
            #if leak == 'p232' or 'p158' or 'p673' or 'p866' or 'p461' or 'p183':
            if leak in true_leak_idx_burst_and_repair: 
                leak_date = max_leak[leak]
                #leak_date = leak_date.strftime('%Y-%d-%m %X')
                leak_date = pd.to_datetime(leak_date)
                plt.plot(leak_date, 0, marker = 'X', color = 'blue', label = 'burst & repair', alpha = 0.8)
            # else:
            #     leak_date = max_leak[leak]
            #     plt.plot(leak_date, 0, marker = 'o', color = 'blue', label = 'burst & repair', alpha = 0.8)

            # if leak in slow_increase_leaks:
            #     if leak == dataset:
            #         if leak == 'p31':
            #             leak_date = max_leak[leak]
            #             leak_date = leak_date.strftime('%Y-%m-%d %X')
            #             leak_date = pd.to_datetime(leak_date)
            #             plt.plot(leak_date, 0, marker = 'o', color = 'black', label = 'burst & repair', alpha = 0.8)
            #         else: 
            #             leak_date = max_leak[leak]
            #             plt.plot(leak_date, 0, marker = 'o', color = 'black', label = 'slow increase & repair' if b == 0 else "", alpha = 0.8)

            # if leak in neverending_leaks:
            #     if leak == dataset:
            #         leak_date = max_leak[leak]
            #         #leak_date = leak_date.strftime('%Y-%d-%m %X')
            #         #leak_date = pd.to_datetime(leak_date)
            #         plt.plot(leak_date, 0, marker = 'o', color = 'red', label = 'neverending' if c == 0 else "",  alpha = 0.8)
            #     #c = c+1

        param_text = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Dataset = {file_path} \n\n Model: {model}') 
        #param_text = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Dataset = {file_path}') 

        param_text2 = (f'optimzer: {optimizer},\nloss function : {loss},\ntraining time: SOY until {time1},\ntest time: between {time1} and {time2},\nvalidation time: {time3} until EOY' )
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3)

        plt.text(0.02, 0.70, param_text, bbox= props, fontsize=8, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.72, param_text2, bbox= props, fontsize=8, transform=plt.gcf().transFigure)

        plt.title('Reconstruction')
        plt.ylabel('Error')
        plt.legend()
        #plt.show()

        gc.collect()
        folder_path = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\figures_23_02_dataset_test\\"

        fig.savefig(path_reconstruction_figures+"\_"+str(kk))
        plt.clf()

        #evaluation plots
        # res = pd.DataFrame(res)
        # res = res.rolling(200).mean()
        # res.values.tolist()
        possible_lags = [288*5, 288*7, 288*14, 288*21, 288*28] #initially used way to low lag i think
        #lag = possible_lags[randrange(len(possible_lags))]
        lag = 288*21

        possible_thresholds = [3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25]
        #threshold = possible_thresholds[randrange(len(possible_thresholds))]
        threshold = 3.3

        possible_influences = [0.9, 0.85, 0.95, 1]
        #influence = possible_influences[randrange(len(possible_influences))]
        influence = 1.4

        abs_tol = 288  

        test = thresholding_algo(res, lag = lag, threshold= threshold, influence= influence)

        pylab.subplot(211)
        pylab.plot(np.arange(1, len(res)+1), res)

        pylab.plot(np.arange(1, len(res)+1),
                test["avgFilter"], color="cyan", lw=2)

        pylab.plot(np.arange(1, len(res)+1),
                test["avgFilter"] + threshold * test["stdFilter"], color="green", lw=2)

        pylab.plot(np.arange(1, len(res)+1),
                test["avgFilter"] - threshold * test["stdFilter"], color="green", lw=2)

        pylab.subplot(212)
        pylab.step(np.arange(1, len(res)+1), test["signals"], color="red", lw=2)
        pylab.ylim(-1.5, 1.5)

        pylab.savefig(path_evaluation_plots+"\__"+str(kk))
        pylab.clf()

        #results
        idx = np.where(test["signals"] == 1)
        number = idx[0][:]

        print(number)
        

        #two iterations for safety's sake, ordered_cluster algorithm might accidently slice a big cluster in two
        a = list(ordered_cluster(number, abs_tol/2))
        container = []
        for aa in a:
            container.append(mean(aa))

        b = list(ordered_cluster(container, abs_tol))
        containerv2 = []
        for bb in b:
            containerv2.append(mean(bb))

        #based on when max value of leak occurs in 2018_leakages.csv-file
        #transferred from random_plotting.py
       
        #p232 er egt 10134, skfitet til 8000 for testeformål, 
        #problem: max_idx from earlier, do not match what the evaluation algorithm detects very closely, need to find instance of when it first happens. 
        #values above might need to be adjusted somewhat.
        #max peak occurs too late in the leak's lifespan, so find other metric. i.e. first time reaches 80% of max peak or something like that
        results_holder = pd.DataFrame(index = [':)'],  columns = ['n', 'Accuracy', 'Precision', 'Recall', 'F1_score','True positive', 'False positive'], data = 0)
        
        for c in containerv2:
            a = any(math.isclose(c, item, abs_tol=abs_tol) for item in holderv2)
            print(c, a)
            if a:   
                current_value = results_holder['True positive'].iloc[-1]
                results_holder.loc[results_holder.index[-1],'True positive'] = current_value + 1
            else:
                current_value = results_holder['False positive'].iloc[-1]
                results_holder.loc[results_holder.index[-1],'False positive'] = current_value + 1
        
        results_holder.loc[results_holder.index[-1], 'n'] = kk
        
        results = pd.concat([results, results_holder])

        results.to_csv(path_result_files +'\_' + (str(kk)) + '.csv')

        algotxt = (f'\nlag: {str(lag)}, threshold: {str(threshold)}, influence: {str(influence)}, abs_tol: {str(abs_tol)}')
        modeltxt =  param_text + param_text2 + algotxt
        print(modeltxt)

        txt_file = open(path_model_config_files+'\_'+str(kk)+'.txt', "w")
        txt_file.write(modeltxt)
        txt_file.close()

        print(results)






    


