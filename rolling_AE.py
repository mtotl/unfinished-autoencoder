from re import A
from tkinter.tix import Y_REGION
from zoneinfo import reset_tzpath
from matplotlib import container
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pytz import FixedOffset
import seaborn as sns

import collections
import os
import os.path
from os import path
from sklearn import cluster
from sklearn.cluster import KMeans

import torch
import torch.nn as nn

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

from random import randrange
from statistics import mean
import math
from statsmodels.tsa.seasonal import STL

from autoencoders_and_more import AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4, AutoEncoder_5, AutoEncoder_6
from autoencoders_and_more import thresholding_algo, ordered_cluster, get_timestep


from autoencoder_test_test import score, score_v2

from os import listdir

from rolling_AE_functions import create_folders, import_data, plot_reconstruction


import pylab

import gc


def train(epochs, model, model_loss):
    try:
        c = model_loss.epoch[-1]
    except:
        c = 0
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
    
    y_pred = model(V(xt))     #output is 4033x36, 4033 timesteps in two weeks, 36 features.
    x1 = V(xt)    

    #output is 4033x36, 4033 timesteps in two weeks, 36 features.
    n, m = np.shape(y_pred)
    for i in range (n):
        for j in range (m):
            df.iloc[i][j] = (x1[i][j]-y_pred[i][j])**2/2 

    return df


def score_v3(x):
    y_pred = model(V(x))
    #print(f'dette er y_pred:\n {y_pred}')
    x1 = V(x)
    #print(f'dette er x1: \n {x1}')
    return (x1-y_pred).detach().numpy()


if __name__ == '__main__':

    time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
    time_2018 = {'Timesteps': time_2018}
    time_df = pd.DataFrame(data = time_2018)

    num_days_tv = 14
    freq_t_v = str(num_days_tv) + 'D'
    every_second_week = pd.bdate_range('2018-01-01','2018-12-31 23:55:00', freq = freq_t_v)
    #every_second_week_shifted = pd.bdate_range('2018-01-07', '2018-12-31 23:55:00', freq = '14D')
    #print(every_second_week_shifted)
    timesteps_in_day = 288

    train_time = every_second_week[1]
    print(f'Training time ends at: {train_time}')

    vald_time = every_second_week[2]
    print(f'Validation time ends at: {vald_time}')

    num_days = 14
    #freq = '14D'
    freq = str(num_days)+'D'
    test_timeframe = pd.bdate_range(vald_time,'2018-12-31 23:55:00', freq = freq)

    vald_time_int = time_df.loc[time_df['Timesteps'] == vald_time]
    vald_time_int = int(vald_time_int.index.values)

    leak_data = [{'leak':'p232', 'start': 9124,   'stop': 11634},
                {'leak': 'p461', 'start': 13807,  'stop': 26530}, 
                {'leak': 'p538', 'start': 39561,  'stop': 43851}, 
                {'leak': 'p628', 'start': 36520,  'stop': 42883}, 
                {'leak': 'p673', 'start': 18335,  'stop': 23455}, 
                {'leak': 'p866', 'start': 43600,  'stop': 46694},
                {'leak': 'p158', 'start': 80097,  'stop': 85125},
                {'leak': 'p183', 'start': 62817,  'stop': 70192}, 
                {'leak': 'p369', 'start': 85851,  'stop': 89815}, 
                {'leak':  'p31', 'start': 57095,  'stop': 64437}]  

    leak_df = pd.DataFrame(data = leak_data)
    some_abrupt_burst_datasets = ['p369', 'p866' ,'p158', 'p183']

    super_results = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accumulated difference'], data = 0)

    for kk in range(100):

        #folders = create_folders()
                #change model configuration
        possible_loss_functions = [nn.MSELoss(), nn.L1Loss()]
        loss = possible_loss_functions[randrange(len(possible_loss_functions))]
        #loss = nn.MSELoss()
        #loss = nn.L1Loss

        possible_batch_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500]
        batch_size = possible_batch_sizes[randrange(len(possible_batch_sizes))]
        #batch_size = 1000 

        possible_epochs = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 750]
        #epochs = possible_epochs[randrange(len(possible_epochs))]
        epochs = 2

        possible_learning_rates = [0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.0025, 0.00025]
        learning_rate = possible_learning_rates[randrange(len(possible_learning_rates))]
        #learning_rate = 1e-05

        number_of_neurons = 36
        #model = possible_models[randrange(len(possible_models))]
        #model = model(len(X_train.columns))
        model = AutoEncoder_6(number_of_neurons)
        print(model)

        adam = torch.optim.Adam(model.parameters(), lr = learning_rate)
        adamW = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        nadam = torch.optim.NAdam(model.parameters(), lr = learning_rate)
        adamax = torch.optim.Adamax(model.parameters(), lr = learning_rate)

        possible_optimizers = [adam, adamW, nadam, adamax]
        optimizer = possible_optimizers[randrange(len(possible_optimizers))] 
        #optimizer = nadam

        #param_text = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Dataset = {alternative_file_path} \n\n Model: {model}') 
        #param_text = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Dataset = {file_path}') 
        #param_text2 = (f'optimzer: {optimizer},\nloss function : {loss},\ntraining time: SOY until {time1},\validation time: between {time1} and {time2}, testing time: {time2} until EOY' )

            #try same model on certain number of datasets
        acc_diff = 0
        for dataset in (some_abrupt_burst_datasets):

            modeltxt = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Optimizer: {optimizer}, Loss function: {loss}, Dataset = {dataset} \n\n Model: {model}') 
            modeltxt2 = (f'Training time ended at: {train_time}, validation time ended at: {vald_time}')
            modeltxt_total = modeltxt + modeltxt2
            # txt_file = open(folders[3]+'\_'+str(kk)+'.txt', "w")
            # txt_file.write(modeltxt_total)
            # txt_file.close()

            res = np.array([]) 
            #dataset = some_abrupt_burst_datasets[randrange(len(some_abrupt_burst_datasets))]

            X = import_data(dataset)

            X = X[:'2018-12-31']
            
            X = (X - X[:train_time].mean())/X[:train_time].std() #based on values found during training
            X_train = X[:train_time]
            X_val = X[train_time:vald_time]

            xtr = torch.FloatTensor(X_train.values) 
            xtv = torch.FloatTensor(X_val.values)

            xdl = DataLoader(xtr, batch_size=batch_size)
            tdl = DataLoader(xtv, batch_size=batch_size)

            #Utilize a named tuple to keep track of scores at each epoch
            model_hist = collections.namedtuple('Model', 'epoch loss val_loss')
            model_loss = model_hist(epoch=[], loss=[], val_loss=[])

            #training the AE
            train(model=model, epochs=epochs, model_loss=model_loss)

            X_per_sensor = pd.DataFrame(index = [''], columns = X.columns, data = 0)
            print(f'this is X_per_sensor {X_per_sensor}')

            for j in range(0, len(test_timeframe)-1):

                leak_found = False
                iter_res = []
                iter_resv2 = []
                time2 = test_timeframe[j]
                time3 = test_timeframe[j+1]

                print(f'Testing from {time2} until {time3}')

                X_test = X[time2:time3]

                X_test_copy = X_test.copy() #should be same format dataframe as X_test but all 0s
                for col in X_test_copy.columns:
                    X_test_copy[col].values[:] = 0

                xt = torch.FloatTensor(X_test.values) 

                #print(f'dette er X_test: {X_test}')

                # for i in range(len(xt)-batch_size): #old version
                #     b = xt[(i):(i+batch_size)]
                #     iter_res.append(score(b))

                for i in range(len(xt)-2): #works, but not 100% sure as to why. re-consider a bit when writing thesis
                    b = xt[(i):(i+2)]
                    iter_res.append(score(b))

                    #per_neuron attempt, prøv deg litt frem, men husk å plotte underveis for å få innblikk i hva som skal være
                    #sjekk demand-plot inn, og tank-inn  og en eller annen trykksensor
                    #har score_v3 fra _test_test som gjør jobben tror jeg
 
                X_test_copy_filled = score_v2(xt, X_test_copy)

                X_per_sensor = pd.concat([X_per_sensor, X_test_copy_filled])

                print(X_per_sensor)

                # print(f'this is X_test:{X_test}')
                # print(f'this is X_test_copy_filled: {X_test_copy_filled}')

                res_df = pd.DataFrame(data = iter_res)
                res_df = res_df.rolling(int(timesteps_in_day/4), min_periods= 1).mean()
                res_np = res_df.to_numpy()
                iter_res = res_np.reshape(len(res_np))

                j_time_holder_int = time_df.loc[time_df['Timesteps'] == test_timeframe[j]]
                j_time_holder_int = int(j_time_holder_int.index.values)

                if j > 1: #might miss it if there's a leak in the first week or two? --> may be some try - exception-python tricks to help -> do same thing on iter_res, but find out how works later
                    threshold = res[-num_days*timesteps_in_day:].max()*1.3 # renews each iteration as res' size increases with iter_res appended
                        #if 50% larger than last two weeks' maximum
                        #may be able to detect peak of the day the leak occurs, 
                        #but need comparison with similar time of day also for higher accuracy
                        #will only trigger at say 10-ish o'clock at peak water demand so to speak.
                        #but leak can occur any point during the day

                    for it in range(len(iter_res)):
                        if iter_res[it] > threshold: # or iter_res[:it].min() > lower_threshold  #or iter_res.min() > lower_threshold:
                            leak_timestep_iter_res = it
                            leak_found = True
                            break
                
                res = np.concatenate([res, np.array(iter_res)])
                print(f'Leak found: {leak_found}')

                if leak_found:
                    leak_timestep = leak_timestep_iter_res + j_time_holder_int 

                    #folder_recon_plot_path = folders[0]
                    #plot_reconstruction(res, model_loss, epochs, leak_df, time_df, folder_recon_plot_path=folder_recon_plot_path, dataset=dataset, iteration = kk)
                    break

                else:
                    leak_timestep = 0


                
                # plt.plot(res)
                # plt.show()
                # plt.clf()


            # for col in X_per_sensor:
            #     to_plot = X_test_copy_filled[col].to_numpy()
            #     #to_plot = X_per_sensor[col].to_numpy()
            #     plt.plot(to_plot, label = col)
            #     plt.title('MSE per sensor')
            #     plt.legend()
            #     plt.show()
            #     plt.clf()

            #     to_plotv2 = X_test[col].to_numpy()

            #     plt.plot(to_plotv2, label = col)
            #     plt.title('X_test, the regular inputbr')
            #     plt.legend()
            #     plt.show()
            #     plt.clf()



            zero_array = np.zeros(shape = (vald_time_int, 1))
            res = np.append(zero_array, res)

            res_df = pd.DataFrame(index = time_df['Timesteps'][:len(res)], columns=['res'], data = res)  #0s during the time spent training and validating
            fig = plt.figure(1, figsize = (14,6)) #gjorde dette en endring? fikk lagret en liten plot før

            colors = sns.color_palette("rocket", 10)
            for i, leak in enumerate(leak_df['leak']): 
                if leak == dataset:        
                    plt.plot(time_df.iloc[leak_df['start'][i]], 0, marker = 'X', color = colors[i], label = dataset, alpha = 0.8)
                    plt.plot(time_df.iloc[leak_df['stop'][i]], 0, marker = 's', color = colors[i], label = '', alpha = 0.8)

            plt.plot(time_df.iloc[leak_timestep], 0, marker = 'x', color = 'red', label = 'Detected leak', markersize=14)

            plt.plot(train_time, 0, marker = 'o', color = 'darkblue', label = 'end of start and validation time', alpha = 0.8)
            plt.plot(vald_time, 0, marker = 'o', color = 'darkblue', label = '', alpha = 0.8)

            plt.plot(res_df, color = 'darkgreen', label = 'Error')
            #plt.xlabel(f'{freq}')
            plt.ylabel('Reconstruction error')
            #plt.legend()
            #plt.show()
            
            #plt.savefig(folders[5]+ "\_"+ str(dataset) + "_" + (str(kk)))
            plt.clf()

            for iii, leak in enumerate(leak_df['leak']): 
                if leak == dataset:
                    dataset_nr = iii

            super_results_holder = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accumulated difference'], data = 0)
            super_results_holder['Model iteration'] = kk
            super_results_holder['Dataset'] = dataset

            super_results_holder['Time detected leak'] = leak_timestep
            super_results_holder['Time true leak'] = leak_df['start'][dataset_nr]

            acc_diff = abs(leak_timestep - leak_df['start'][dataset_nr]) + acc_diff

            super_results_holder['Difference'] = (abs(leak_timestep - leak_df['start'][dataset_nr]))
            #super_results_holder['Accumulated difference'] = super_results_holder.loc[super_results_holder.index[-1]]['Difference'] + super_results.loc[super_results.index[-1]]['Accumulated difference']
            super_results_holder['Accumulated difference'] = acc_diff

            super_results = pd.concat([super_results, super_results_holder])


            print(super_results_holder)

            print('\n\n\n')

            print(super_results)
            #super_results.to_csv(folders[2] +'\_' + (str(kk)) + '.csv')


            















