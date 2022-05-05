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
from datetime import datetime

import math
from statsmodels.tsa.seasonal import STL

from autoencoders_and_more import AutoEncoder_1, AutoEncoder_2, AutoEncoder_3, AutoEncoder_4, AutoEncoder_5, AutoEncoder_6
from autoencoders_and_more import thresholding_algo, ordered_cluster


from autoencoder_test_test import score, score_v2

from os import listdir

from rolling_AE_functions import create_folders, import_data, plot_reconstruction, leak_location_plot, threshold_function_v2, threshold_function_v3
import pylab

import gc

    #training, haven't made any changes since start
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
        if epoch%500==0 or epoch == epochs:
            print(f'Epoch: {epoch}   Loss: {l.item():.5f}    Val_Loss: {np.mean(val_scores):.5f}') 


    #regular score function, 105kx1 output
def score(x):
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred, x1).item()


def score_v2(xt, df):
    """"
    Output from this function is a dataframe in which each cell is the MSE between
    input cell and model's predicition for the given input cell
    so 105kx36 or so, if whole year is run through, but typically in 2 week chunks, 4033x36
    """
    y_pred = model(V(xt))     
    x1 = V(xt)    

    n, m = np.shape(y_pred)
    for i in range (n):
        df.iloc[i] = (x1[i].detach().numpy()-y_pred[i].detach().numpy())**2/2
        
    return df


if __name__ == '__main__':

    #used for keeping track of time, and switching between time formats of timestep number 39933 and say may 5th
    time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
    time_2018 = {'Timesteps': time_2018}
    time_df = pd.DataFrame(data = time_2018)


    leak_flows = pd.read_csv('2018_leakages_dot.csv', delimiter = ';')

    #nå regnes start fra 0
    leak_data = [{'leak':'p232', 'start': 8687,   'stop': 11634, 'h_number': 1,  'h_stop': 3}, 
                {'leak': 'p461', 'start': 6625,   'stop': 26530, 'h_number': 1,  'h_stop': 4},
                {'leak': 'p538', 'start': 39561,  'stop': 43851, 'h_number': 7,  'h_stop': 11},
                {'leak': 'p628', 'start': 35076,  'stop': 42883, 'h_number': 6,  'h_stop': 10},
                {'leak': 'p673', 'start': 18335,  'stop': 23455, 'h_number': 2,  'h_stop': 6},
                {'leak': 'p866', 'start': 43600,  'stop': 46694, 'h_number': 6,  'h_stop': 10},
                {'leak': 'p158', 'start': 80097,  'stop': 85125, 'h_number': 17, 'h_stop': 20},
                {'leak': 'p183', 'start': 62817,  'stop': 70192, 'h_number': 12, 'h_stop': 16},
                {'leak': 'p369', 'start': 85851,  'stop': 89815, 'h_number': 18, 'h_stop': 21},
                {'leak':  'p31', 'start': 51573,  'stop': 64437, 'h_number': 9,  'h_stop': 20}, # skjer noe rart med p31 <-- er i sone A
                {'leak': 'p654', 'start': 73715,  'stop': -9999, 'h_number': 10, 'h_stop': 21},  #-9999 since never ends, start at 5 m3/h
                {'leak': 'p257', 'start': 2312,   'stop': -9999, 'h_number': 1,  'h_stop': 21},  #starter omtrentlig med en gang, 1 er laveste h tallet som kan kjøres
                {'leak': 'p810', 'start': 61254,  'stop': -9999, 'h_number': 13, 'h_stop': 21},   #-9999 since never ends, start at 5 m3/h
                {'leak': 'p427', 'start': 13301,  'stop': -9999, 'h_number': 1,  'h_stop': 21},]  #-9999 since never ends, start at 5 m3/h

    #leak_data = [{'leak':'p232', 'start': 9124,   'stop': 11634, 'h_number': 1,  'h_stop': 24}, 
                # {'leak': 'p461', 'start': 13807,  'stop': 26530, 'h_number': 1,  'h_stop': 24},
                # {'leak': 'p538', 'start': 39561,  'stop': 43851, 'h_number': 1,  'h_stop': 24},
                # {'leak': 'p628', 'start': 36520,  'stop': 42883, 'h_number': 1,  'h_stop': 24},
                # {'leak': 'p673', 'start': 18335,  'stop': 23455, 'h_number': 1,  'h_stop': 24},
                # {'leak': 'p866', 'start': 43600,  'stop': 46694, 'h_number': 1,  'h_stop': 24},
                # {'leak': 'p158', 'start': 80097,  'stop': 85125, 'h_number': 1, 'h_stop': 24},
                # {'leak': 'p183', 'start': 62817,  'stop': 70192, 'h_number': 1, 'h_stop': 24},
                # {'leak': 'p369', 'start': 85851,  'stop': 89815, 'h_number': 1, 'h_stop': 24},
                # {'leak':  'p31', 'start': 57095,  'stop': 64437, 'h_number': 1,  'h_stop': 24}, # skjer noe rart med p31
                # {'leak': 'p654', 'start': 73715,  'stop': -9999, 'h_number': 1, 'h_stop': 24},  #-9999 since never ends, start at 5 m3/h
                # {'leak': 'p257', 'start': 288*7,  'stop': -9999, 'h_number': 1, 'h_stop': 24},  #starter omtrentlig med en gang, 1 er laveste h tallet som kan kjøres
                # {'leak': 'p810', 'start': 61254,  'stop': -9999, 'h_number': 1, 'h_stop': 24},   #-9999 since never ends, start at 5 m3/h
                # {'leak': 'p427', 'start': 13301,  'stop': -9999, 'h_number': 1,  'h_stop': 24},] 

    leak_df = pd.DataFrame(data = leak_data)

    burst_datasets = ['p158', 'p183', 'p369', 'p538', 'p673', 'p866'] 
    slow_increase_datasets = ['p232','p461','p628','p31']
    neverending_datasets = ['p257', 'p427', 'p654', 'p810']
    
    #a few datasets to test on initially https://prnt.sc/ry3QhC8_YrRp <--25_04 https://prnt.sc/73z9Rp2wcoOV more 25_04
    #p369: quick burst, fant med rolling_AE greit, sliter finne med 25_04 konfigurasjon, 
    #p183: fant ikke med rolling_AE greit
    #p628: slow increase
    #p538 gikk greit 
    #p158<-------- tror regn flaks at finner noe lekkasje, når graver i det, så ser en ikke tydelig lekkasje
    #p654: neverending
    #p866, opprinnelige datasettet, funnet enkelt med rolling_ae og første jeg lagde her
    #p232, skjer så tidlig i året, må trolig fikse på 2-uker grensene eller lignende
        #evt, lage spesial trenings-fase for det datasettet
    
    
        #husk kjøre disse, var ikke ok sist
    #some_abrupt_burst_datasets = ['p31']

    #some_abrupt_burst_datasets = ['p257', 'p427', 'p654', 'p810'] # kjør disse etterpå

    #for p628,  gjør en vurdering om hvilke sensorer som detekterer først, tror ikke nærmeste slår helt ut alene
    #some_abrupt_burst_datasets = ['p538','p158','p461','p31','p183','p866','p628','p369','p232']
    some_abrupt_burst_datasets = ['p232']
    #dataframe to keep the total results, the preliminary results are concatenated to each iteration
    super_results = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accum. difference'], data = 0)

    for kk in range(20000):


        startTime = datetime.now()
        """
        General idea is the following:
            first for-loop creates 100 different configurations of an autoencoder. Can also switch between different autoencoder
            architectures, but initially only AE_6
            then, for a given model configuration, it is run on each of the datasets found in some_abrupt_burst_datasets
            in the while loop, it is run until a leak is found
        """
            #creates folders for saving various information
        folders = create_folders()

                #change model configuration
        possible_loss_functions = [nn.MSELoss(), nn.L1Loss()]
        #loss = possible_loss_functions[randrange(len(possible_loss_functions))]
        loss = nn.MSELoss()
        #loss = nn.L1Loss() #use nn.L1loss()/MAE for the _correct_versions

        possible_batch_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500]
        #batch_size = possible_batch_sizes[randrange(len(possible_batch_sizes))]
        batch_size = 750 

        possible_epochs = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 750]
        #epochs = possible_epochs[randrange(len(possible_epochs))]
        epochs = 2500

        possible_learning_rates = [0.001, 0.0001, 0.00001, 0.005, 0.0005, 0.0025, 0.00025]
        #learning_rate = possible_learning_rates[randrange(len(possible_learning_rates))]
        learning_rate = 0.0025

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
        #optimizer = possible_optimizers[randrange(len(possible_optimizers))]  
        optimizer = adam #used nadam for the _correct_versions

            #starting point of the accumulated timestep difference, ideally a configuration has a low accumulation of timestep difference
            #between the detected leak location and the true leak location, although this criteria has its drawbacks, 
            #it needs more refinement later

        accumulated_diff = 0
        possible_multipliers = np.arange(1, 6, 0.25)
        possible_multipliers_v2 = np.arange(2, 6, 0.25)
        
        #comparison_number_multiplier = possible_multipliers[randrange(len(possible_multipliers))]
        #comparison_number_multiplier = possible_multipliers[kk]
        comparison_number_multiplier = 1
        comparison_number_multiplier_v2 = possible_multipliers_v2[kk]
        
        #comparison_number_multiplier_v2 = 1.15

        for dataset_number, dataset in enumerate(some_abrupt_burst_datasets):
            res = np.array([]) 

            leak_found = False
            no_leak_found = False

            timesteps_in_day = 288
            num_days_tv = 14
            freq_t_v = str(num_days_tv) + 'D'
            num_days = 14
            freq = str(num_days)+'D'

                #timestep frequency in which the training, validation changed, currently 2 weeks
            every_second_week = pd.bdate_range('2018-01-01','2018-12-31 23:55:00', freq = freq_t_v)

            #Empty dataframe in which to concatenate output from score-functions later on
            Y = import_data('p232') #import a any dataset, to extract columns-formatting
            X_per_sensor = pd.DataFrame(columns = Y.columns)
           
            for iii, leak in enumerate(leak_df['leak']): 
                if leak == dataset:
                    dataset_nr = iii
                #no idea what these variables are typically called, but h = 0 first iteration, h = 1 second iteration and so on.
            h = leak_df.loc[dataset_nr]['h_number']

            while leak_found == False:
                
                #updates training and validation time each iteration of the while-loop, 
                start_train_time = every_second_week[h-1]
                #start_train_time = 0

                train_time = every_second_week[h]
                vald_time = every_second_week[h+1]

                if h > leak_df.loc[dataset_nr]['h_stop']:
                    leak_found = True
                    leak_timestep = -9999
                    break

                print(f'Start training time:  {start_train_time}, training until: {train_time}, validation until: {vald_time}')

                #if end of year and leak still not found.
                if vald_time >= pd.to_datetime("2018-12-31 00:00:00"):
                    no_leak_found = True
                    break

                #separate timeframe-variable to keep track of training times, freq is also 14 days.
                test_timeframe = pd.bdate_range(vald_time,'2018-12-31 23:55:00', freq = freq)

                time2 = test_timeframe[0]
                time3 = test_timeframe[1]
                print(f'Testing from {time2} until {time3}')

                #keep track of valid_time timestep in numerical value for later 
                vald_time_int = time_df.loc[time_df['Timesteps'] == vald_time]
                vald_time_int = int(vald_time_int.index.values)

                #import the dataframe with the given dataset, and normalize
                X = import_data(dataset)
                X = X[:'2018-12-31']
                X = (X - X[:train_time].mean())/X[:train_time].std() 

                #training and validation data is extracted from X
                X_train = X[start_train_time:train_time]
                #X_train = X[:train_time]

                X_val = X[train_time:vald_time]
                xtr = torch.FloatTensor(X_train.values) 
                xtv = torch.FloatTensor(X_val.values)

                #testing the AE between time2 and time3
                X_test = X[time2:time3]
                xt = torch.FloatTensor(X_test.values) 

                #creating the dataloader for validation and training
                xdl = DataLoader(xtr, batch_size=batch_size)
                tdl = DataLoader(xtv, batch_size=batch_size)

                #Utilize a named tuple to keep track of scores at each epoch
                model_hist = collections.namedtuple('Model', 'epoch loss val_loss')
                model_loss = model_hist(epoch=[], loss=[], val_loss=[])

                #training the AE
                train(model=model, epochs=epochs, model_loss=model_loss)
                
                #makes a copy of the X_test dataframe to have all the values set to zero, before later filling it from score functions
                X_test_copy = X_test.copy() 
                #X_test_copy_v3 = X_test.copy()

                for col in X_test_copy.columns:
                    X_test_copy[col].values[:] = 0

                #extract values from X_test to run through the model, normal score-function
                iter_res = []
                xt = torch.FloatTensor(X_test.values) 

                for i in range(len(xt)-2): 
                    b = xt[(i):(i+2)]
                    iter_res.append(score(b))

                #fills up the aforementioned dataframe with the MSE per sensor
                X_test_copy_filled = score_v2(xt, X_test_copy)

                #    an idea for later; can remove seasonality from test/validation stage, then substract it from testing with some scaling factor depending on how far into the year it's gone
                
                    #get the numerical value for the beginning of the testing-time
                h_time_holder_int = time_df.loc[time_df['Timesteps'] == time2]
                h_time_holder_int = int(h_time_holder_int.index.values)

                X_per_sensor = pd.concat([X_per_sensor, X_test_copy_filled])
                #X_per_sensor.to_csv(f'01_may_testing_{epochs}_bs{batch_size}_lr{learning_rate}_dataset{dataset}_.csv')

                if h > 1:
                    thres_results_df, leak_found, leak_timestep = threshold_function_v3(X_per_sensor, dataset, vald_time_int, comparison_number_multiplier, comparison_number_multiplier_v2)

                #concatenate to res
                res = np.concatenate([res, np.array(iter_res)])
                #np.savetxt(f'res_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dataset{dataset}_21_04.csv', res, delimiter= ',')

                print(f'Leak found: {leak_found}')

                h = h + 1 

                if leak_found:
                    print(f'Leak is found : -), {dataset}')
                    print(thres_results_df)
                    
                    #thres_results_df.to_csv(f'02_05_{kk}_{dataset}_{datetime.now().hour}_{datetime.now().minute}.csv')
                    try:
                        super_results.to_csv(f'{kk}_superresults_03_05_overnight_STD_testing.csv')
                    except:
                        pass

                            #saves a reconstruction plot with the learning and validation curves, similar as to shown before in the semester
                    #folder_recon_plot_path = folders[0]
                    #plot_reconstruction(res, model_loss, epochs, leak_df, time_df, folder_recon_plot_path=folder_recon_plot_path, dataset=dataset, iteration = kk)
                    break
                else:
                    leak_timestep = 0


                #various model configuration information that's saved for later inspection
            # modeltxt = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Optimizer: {optimizer}, Loss function: {loss}, Dataset = {dataset} \n\n Model: {model}') 
            # modeltxt2 = (f'Training time ended at: {train_time}, validation time ended at: {vald_time}')
            # modeltxt_total = modeltxt + modeltxt2
            # txt_file = open(folders[3]+'\_'+str(kk)+'_12_04.txt', "w")
            # txt_file.write(modeltxt_total)
            # txt_file.close()

            #fills the time in which training and validation occured with 0s with goal of maintaining the datatimeindex on the x-axis in graph
            zero_array = np.zeros(shape = (vald_time_int, 1))
            res = np.append(zero_array, res)

            #dataframe of the results to be plotted in the leak_detection plot.
            #leak_location_plot(time_df, res, leak_df, leak_timestep, train_time, vald_time, folders, dataset, kk)
            
            
             #har iikke giddet sjekket for bugs
                    #creates a dataframe in which to save results
            super_results_holder = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accum. difference'], data = 0)
            super_results_holder['Model iteration'] = kk
            super_results_holder['Dataset'] = dataset
            super_results_holder['Time detected leak'] = leak_timestep
            if leak_timestep != 0:
                super_results_holder['Volume'] = leak_flows.iloc[leak_timestep][dataset]
            else:
                super_results_holder['Volume'] = -9999

            #super_results_holder['Threshold'] = threshold_number
            if no_leak_found:
                super_results_holder['Time detected leak'] == -9999

            super_results_holder['Time true leak'] = leak_df['start'][dataset_nr]

            if no_leak_found == False:
                accumulated_diff = abs(leak_timestep - leak_df['start'][dataset_nr]) + accumulated_diff

            super_results_holder['Difference'] = (abs(leak_timestep - leak_df['start'][dataset_nr]))
            super_results_holder['Accum. difference'] = accumulated_diff
            super_results_holder['Running time 1 DS'] = str(datetime.now()-startTime)
            super_results_holder['cmp_1'] = comparison_number_multiplier #for averages
            super_results_holder['cmp_2'] = comparison_number_multiplier_v2 #for STDs

            super_results = pd.concat([super_results, super_results_holder])

            print(super_results)
            super_results.to_csv(folders[2] +'\_02_may_testing_' + (str(kk)) + '.csv')
          