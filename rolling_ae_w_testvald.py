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
from autoencoders_and_more import thresholding_algo, ordered_cluster, get_timestep


from autoencoder_test_test import score, score_v2

from os import listdir

from rolling_AE_functions import create_folders, import_data, plot_reconstruction


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
        if epoch%100==0 or epoch == epochs-1:
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
        for j in range (m):
            df.iloc[i][j] = (x1[i][j]-y_pred[i][j])**2/2 

    return df


if __name__ == '__main__':

    #used for keeping track of time, and switching between time formats of timestep number 39933 and say may 5th
    time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
    time_2018 = {'Timesteps': time_2018}
    time_df = pd.DataFrame(data = time_2018)

    #used to determine when leaks first occur, the end is currently not used
    leak_data = [{'leak':'p232', 'start': 9124,   'stop': 11634},
                {'leak': 'p461', 'start': 13807,  'stop': 26530}, 
                {'leak': 'p538', 'start': 39561,  'stop': 43851}, 
                {'leak': 'p628', 'start': 36520,  'stop': 42883}, 
                {'leak': 'p673', 'start': 18335,  'stop': 23455}, 
                {'leak': 'p866', 'start': 43600,  'stop': 46694},
                {'leak': 'p158', 'start': 80097,  'stop': 85125},
                {'leak': 'p183', 'start': 62817,  'stop': 70192}, 
                {'leak': 'p369', 'start': 85851,  'stop': 89815}, 
                {'leak':  'p31', 'start': 57095,  'stop': 64437},
                {'leak': 'p654', 'start': 73715,  'stop': -9999}]  #-9999 since never ends, start at 5 m3/h
    leak_df = pd.DataFrame(data = leak_data)

    #a few datasets to test on initially
    #p369: quick burst, fant med rolling_AE greit
    #p183: fant ikke med rolling_AE greit
    #p628: slow increase
    #p654: neverending
    #p866, opprinnelige datasettet, funnet enkelt med rolling_ae og første jeg lagde her
    #some_abrupt_burst_datasets = ['p183', 'p866','p369','p628','p654']
    some_abrupt_burst_datasets = ['p628','p183','p866','p369']

    #dataframe to keep the total results, the preliminary results are concatenated to each iteration
    super_results = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accumulated difference'], data = 0)

    for kk in range(100):

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
        acc_diff = 0

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
            print(every_second_week[0])

            #Empty dataframe in which to concatenate output from score-functions later on
            Y = import_data('p232') #import a any dataset, to extract columns-formatting
            X_per_sensor = pd.DataFrame(index = [''], columns = Y.columns)

            # possible_thresholds_numbers = np.arange(3, 150, 0.5)
            # #threshold_number = possible_thresholds_numbers[randrange(len(possible_thresholds_numbers))] 
            # threshold_number = 1000
            
                #no idea what these variables are typically called, but h = 0 first iteration, h = 1 second iteration and so on.
            h = 1

            while leak_found == False:
                
                #updates training and validation time each iteration of the while-loop, 
                train_time = every_second_week[h]
                vald_time = every_second_week[h+1]

                print(f'Training until: {train_time}, validation until: {vald_time}')


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

                #import the dataframe with the given dataset, and do a little data-preprocessing
                X = import_data(dataset)
                X = X[:'2018-12-31']
                X = (X - X[:train_time].mean())/X[:train_time].std() 

                #training and validation data is extracted from X
                X_train = X[:train_time]
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
                X_test_copy_v3 = X_test.copy()
                for col in X_test_copy.columns:
                    X_test_copy[col].values[:] = 0
                    X_test_copy_v3[col].values[:] = 0

                #extract values from X_test to run through the model, normal score-function
                iter_res = []
                xt = torch.FloatTensor(X_test.values) 

                for i in range(len(xt)-2): 
                    b = xt[(i):(i+2)]
                    iter_res.append(score(b))

                #fills up the aforementioned dataframe with the MSE per sensor
                X_test_copy_filled = score_v2(xt, X_test_copy)

                #briefly makes iter_res into a dataframe to utilize the .rolling method, and turning it back to a numpy array.
                #might not be beneficial afterall, will require some testing
                #    an idea for later; can remove seasonality from test/validation stage, then substract it from testing with some scaling factor depending on how far into the year it's gone
                
                # res_df = pd.DataFrame(data = iter_res)
                # res_df = res_df.rolling(int(timesteps_in_day/4), min_periods= 1).mean()
                # res_np = res_df.to_numpy()
                # iter_res = res_np.reshape(len(res_np))

                    #get the numerical value for the beginning of the testing-time
                h_time_holder_int = time_df.loc[time_df['Timesteps'] == time2]
                h_time_holder_int = int(h_time_holder_int.index.values)

                """"
                    THRESHOLD:
                    Since threshold depends on data from the prior two weeks, h > 1 is there to ensure that there's data to deduce threshold from.

                    Likely to need further refinement at a later stage
                    might miss it if there's a leak in the first week or two? --> may be some try - exception-python tricks to help -> do same thing on iter_res, but find out how works later
                    renews each iteration as res' size increases with iter_res appended
                    if 30% larger than last two weeks' maximum
                    may be able to detect peak of the day the leak occurs, 
                    but need comparison with similar time of day also for higher accuracy
                    will only trigger at say 10-ish o'clock at peak water demand so to speak.
                    but leak can occur any point during the day

                    05/04: probably have to create multiple thresholds in order to search for specific types of leaks.
            
                """
 
                #threshold = []

                if h > 1:
                    #print(f'this is res.max etc: {res[-num_days*timesteps_in_day:].max()}')                    
                    #threshold = res[-num_days*timesteps_in_day:].max()*1.6
                    # for col in (X_per_sensor.columns):
                    #     #print(X_per_sensor[col][:-14*288].max())
                    
                    #     threshold.append(X_per_sensor[col][:-14*288].max())
                    #     np.savetxt(f'threshold_{dataset}', np.array([threshold]))
                    threshold = res[-num_days*timesteps_in_day:].max()*100000
                    leak_counter = 0
                    leak_col = []

                    for it in range(len(iter_res)):
                        if iter_res[it] > threshold: 
                            leak_timestep_iter_res = it
                            leak_found = True
                            break

                    # for i, col in enumerate(X_test_copy_filled.columns):
                    #     #print(threshold)

                    #     #print(f'{col}, {X_test_copy_filled[col].max()}')
                        
                    #     if X_test_copy_filled[col].max() > threshold[i]*threshold_number:
                    #         leak_counter = 1 + leak_counter
                    #         leak_col.append(col)

                    #         if leak_counter > 5:

                    #             it = X_test_copy_filled[col].idxmax()
                    #             it = time_df.loc[time_df['Timesteps'] == it]
                    #             leak_timestep_iter_res = int(it.index.values)
                    #             leak_found = True

                    #             break
                

                X_per_sensor = pd.concat([X_per_sensor, X_test_copy_filled])
                X_per_sensor.to_csv(f'X_per_sensor_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dataset{dataset}_1.csv')

                        #iterates through the iteration_results variable to see if any occurence that exceeds the threshold has appeared
                    # for it in range(len(iter_res)):
                    #     if iter_res[it] > threshold: # or iter_res[:it].min() > lower_threshold  #or iter_res.min() > lower_threshold:
                    #         leak_timestep_iter_res = it
                    #         leak_found = True
                    #         break
                
                #concatenate to res
                res = np.concatenate([res, np.array(iter_res)])
                np.savetxt(f'res_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dataset{dataset}_1.csv', res, delimiter= ',')

                print(f'Leak found: {leak_found}')

                if leak_found:
                    #numerical value of the timestep in which leak_found became set to True
                    
                    leak_timestep = leak_timestep_iter_res + h_time_holder_int 

                    # plt.plot(res)
                    # plt.show()
                    # plt.clf()

                            #saves a reconstruction plot with the learning and validation curves, similar as to shown before in the semester
                    #folder_recon_plot_path = folders[0]
                    #plot_reconstruction(res, model_loss, epochs, leak_df, time_df, folder_recon_plot_path=folder_recon_plot_path, dataset=dataset, iteration = kk)
                    break
                else:
                    leak_timestep = 0

                h = h + 1 

                    #can be un-commented to see preliminary results through the year
                # plt.plot(res)
                # plt.show()
                # plt.clf()

                #similar as above, but comparison with x_test input and the MSE 
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


                #various model configuration information that's saved for later inspection
            modeltxt = (f'Epochs: {epochs}, Learning rate: {learning_rate}, Batch_size: {batch_size}, Optimizer: {optimizer}, Loss function: {loss}, Dataset = {dataset} \n\n Model: {model}') 
            modeltxt2 = (f'Training time ended at: {train_time}, validation time ended at: {vald_time}')
            modeltxt_total = modeltxt + modeltxt2
            txt_file = open(folders[3]+'\_'+str(kk)+'.txt', "w")
            txt_file.write(modeltxt_total)
            txt_file.close()

            #fills the time in which training and validation occured with 0s with goal of maintaining the datatimeindex on the x-axis in graph
            zero_array = np.zeros(shape = (vald_time_int, 1))
            res = np.append(zero_array, res)

            #dataframe of the results to be plotted in the leak_detection plot.

            """
                make into separate plotting function at some point
            """
            # res_df = pd.DataFrame(index = time_df['Timesteps'][:len(res)], columns=['res'], data = res)  #0s during the time spent training and validating
            # fig = plt.figure(1, figsize = (14,6)) #gjorde dette en endring? fikk lagret en liten plot før

            # colors = sns.color_palette("rocket", 10)
            # for i, leak in enumerate(leak_df['leak']): 
            #     if leak == dataset:  
            #         #start and stop point of the given dataset's leak      
            #         plt.plot(time_df.iloc[leak_df['start'][i]], 0, marker = 'X', color = colors[i], label = dataset, alpha = 0.8)
            #         plt.plot(time_df.iloc[leak_df['stop'][i]], 0, marker = 's', color = colors[i], label = '', alpha = 0.8)

            # plt.plot(time_df.iloc[leak_timestep], 0, marker = 'x', color = 'red', label = 'Detected leak', markersize=14)
            # plt.plot(train_time, 0, marker = 'o', color = 'darkblue', label = 'end of start and validation time', alpha = 0.8)
            # plt.plot(vald_time, 0, marker = 'o', color = 'darkblue', label = '', alpha = 0.8)

            # plt.plot(res_df, color = 'darkgreen', label = 'Error')
            # #plt.xlabel('')
            # plt.ylabel('Reconstruction error')
            # plt.legend()
            #plt.show()
                #saved to the appropriate folder                
            #plt.savefig(folders[5]+ "\_"+ str(dataset) + "_" + (str(kk)))
            #plt.clf()
        
            for iii, leak in enumerate(leak_df['leak']): 
                if leak == dataset:
                    dataset_nr = iii

                #creates a dataframe in which to save results
            super_results_holder = pd.DataFrame(index = [':)'], columns = ['Model iteration','Dataset','Time detected leak', 'Time true leak', 'Difference','Accumulated difference', 'Threshold'], data = 0)
            super_results_holder['Model iteration'] = kk
            super_results_holder['Dataset'] = dataset
            super_results_holder['Time detected leak'] = leak_timestep
            #super_results_holder['Threshold'] = threshold_number
            if no_leak_found:
                super_results_holder['Time detected leak'] == -9999

            super_results_holder['Time true leak'] = leak_df['start'][dataset_nr]

            if no_leak_found == False:
                acc_diff = abs(leak_timestep - leak_df['start'][dataset_nr]) + acc_diff

            super_results_holder['Difference'] = (abs(leak_timestep - leak_df['start'][dataset_nr]))
            super_results_holder['Accumulated difference'] = acc_diff
            super_results_holder['Running time 1 DS'] = str(datetime.now()-startTime)

            super_results = pd.concat([super_results, super_results_holder])

            print(super_results)
            super_results.to_csv(folders[2] +'\_' + (str(kk)) + '.csv')












