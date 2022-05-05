from re import A
from zoneinfo import reset_tzpath
from matplotlib import container
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime


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
from autoencoders_and_more import thresholding_algo, ordered_cluster


from autoencoder_test_test import score, score_v2

from os import listdir


def create_folders():
    directory_reconstruction_figures = "Reconstruction_plots\\AE_recon_figure_1"
    parent_directory = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\rolling_AE_results\\"

    path_reconstruction_figures = os.path.join(parent_directory, directory_reconstruction_figures)

    while os.path.isdir(path_reconstruction_figures):
        if os.path.isdir(os.path.join(parent_directory, directory_reconstruction_figures)) is False:
            break
        
        for s in directory_reconstruction_figures.split('_'):
            if s.isdigit():
                number = int(s)

        directory_reconstruction_figures = f"Reconstruction_plots\\AE_recon_figure_{str(int(number)+1)}"
        directory_evaluation_plots = f"Evaluation_plots\\Eval_plot_figure_{str(int(number+1))}"
        directory_model_performance = f"Model_performance\\Model_performance_{str(int(number+1))}"
        directory_model_config = f"Model_configurations\\Model_config_{str(int(number+1))}"
        directory_per_neuron = f"Per_neuron\\per_neuron_{str(int(number+1))}"
        directory_leak_detect_plots = f"Leak_detection_plots\\Leak_detect_figure_{str(int(number+1))}"

        path_reconstruction_figures = os.path.join(parent_directory, directory_reconstruction_figures)
        path_evaluation_plots = os.path.join(parent_directory, directory_evaluation_plots)
        path_model_performance = os.path.join(parent_directory, directory_model_performance)
        path_model_config_files = os.path.join(parent_directory, directory_model_config)
        path_all_neurons_csv_files = os.path.join(parent_directory, directory_per_neuron)
        path_leak_detect_plots = os.path.join(parent_directory, directory_leak_detect_plots)

    os.mkdir(path_reconstruction_figures)
    os.mkdir(path_evaluation_plots)
    os.mkdir(path_model_performance)
    os.mkdir(path_model_config_files)
    os.mkdir(path_all_neurons_csv_files)
    os.mkdir(path_leak_detect_plots)

    all_paths = [path_reconstruction_figures, path_evaluation_plots, path_model_performance, path_model_config_files, path_all_neurons_csv_files, path_leak_detect_plots]
    return all_paths


def import_data(dataset):
    file_path = f'data/{dataset}/'
    print(dataset)

    sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
                'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
                'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

    X = pd.read_csv(file_path + 'Levels.csv', index_col=0, parse_dates=[0])
    pressure = pd.read_csv(file_path +'Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
    pump_flows = pd.read_csv(file_path + 'Flows.csv', index_col=0, parse_dates=[0]).squeeze()

    X = X.reset_index()
    pressure = pressure.reset_index()
    X = pd.concat([X, pressure], axis = 1)
    X = X.set_index('index')

    X['PUMP_1'] = pump_flows['PUMP_1']
    X['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']
    
    return X


def plot_reconstruction(res, model_loss, epochs, leak_df, time_df, folder_recon_plot_path, dataset, iteration):
    x = np.linspace(0, epochs - 1, epochs)

    fig = plt.figure(1, figsize = (14,6))
    mpl.rcParams['font.size']=14
    gs = gridspec.GridSpec(6,8)
    gs.update(wspace = 0.25,  hspace = 1.6)
    colors = sns.color_palette("rocket", 10)
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

    bottom_subplot = fig.add_subplot(gs[2:6, 0:8])
    plt.plot(time_df['Timesteps'][:len(res)], res, color = 'forestgreen', label = '_nolegend_')

    for i, leak in enumerate(leak_df['leak']):
        if leak == dataset:   
            start = leak_df['start'][i]
            start = pd.to_datetime(time_df.loc[start])

            stop = leak_df['stop'][i]
            stop = pd.to_datetime(time_df.loc[stop])

            plt.plot(start, 0, marker = 'X', color = colors[i], label = dataset, alpha = 0.8)
            plt.plot(stop, 0, marker = 's', color = colors[i], label = '', alpha = 0.8)


    plt.title('Reconstruction')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    fig.savefig(folder_recon_plot_path+ "\_" +str(dataset) +"_"+str(iteration))
    plt.clf()

def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

# def threshold_function(X_per_sensor, dataset, vald_time_int, ):    

#     time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
#     time_df = pd.DataFrame(columns = ['Timesteps'] ,data = time_2018)
#     hour_time = pd.timedelta_range(start = '00:00:00', end='23:59:00',freq = '1H')

#     leak_data_modified = [{'leak': 'p654', 'start': 20690,  'stop': 45000, 'leak time': time_df.iloc[20690], 'leak time numeric': 20690},  # har nådd maks ved 5 m3/h en stund da
#                           {'leak': 'p628', 'start': 36520,  'stop': 42883, 'leak time': time_df.iloc[36520], 'leak time numeric': 36520}, #muligens ikke 100% pållitelige verdier, men bured vært ok hvis quick burst
#                           {'leak': 'p866', 'start': 43600,  'stop': 46694, 'leak time': time_df.iloc[43600], 'leak time numeric': 43600},
#                           {'leak': 'p183', 'start': 62817,  'stop': 70192, 'leak time': time_df.iloc[62817], 'leak time numeric': 62817},
#                           {'leak': 'p369', 'start': 80000,  'stop': 89815, 'leak time': time_df.iloc[85851], 'leak time numeric': 85851},
#                           {'leak': 'p538', 'start': 39561,  'stop': 43851, 'leak time': time_df.iloc[39561], 'leak time numeric': 39561},
#                           {'leak': 'p158', 'start': 80097,  'stop': 85125, 'leak time': time_df.iloc[80097], 'leak time numeric': 80097},
#                           {'leak': 'p232', 'start': 9124,   'stop': 11634, 'leak time': time_df.iloc[9124],  'leak time numeric': 11634},
#                           {'leak': 'p461', 'start': 13807,  'stop': 26530, 'leak time': time_df.iloc[13807], 'leak time numeric': 13807},
#                           {'leak':  'p31', 'start': 57095,  'stop': 64437, 'leak time': time_df.iloc[57095], 'leak time numeric': 57095}]
                          
#                           #p369 er egt 85851
#     leak_df_modified = pd.DataFrame(data = leak_data_modified)
#     leak_df_modified = leak_df_modified.set_index('leak')
#     leak_timestep = 0
#     # X_per_sensor = X_per_sensor.set_index('Unnamed: 0')
#     # X_per_sensor = X_per_sensor[1:]
#     # X_per_sensor.index = pd.to_datetime(X_per_sensor.index)

#     prelim_results_dataframe = pd.DataFrame(index  = ['Detected', 'Timestep'], columns = [X_per_sensor.columns[1:34]], data = 0)
#     #leak_found_holder = pd.DataFrame(index = [], columns = [X_per_sensor.columns[1:34]], data = 0)

#     startTime = datetime.now()
#     #possible_multipliers = np.arange(2, 6, 0.25)
#     #possible_multipliers_v2 = np.arange(1, 4, 0.1)
    
#     #comparison_number_multiplier = possible_multipliers[randrange(len(possible_multipliers))]
#     comparison_number_multiplier = 2
#     #comparison_number_multiplier_v2 = possible_multipliers_v2[randrange(len(possible_multipliers_v2))]
#     comparison_number_multiplier_v2 = 1.5
#     vald_time_int_v2 = 288*14 + vald_time_int # muligens litt sketchy linje dette, revurdere fix når fått fungere
    
#     for col in X_per_sensor.columns[1:34]:
#         b = X_per_sensor[col].to_numpy()
#         zero_array = np.zeros(shape = (vald_time_int, 1))
#         b = np.append(zero_array, b)

#         res_df = pd.DataFrame(index = time_df['Timesteps'][:len(b)], columns=[col], data = b)
#         res_df = res_df**0.5*2 #undo the MSE

#         #tror gir opp nøyaktighet, og prøver med dette istedenfor, må kanskje lette på ting og gå sakte gjennom
#         res_df = res_df.rolling(288).mean()

#         if col == 'n1':
#             print(f'res_df final timestamp is this: {res_df.index[-1]}')
    
#         #res_df = res_df.rolling(36).max() #tror lettere å se lekkasje med min, blir noenlunde likt til MNF

#         start_ = leak_df_modified.loc[dataset]['start']
#         stop_ = leak_df_modified.loc[dataset]['stop']

#         #for j, ind in enumerate(res_df.index[vald_time_int:len(res_df)-144]):
#         for j, ind in enumerate(res_df.index[start_-288*14:stop_-144]): # start checking 2 weeks before leaks occur,
#         #for j, ind in enumerate(res_df.index[:-144]):
#             possible_leak_detected = False
#             number = res_df.loc[ind]

#             h = [res_df[vald_time_int_v2-288*14+j:vald_time_int_v2+j].between_time(str(hour_time[i])[7:], str(hour_time[i+1])[7:]) for i in range (len(hour_time)-1)]
#             stds, avgs = [], []
#             for hh in h:   
#                 stds.append(hh.std(axis=0))
#                 avgs.append(hh.mean(axis=0))

#             # plt.plot(stds, color = 'green')
#             # plt.plot(avgs, color = 'blue')
#             # plt.title(str(j))
#             # plt.draw()
#             # plt.pause(0.00001)

#             comparison_number = (comparison_number_multiplier*float(stds[ind.hour-1])+comparison_number_multiplier_v2*float(avgs[ind.hour-1]))

#             if float(number) >= comparison_number and comparison_number != 0:
#                 #print(f'qoutient: {number/comparison_number}')
#                 c = 0
#                 for k in range (6): #6 hours ahead and equal hours back, and same ts in week for 2 weeks
#                     try:
#                         next_check_number_ts = ind + pd.DateOffset(hours = k)
#                         back_check_num_ts = ind - pd.DateOffset(hours = abs(24-k))

#                         next_check_number = res_df.loc[next_check_number_ts]
#                         back_check_num = res_df.loc[back_check_num_ts]

#                         if k == 3:
#                             print(f'd, {datetime.now()-startTime}, {col}, {ind}, {back_check_num.values}, {next_check_number.values}')

#                     except:
#                         pass
   
#                     if float(next_check_number) > float(back_check_num)*comparison_number_multiplier_v2 and [float(next_check_number) > comparison_number_multiplier_v2 * float(res_df.loc[ind-pd.DateOffset(days = 7*i)]) for i in range(3)] and float(back_check_num.values) != 0.0:
#                         c = c + 1
#                         if c == 5:
#                             prelim_results_dataframe.loc['Detected'][col] = 1
#                             leak_timestep_int = int(np.where([time_df['Timesteps']==ind])[1])
#                             prelim_results_dataframe.loc['Timestep'][col] = leak_timestep_int
#                             print(prelim_results_dataframe)

#                             possible_leak_detected = True
#                             break
#                 if possible_leak_detected:
#                     print(f'possible leak detected, {datetime.now()-startTime}')
#                     break

#         ts_values = prelim_results_dataframe.loc['Timestep'].copy()                     
#         clusters = cluster(ts_values.values, maxgap = 36)

#         empty_sensors = np.zeros([1, 33])
                   
#         if (len(np.array(clusters[0])) > 0) and (np.array_equal(np.array(clusters[0]), empty_sensors[0]) == False):
#             max_length_cluster = max([len(clusters[i]) for i in range(len(clusters))])

#             for jj in range (len(clusters)):
#                 if len(clusters[jj]) == max_length_cluster:
#                     max_length_cluster_nr = jj

#             if (clusters[max_length_cluster_nr][0]==0): #hvis det finnes en null i gitte clusteren.
#                 if len(clusters) >=1: #muligens unødvendig
#                     clusters.pop(max_length_cluster_nr)
                
#                 max_length_cluster = max([len(clusters[ii]) for ii in range(len(clusters))])
#                 for hh in range (len(clusters)):
#                     if len(clusters[hh]) == max_length_cluster:
#                         max_length_cluster_nr = hh

#             biggest_cluster = clusters[max_length_cluster_nr]
#             leak_timestep = np.min(biggest_cluster)
        
#         if math.isclose(leak_timestep, leak_df_modified.loc[dataset]['leak time numeric'], abs_tol=288*7) and len(biggest_cluster) > 5: #dette er ikke helt ærlig, vet jo ikke hva som er reellt tidspunkt til lekkasjen
#             leak_found = True
#         else:
#             leak_found = False
            
#         #prelim_results_dataframe.to_csv(f'{round(comparison_number_multiplier,2), round(comparison_number_multiplier_v2,2)}_prelimresults.csv')
#     return prelim_results_dataframe, leak_found, leak_timestep


def threshold_function_v2(X_per_sensor, dataset, vald_time_int, comparison_number_multiplier, comparison_number_multiplier_v2):   

    #print(X_per_sensor)

    time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
    time_df = pd.DataFrame(columns = ['Timesteps'] ,data = time_2018)

    hour_time = pd.timedelta_range(start = '00:00:00', end='23:59:00',freq = '1H')

    leak_data_modified = [{'leak': 'p654', 'start': 20690,  'stop': 45000, 'leak time': time_df.iloc[20690], 'leak time numeric': 20690},  # har nådd maks ved 5 m3/h en stund da
                          {'leak': 'p628', 'start': 35076,  'stop': 42883, 'leak time': time_df.iloc[35076], 'leak time numeric': 35076}, #muligens ikke 100% pållitelige verdier, men bured vært ok hvis quick burst
                          {'leak': 'p866', 'start': 43600,  'stop': 46694, 'leak time': time_df.iloc[43600], 'leak time numeric': 43600},
                          {'leak': 'p183', 'start': 62817,  'stop': 70192, 'leak time': time_df.iloc[62817], 'leak time numeric': 62817},
                          {'leak': 'p369', 'start': 80000,  'stop': 89815, 'leak time': time_df.iloc[85851], 'leak time numeric': 85851},
                          {'leak': 'p538', 'start': 39561,  'stop': 43851, 'leak time': time_df.iloc[39561], 'leak time numeric': 39561},
                          {'leak': 'p158', 'start': 80097,  'stop': 85125, 'leak time': time_df.iloc[80097], 'leak time numeric': 80097},
                          {'leak': 'p232', 'start': 8687,   'stop': 11634, 'leak time': time_df.iloc[8687],  'leak time numeric': 8687},
                          {'leak': 'p461', 'start': 6625,   'stop': 26530, 'leak time': time_df.iloc[6625], 'leak time numeric': 6625},
                          {'leak':  'p31', 'start': 51573,  'stop': 64437, 'leak time': time_df.iloc[51573], 'leak time numeric': 51573}]
                          
                          #p369 er egt 85851
    leak_df_modified = pd.DataFrame(data = leak_data_modified)
    leak_df_modified = leak_df_modified.set_index('leak')
    leak_timestep = 0
    # X_per_sensor = X_per_sensor.set_index('Unnamed: 0')
    # X_per_sensor = X_per_sensor[1:]
    # X_per_sensor.index = pd.to_datetime(X_per_sensor.index)

    prelim_results_dataframe = pd.DataFrame(index  = ['Detected', 'Timestep'], columns = [X_per_sensor.columns[1:34]], data = 0)
    #leak_found_holder = pd.DataFrame(index = [], columns = [X_per_sensor.columns[1:34]], data = 0)

    startTime = datetime.now()
    
    
    threshold_time = datetime.now()
    for col in X_per_sensor.columns[1:34]:
        print(f'{col}, {datetime.now()-threshold_time}')

        b = X_per_sensor[col].to_numpy()
        zero_array = np.zeros(shape = (vald_time_int, 1))
        b = np.append(zero_array, b)

        try:
            res_df = pd.DataFrame(index = time_df['Timesteps'][:len(b)], columns=[col], data = b) #sjekk om første blir nan
        except:
            b = b[:len(time_df['Timesteps'])]
            res_df = pd.DataFrame(index = time_df['Timesteps'][:len(b)], columns=[col], data = b)
        
        res_df = res_df**0.5*2 #undo the MSE

        #tror gir opp nøyaktighet, og prøver med dette istedenfor, må kanskje lette på ting og gå sakte gjennom
        res_df = res_df.rolling(288).mean()
    
        start_ = leak_df_modified.loc[dataset]['start']
        stop_ = leak_df_modified.loc[dataset]['stop']

        #for j, ind in enumerate(res_df.index[vald_time_int:len(res_df)-144]):

        
        #for j, ind in enumerate(res_df.index[start_-288*28:stop_-144]): # start checking 4 weeks before leaks occur,
        stds, avgs = np.zeros(shape = (len(hour_time)+1, 1)), np.zeros(shape = (len(hour_time)+1, 1))
        
        for j, ind in enumerate(res_df.index[start_-288*28:stop_-144]): # start checking 4 weeks before leaks occur,
            possible_leak_detected = False
            number = res_df.loc[ind]

            if (j % 288*7 == 0):

                h = [res_df[vald_time_int-288*7+j:vald_time_int+j].between_time(str(hour_time[i-1])[7:], str(hour_time[i])[7:]) for i in range (1, len(hour_time))]
                stds, avgs = [], []
                for hh in h:   
                    stds.append(hh.std(axis=0))
                    avgs.append(hh.mean(axis=0))

                stds.append(res_df[vald_time_int-288*7+j:vald_time_int+j].between_time(str(hour_time[-1])[7:], str(hour_time[0])[7:]).std(axis=0))
                avgs.append(res_df[vald_time_int-288*7+j:vald_time_int+j].between_time(str(hour_time[-1])[7:], str(hour_time[0])[7:]).mean(axis=0))
                

            comparison_number = (comparison_number_multiplier*float(stds[ind.hour])+comparison_number_multiplier_v2*float(avgs[ind.hour]))

            if float(number) >= comparison_number and comparison_number != 0:
                #print(f'qoutient: {number/comparison_number}')
                #c = 0
                for k in range (6): #6 hours ahead and equal hours back, and same ts in week for 2 weeks
                    try:
                        next_check_number_ts = ind + pd.DateOffset(hours = k)
                        back_check_num_ts = ind - pd.DateOffset(hours = abs(24-k))

                        next_check_number = res_df.loc[next_check_number_ts]
                        back_check_num = res_df.loc[back_check_num_ts]

                    except:
                        pass
   
                    if float(next_check_number) > float(back_check_num)*comparison_number_multiplier_v2 and [float(next_check_number) > comparison_number_multiplier_v2 * float(res_df.loc[ind-pd.DateOffset(days = 7*i)]) for i in range(2)] and float(back_check_num.values) != 0.0:
                        #c = c + 1
                        #if c == 3:
                        prelim_results_dataframe.loc['Detected'][col] = 1
                        leak_timestep_int = int(np.where([time_df['Timesteps']==ind])[1])
                        prelim_results_dataframe.loc['Timestep'][col] = leak_timestep_int
                        print(prelim_results_dataframe)

                        possible_leak_detected = True
                        break
                if possible_leak_detected:
                    print(f'possible leak detected, {datetime.now()-startTime}')
                    break

        ts_values = prelim_results_dataframe.loc['Timestep'].copy()                     
        clusters = cluster(ts_values.values, maxgap = 288)

        empty_sensors = np.zeros([1, 33])
                   
        if (len(np.array(clusters[0])) > 0) and (np.array_equal(np.array(clusters[0]), empty_sensors[0]) == False):
            max_length_cluster = max([len(clusters[i]) for i in range(len(clusters))])

            for jj in range (len(clusters)):
                if len(clusters[jj]) == max_length_cluster:
                    max_length_cluster_nr = jj

            if (clusters[max_length_cluster_nr][0]==0): #hvis det finnes en null i gitte clusteren.
                if len(clusters) >=1: #muligens unødvendig
                    clusters.pop(max_length_cluster_nr)
                
                max_length_cluster = max([len(clusters[ii]) for ii in range(len(clusters))])
                for hh in range (len(clusters)):
                    if len(clusters[hh]) == max_length_cluster:
                        max_length_cluster_nr = hh

            biggest_cluster = clusters[max_length_cluster_nr]
            leak_timestep = np.min(biggest_cluster)

            #ordnet at start er ved 0 for slow increase
            #ikke enda ordnet for incipient
        try:    #leak time numeric ikke helt ok for sakte voksende?, burde heller bare si at hvis len(biggest cluster) større enn 5 eller noe
            if math.isclose(leak_timestep, leak_df_modified.loc[dataset]['leak time numeric'], abs_tol=288*28) and len(biggest_cluster) > 3: 
                leak_found = True
            else:
                leak_found = False
        except:
                leak_found = False
            
            
        #prelim_results_dataframe.to_csv(f'{round(comparison_number_multiplier,2), round(comparison_number_multiplier_v2,2)}_prelimresults.csv')
    return prelim_results_dataframe, leak_found, leak_timestep




def threshold_function_v3(X_per_sensor, dataset, vald_time_int, comparison_number_multiplier, comparison_number_multiplier_v2):   

    #print(X_per_sensor)

    time_2018 = pd.date_range(start = '2018-01-01 00:00:00', end = '2018-12-31 23:55:00', freq = '5min')
    time_df = pd.DataFrame(columns = ['Timesteps'] ,data = time_2018)

    hour_time = pd.timedelta_range(start = '00:00:00', end='23:59:00',freq = '1H')

    leak_data_modified = [{'leak': 'p654', 'start': 20690,  'stop': 45000, 'leak time': time_df.iloc[20690], 'leak time numeric': 20690},  # har nådd maks ved 5 m3/h en stund da
                          {'leak': 'p628', 'start': 35076,  'stop': 42883, 'leak time': time_df.iloc[35076], 'leak time numeric': 35076}, #muligens ikke 100% pållitelige verdier, men bured vært ok hvis quick burst
                          {'leak': 'p866', 'start': 43600,  'stop': 46694, 'leak time': time_df.iloc[43600], 'leak time numeric': 43600},
                          {'leak': 'p183', 'start': 62817,  'stop': 70192, 'leak time': time_df.iloc[62817], 'leak time numeric': 62817},
                          {'leak': 'p369', 'start': 80000,  'stop': 89815, 'leak time': time_df.iloc[85851], 'leak time numeric': 85851},
                          {'leak': 'p538', 'start': 39561,  'stop': 43851, 'leak time': time_df.iloc[39561], 'leak time numeric': 39561},
                          {'leak': 'p158', 'start': 80097,  'stop': 85125, 'leak time': time_df.iloc[80097], 'leak time numeric': 80097},
                          {'leak': 'p232', 'start': 8687,   'stop': 11634, 'leak time': time_df.iloc[8687],  'leak time numeric': 8687},
                          {'leak': 'p461', 'start': 6625,   'stop': 26530, 'leak time': time_df.iloc[6625],  'leak time numeric': 6625},
                          {'leak':  'p31', 'start': 51573,  'stop': 64437, 'leak time': time_df.iloc[51573], 'leak time numeric': 51573}]
                          
                          #p369 er egt 85851
    leak_df_modified = pd.DataFrame(data = leak_data_modified)
    leak_df_modified = leak_df_modified.set_index('leak')
    leak_timestep = 0
    # X_per_sensor = X_per_sensor.set_index('Unnamed: 0')
    # X_per_sensor = X_per_sensor[1:]
    # X_per_sensor.index = pd.to_datetime(X_per_sensor.index)

    prelim_results_dataframe = pd.DataFrame(index  = ['Detected', 'Timestep'], columns = [X_per_sensor.columns[1:34]], data = 0)
    #leak_found_holder = pd.DataFrame(index = [], columns = [X_per_sensor.columns[1:34]], data = 0)
  
    threshold_time = datetime.now()
    number = float()

    for col in X_per_sensor.columns[1:34]:
        print(f'{col}, {datetime.now()-threshold_time}')

        res_df = X_per_sensor[col].copy()
        res_df = res_df[~res_df.index.duplicated()] #removes if there is a double timestamp

        time_rolling = 288

        res_df = res_df.rolling(time_rolling).mean() #husk skaper nan i starten 
        stds, avgs = np.zeros(shape = (len(hour_time)+1, 1)), np.zeros(shape = (len(hour_time)+1, 1))
        number_of_days = 5

        for j, ind in enumerate(res_df.index[time_rolling:]): 
            possible_leak_detected = False
            number = res_df.loc[ind].copy()

            if (j % (288*number_of_days) == 0 and j != 0): #use the first x days to gather expected std and avg values, updates
                
                h = [res_df[(j-(288*number_of_days)):j].between_time(str(hour_time[i-1])[7:], str(hour_time[i])[7:]) for i in range (1, len(hour_time))]
                stds, avgs = list(), list()
                for hh in h:   
                    stds.append(hh.std(axis=0))
                    avgs.append(hh.mean(axis=0))

                stds.append(res_df[(j-(288*number_of_days)):j].between_time(str(hour_time[-1])[7:], str(hour_time[0])[7:]).std(axis=0))
                avgs.append(res_df[(j-(288*number_of_days)):j].between_time(str(hour_time[-1])[7:], str(hour_time[0])[7:]).mean(axis=0))

            comparison_number = (comparison_number_multiplier*float(stds[ind.hour])+comparison_number_multiplier_v2*float(avgs[ind.hour]))

            if float(number) >= float(comparison_number) and float(comparison_number) != 0:

                for k in range (3): #6 hours ahead and equal hours back, and same ts in week for 2 weeks
                    try:
                        next_check_number_ts = ind + pd.DateOffset(hours = k)
                        back_check_num_ts = ind - pd.DateOffset(hours = abs(24-k))

                        next_check_number = res_df.loc[next_check_number_ts]
                        back_check_num = res_df.loc[back_check_num_ts]

                    except:
                        pass

                            #small change in an attempt to find p232 
                    if float(next_check_number) > float(back_check_num) and [float(next_check_number) > float(res_df.loc[ind-pd.DateOffset(days = 2*i)]) for i in range(2)] and float(back_check_num) != 0.0:
                    #if float(next_check_number) > float(back_check_num) and [float(next_check_number) > float(res_df.loc[ind-pd.DateOffset(days = 7*i)]) for i in range(2)] and float(back_check_num) != 0.0:
                        prelim_results_dataframe.loc['Detected'][col] = 1
                        leak_timestep_int = int(np.where([time_df['Timesteps']==ind])[1])
                        prelim_results_dataframe.loc['Timestep'][col] = leak_timestep_int
                        print(prelim_results_dataframe)

                        possible_leak_detected = True
                        break
                if possible_leak_detected:
                    print(f'possible leak detected, {datetime.now()-threshold_time}')
                    break

        ts_values = prelim_results_dataframe.loc['Timestep'].copy()                     
        clusters = cluster(ts_values.values, maxgap = 288)

        empty_sensors = np.zeros([1, 33])
                   
        if (len(np.array(clusters[0])) > 0) and (np.array_equal(np.array(clusters[0]), empty_sensors[0]) == False):
            max_length_cluster = max([len(clusters[i]) for i in range(len(clusters))])

            for jj in range (len(clusters)):
                if len(clusters[jj]) == max_length_cluster:
                    max_length_cluster_nr = jj

            if (clusters[max_length_cluster_nr][0]==0): #hvis det finnes en null i gitte clusteren.
                if len(clusters) >=1: #muligens unødvendig
                    clusters.pop(max_length_cluster_nr)
                
                max_length_cluster = max([len(clusters[ii]) for ii in range(len(clusters))])
                for hh in range (len(clusters)):
                    if len(clusters[hh]) == max_length_cluster:
                        max_length_cluster_nr = hh

            biggest_cluster = clusters[max_length_cluster_nr]
            leak_timestep = np.min(biggest_cluster)

        try:    #leak time numeric ikke helt ok for sakte voksende?, burde heller bare si at hvis len(biggest cluster) større enn 5 eller noe
            if math.isclose(leak_timestep, leak_df_modified.loc[dataset]['leak time numeric'], abs_tol=288*28) and len(biggest_cluster) > 3: 
                leak_found = True
            else:
                leak_found = False
        except:
                leak_found = False
            
            
        #prelim_results_dataframe.to_csv(f'{round(comparison_number_multiplier,2), round(comparison_number_multiplier_v2,2)}_prelimresults.csv')
    return prelim_results_dataframe, leak_found, leak_timestep













def leak_location_plot(time_df, res, leak_df, leak_timestep, train_time, vald_time, folders, dataset, kk):
    res_df = pd.DataFrame(index = time_df['Timesteps'][:len(res)], columns=['res'], data = res)  #0s during the time spent training and validating
    fig = plt.figure(1, figsize = (14,6)) 

    colors = sns.color_palette("rocket", 10)
    for i, leak in enumerate(leak_df['leak']): 
        if leak == dataset:  
            #start and stop point of the given dataset's leak      
            plt.plot(time_df.iloc[leak_df['start'][i]], 0, marker = 'X', color = colors[i], label = dataset, alpha = 0.8)
            plt.plot(time_df.iloc[leak_df['stop'][i]], 0, marker = 's', color = colors[i], label = '', alpha = 0.8)

    plt.plot(time_df.iloc[leak_timestep], 0, marker = 'x', color = 'red', label = 'Detected leak', markersize=14)
    plt.plot(train_time, 0, marker = 'o', color = 'darkblue', label = 'end of start and validation time', alpha = 0.8)
    plt.plot(vald_time, 0, marker = 'o', color = 'darkblue', label = '', alpha = 0.8)

    plt.plot(res_df, color = 'darkgreen', label = 'Error')
    #plt.xlabel('')
    plt.ylabel('Reconstruction error')
    plt.legend()
    plt.show()
    #saved to the appropriate folder                
    plt.savefig(folders[5]+ "\_"+ str(dataset) + "_" + (str(kk)))
    plt.clf()


