from re import A
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


