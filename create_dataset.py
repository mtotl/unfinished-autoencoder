
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

from statsmodels.tsa.seasonal import STL


data_file_path = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\data\\"

data_name = [f for f in listdir(data_file_path)]
sensor_columns = ['n1', 'n4', 'n31', 'n54', 'n469', 'n415', 'n519', 'n495', 'n105', 'n114', 'n516', 'n549',
                    'n332', 'n506', 'n188', 'n410', 'n429', 'n458', 'n613', 'n342', 'n163', 'n216', 'n229', 
                    'n644', 'n636', 'n679', 'n288', 'n726', 'n286', 'n722', 'n752', 'n769', 'n740'] 

#based on file 2018_leakages.csv
burst_and_repaired_leaks = ['p158','p183','p232','p369','p461','p538','p628','p673','p866']
half_burst_and_repaired_leaks = ['p461','p538','p628','p673','p866','p232']
slow_increase_leaks = ['p31','p461']
neverending_leaks = ['p257', 'p427', 'p654', 'p810']

all_data = []
for dn in data_name:
    if dn in half_burst_and_repaired_leaks or dn == 'noleak':
        print(dn)

        data = pd.read_csv(data_file_path +dn+'\Levels.csv', index_col=0, parse_dates=[0])
        pressure = pd.read_csv(data_file_path +dn+'\Pressures.csv', index_col=0, parse_dates=[0], usecols=sensor_columns)
        pump_flows = pd.read_csv(data_file_path +dn+'\Flows.csv', index_col=0, parse_dates=[0]).squeeze()

        data = data.reset_index()
        pressure = pressure.reset_index()
        data = pd.concat([data, pressure], axis = 1)
        data = data.set_index('index')

        data['PUMP_1'] = pump_flows['PUMP_1']
        data['Demand'] = pump_flows['p227']+pump_flows['p235']-pump_flows['PUMP_1']

        all_data.append(data)

    # if i == 2:
    #     break
    # i = i+1

#sette sammen et knippe datasett og ta gjennomsnitt av verdiene deres. forløpelig, evt finnes bedre måter? lagre til en csv for å
#jobbe videre i andre skript


# from test import get_coordinates

# a = get_coordinates()
# print(a)


z, n, m = np.shape(all_data)
print(z,n,m)

no_leak = all_data[0][:][:]
print(no_leak)

summarized_data = np.zeros([n, m])

for i in range(1, z):
    summarized_data = all_data[i][:][:] + summarized_data

new_dataset = summarized_data-no_leak*(len(half_burst_and_repaired_leaks)-1)

new_dataset = pd.DataFrame(data =new_dataset)

no_leak  = pd.DataFrame(data = no_leak)

# for col in new_dataset.columns:
#     plt.plot(new_dataset[col], label = col)
#     plt.legend()
#     plt.show()

for col in new_dataset.columns: #reconsider, apply 
    a = STL(new_dataset[col], period = 288, robust= True)
    a_res = a.fit()
    a_seasonal = a_res.seasonal
    new_dataset[col] = new_dataset[col] - a_seasonal

# p232withoutseasonality = data

# cumulative_data = np.zeros([n,m])
# for i in range(z):
#     cumulative_data = all_data[i]+cumulative_data
# average_data = cumulative_data/z
# print(average_data)
# average_data.index.name = None

out_file_path = r"C:\Users\m-tot\Desktop\Python spring 2022\autoencoder_1\custom_data\\"
new_dataset.to_csv(out_file_path + "halfabruptleakswithseasonalityremoved.csv")

#halfabruptleaksv2  includes p232


