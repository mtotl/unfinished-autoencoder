import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np



#points = [31 etc]
data = pd.read_csv('2018_Leakages_dot.csv', delimiter= ';',parse_dates= True, index_col = 0)

print(np.shape(data))

#convert df to right datatype, timestamp is still in str
# for column in data.columns:
#      if column != "Timestamp":
#          data[column].astype(float)

#data = data.resample('1D').mean()
#print(data)
#print(type(data.index)) #to do: 

mpl.rcParams['font.size']=14
fig = plt.figure(1, figsize = (12,12)) #used to be 12,12 

gs = gridspec.GridSpec(4, 16)
gs.update(wspace = 1.6,  hspace = 0.8)

colors = sns.color_palette("rocket", 6)
sns.set_style('darkgrid')

n = [2,4,6,8,10,12,14,16]
m = 0



# fig, ax = plt.subplots()
# fig.canvas.draw()

# leak_nr = 'p232'

# ax.plot(range(len(data[leak_nr])),data[leak_nr], color = colors[1])
# plt.ylabel(r'Water loss $(\frac{m^3}{h})$')
# plt.title(f'A: Leak in pipe: {leak_nr}')

# labels = [item.get_text() for item in ax.get_xticklabels()]
# print(labels)
# labels[1] = '01-2018'
# labels[2] = '03-2018'
# labels[3] = '05-2018'
# labels[4] = '07-2018'
# labels[5] = '09-2018'
# labels[6] = '11-2018'

# ax.set_xticklabels(labels)


#plt.show()

for it, column in enumerate(data.columns):
    if it <= 6:
        p = fig.add_subplot(gs[0:2, n[it]-2:n[it]])
        #range(len(data[column])) -> using index instead creates a mess, dunno why
        #data[column].plot()
        plt.plot(range(len(data[column])), data[column], color = colors[1])
        #data.subplot(use_index = True, color  = colors[1])
        #plt.ylabel(r'Water loss, m3/h')
        plt.title(column)

    else:
        #p2 = fig.add_subplot(gs[2:4, :])
        # props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
        # textstr = (f'\n batch_size = {batch_size},\n optimzer = {optimizer}, \n learning rate =  {learning_rate}')
        # plt.text(5, 5, textstr, bbox = props)
        p2 = fig.add_subplot(gs[2:4, n[m]-2:n[m]])
        plt.plot(range(len(data[column])),data[column], color = colors[1])
        #plt.ylabel(r'Water loss, m3/h')
        plt.title(column)
        m = m+1

plt.show()

# #find date and time of peak leak
max_leak = {}
data = data.reset_index()
for col in data.columns:
    max_leak[col] = data[col].idxmax()

print(max_leak)

# for  m in (max_leak):
#     print(m)
#     plt.plot(max_leak[m], 5, marker = 'o')
# plt.show()
