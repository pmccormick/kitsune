import pandas as pd
import sys
import os
from matplotlib import pyplot as plt

data_file,ext= os.path.splitext(sys.argv[1])
lp_components=data_file.split('__')
print(lp_components)
exec_name=lp_components[1]
gpu_name=lp_components[2].replace('_', ' ')
tstamp=lp_components[3]

columns = ["ThreadsPerBlock", "Time"]
df = pd.read_csv(sys.argv[1], usecols=columns, skiprows=[0,1,2])

title_font = {'family': 'serif',
              'color':  'darkblue',
              'weight': 'bold',
              'size': 20,
             }

axis_label_font = {'family': 'serif',
                   'color':  'darkblue',
                   'size': 16,
                  }

plot_title = 'Launch Parameter Study\n' + 'Executable: ' + exec_name + '\n' + 'GPU: ' + gpu_name + '\n' + tstamp
plt.rcParams["figure.figsize"] = [7.00, 5.00]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
fig, ax = plt.subplots()
ax.plot(df.ThreadsPerBlock, df.Time)
plt.title(plot_title, fontdict=title_font)
plt.xlabel('Threads Per Block', fontdict=axis_label_font)
plt.ylabel('Execution Time (seconds)', fontdict=axis_label_font)
ax.grid()
fig.savefig('lp_' + exec_name + '.pdf')
#plt.show()



