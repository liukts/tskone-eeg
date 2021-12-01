import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LW = 3
MS = 8
FS = 22
lightGray = "#D0D0D0"
lightOrange = '#FFC29B'
matplotlib.rcParams['axes.linewidth'] = 3.5 #set the value globally

targ_dir = ['211123_lstm_tskone_direct_0_05hu_lr2.00e-2','211123_lstm_tskone_context_0_05hu_lr9.00e-4']
ccs = []
for dir_idx,dir in enumerate(targ_dir):
    ccs.append(np.load('./results/'+dir+'/cc.npy'))

for i in range(0,len(ccs)):
    plt.plot(np.mean(ccs[i],axis=1),'.-')
plt.savefig('cc.png')