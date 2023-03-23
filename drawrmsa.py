import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

name_list=['BP','entropy','policy_loss','value_loss']
Ylist_bp = []
Ylist_Ploss = []
Ylist_Vloss = []
Ylist_entro = []
for name in name_list:

    f = open(r'./{}.dat'.format(name), encoding='utf-8')
    if name==name_list[0]:

        for line in f:
            s = line.strip().split('\t')
            Ylist_bp.append(float(s[0]))
        f.close()
    if name==name_list[1]:

        for line in f:
            s = line.strip().split('\t')
            Ylist_entro.append(float(s[0]))
        f.close()
    if name==name_list[2]:

        for line in f:
            s = line.strip().split('\t')
            Ylist_Ploss.append(float(s[0]))
        f.close()
    if name==name_list[3]:

        for line in f:
            s = line.strip().split('\t')
            Ylist_Vloss.append(float(s[0]))
        f.close()


fig, ax = plt.subplots(2, 2, figsize=(10, 5))
line=0.05
for a in ax.reshape(-1,1):
    a[0].set_xlabel("epochs")
ax[0][0].plot(Ylist_bp,    linewidth=line, color='black', label='BP')
ax[0][0].legend()
ax[1][0].plot(Ylist_entro, linewidth=line,color='black', label='entropy')
ax[1][0].legend()
ax[0][1].plot(Ylist_Ploss, linewidth=line,color='black', label='policy_loss')
ax[0][1].legend()
ax[1][1].plot(Ylist_Vloss, linewidth=line, color='black', label='value_loss')
ax[1][1].legend()
plt.savefig("./res/result1000-data")
plt.show()
