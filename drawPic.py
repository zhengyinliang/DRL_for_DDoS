import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)

name_list1=['DosBP','migLoss','blockLoss','reward']

name_list3 = ['bpBeforeDos','DosBP','bpAfterDos','blockLoss']

name_list2=['migProbal1','migProbal2','migProbal3','migProbal4']
src = 'test21'

def draw(name_list,src,type): 
    Ylist_bp = []
    Ylist_reward = []
    Ylist_entro = []
    Ylist_migLoss = []
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
                Ylist_reward.append(float(s[0]))
            f.close()
        if name==name_list[3]:
            for line in f:
                s = line.strip().split('\t')
                Ylist_migLoss.append(float(s[0]))
            f.close()
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    line=2
    for a in ax.reshape(-1,1):
        a[0].set_xlabel("epochs")
    ax[0][0].plot(Ylist_bp,    linewidth=line, color='black', label=name_list[0])
    ax[0][0].legend()
    ax[1][0].plot(Ylist_entro, linewidth=line,color='black', label=name_list[1])
    ax[1][0].legend()
    ax[0][1].plot(Ylist_reward, linewidth=line,color='black', label=name_list[2])
    ax[0][1].legend()

    ax[1][1].plot(Ylist_migLoss, linewidth=line,color='black', label=name_list[3])
    ax[1][1].legend()
    plt.savefig("./"+src+"-"+type)
    #plt.show()

draw(name_list1,src,"loss")
draw(name_list2,src,"mig")
draw(name_list3,src,"bp")

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

name_list = ['MN_1', 'MN_2', 'MN_4', 'MN_5','ME']
num_list = [4, 7, 6, 6, 11]
num_list1 = [4, 5, 7, 5, 10]
plt.bar(range(len(num_list)), num_list, label='uRLLC', fc=sns.xkcd_rgb["medium purple"],width=0.5)
plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='eMBB', tick_label=name_list, fc=sns.xkcd_rgb["blue"],width=0.5)
my_y_ticks = np.arange(0, 25, 5)
plt.yticks(my_y_ticks)
plt.xlabel("node")
plt.ylabel("migrate slice number")
plt.legend()
#plt.show()
plt.savefig("./"+"basefenbu")


name_list = ['MN_1', 'MN_2', 'MN_4', 'MN_5','ME']
num_list = [6, 5, 5, 8, 11]

num_list1 = [4, 8, 9, 4, 9]

plt.bar(range(len(num_list)), num_list, label='uRLLC', fc=sns.xkcd_rgb["medium purple"],width=0.5)
plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='eMBB', tick_label=name_list, fc=sns.xkcd_rgb["blue"],width=0.5)
my_y_ticks = np.arange(0, 25, 5)
plt.yticks(my_y_ticks)
plt.xlabel("node")
plt.ylabel("migrate slice number")

plt.legend()
#plt.show()
plt.savefig("./"+"rl_fenbu")
