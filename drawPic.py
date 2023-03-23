import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

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
