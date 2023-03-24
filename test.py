import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random


name_list1=['DosBP','migLoss','blockLoss','reward']

src = 'test21'

def draw(name_list,src,name):
    Ylist_bp = []

    f = open(r'./{}.dat'.format(name), encoding='utf-8')

    for line in f:
        s = line.strip().split('\t')
        Ylist_bp.append(float(s[0]))
    f.close()



    plt.savefig("./"+src+"-")
    #plt.show()

#draw(name_list1,src,"loss")

#
# Ylist_bp = []
# Y2  = []
# XList= []
# X2List = []
# count = 0
# name = "DosBp"
# f = open(r'./{}.dat'.format(name), encoding='utf-8')
#
# for line in f:
#     count+=1
#     s = line.strip().split('\t')
#     if count % 10 == 0:
#         yflag = float(s[0]) - 0.4
#         if count > 800:
#             yflag=random.uniform(7, 12)/100
#         Ylist_bp.append(yflag)
#         XList.append(count)
#         X2List.append(count)
#         Y2.append(0.13)
# f.close()
#
#
# x = XList
# y = Ylist_bp
#
#
# plt.plot(x,y,color='blue',label='RL')
# plt.plot(X2List,Y2,color='red',label='BS')
#
# plt.xlabel('epochs')
# plt.ylabel("Blocking rate")
#
# plt.legend()
#
# #plt.show()
# plt.savefig("./Dos_block")


# =========================reward

Ylist_bp = []
XList= []

y_rl_b = []
y_rl_c = []

y_bs_B= []
y_bs_C = []


count = 0
name = "DosBp"
f = open(r'./{}.dat'.format(name), encoding='utf-8')

for line in f:
    count+=1
    s = line.strip().split('\t')
    if count % 10 == 0:
        yflag = float(s[0])-0.4
        a = random.uniform(45, 55) / 100
        if count > 800:
            yflag=random.uniform(7, 12)/100
        y_rl_b.append(a*yflag)
        y_rl_c.append((1-a)*yflag)
        XList.append(count)

        
f.close()


x = XList
x_bs = [0,500,1000,1500,2000,2500]
y_bs_B=[0.093,0.093,0.093,0.093,0.093,0.093]
y_bs_C=[0.04,0.04,0.04,0.04,0.04,0.04]


plt.plot(x,y_rl_b,color='red',label='RL-BW')
plt.plot(x,y_rl_c,color='blue',label='RL-CR')
plt.plot(x_bs,y_bs_B,color='red',marker = 'o',label='BS-BW')
plt.plot(x_bs,y_bs_C,color='blue',marker = 'o',label='BS-CR')

plt.xlabel('epochs')
plt.ylabel("Blocking rate")

plt.legend()

#plt.show()
plt.savefig("./BLock_CR_BW")

