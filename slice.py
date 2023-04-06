import InputConstants

input = InputConstants.Inputs()
import numpy as np
import tensorflow as tf
import random

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)


class Slice:
    '''
    切片属性
    id
    类型
    '''

    def __init__(self, id):
        self.id = id

        self.rd = random.randint(0, 3)
        # self.rd = random.randint(0, 1)
        if self.rd == 0:
            self.type = 1
        else:
            self.type = 0
        # self.type = 1  # urllc
        # self.type = 0
        flag = self.type
        self.aau = input.aauNums[flag]
        self.resource = input.resoucers[flag]
        self.bandwidth = input.bandWidth[flag]
        self.transLatency = input.transLatency[flag]
        # self.level = 1
        self.level = random.randint(1, 4)
        self.reliability = random.randint(1, 4)
        self.arrTime = None
        self.endTime = None
        # node， server and vm id
        self.real = False

        self.aauid = None
        self.DU = None
        self.CU = None
        self.MEC = None
        self.front = None
        self.mid = None
        self.back = None
        self.list_AAU = []

    def generateAAU(self, topo):
        if self.aau <= topo.idle_AAU_num:
            i = 0
            total_aau_num = 15 * 5 * 6
            while i < self.aau:
                temp = random.randint(1, total_aau_num)
                if topo.aau_map_slice_num.get(temp):
                    existing_slice_num = topo.aau_map_slice_num[temp]
                    if existing_slice_num < topo.sub_aau_num_each_aau:
                        self.list_AAU.append(temp)
                        topo.aau_map_slice_num[temp] += 1
                        i += 1
                else:
                    topo.aau_map_slice_num[temp] = 1
                    self.list_AAU.append(temp)
                    i += 1
            topo.idle_AAU_num -= self.aau


