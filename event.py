'''
src_dest_pair
'''
import copy
import InputConstants
import numpy as np
import random
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

from queue import PriorityQueue
from slice import Slice

inputs = InputConstants.Inputs()

def event():
    '''
    input: null
    output: slice event list [time, arrive/leave, slice]
    '''
    def nexExp(rate):
        return (-1/rate)*np.log(random.random())
    arriveRate = inputs.trafficLoad
    pre = 0
    arrLeaveQuene = []
    for i in range(inputs.sliceNum):
        temp = Slice(i)
        arrTime = pre + nexExp(arriveRate)
        temp.arrTime = arrTime
        arr = [arrTime,'arrive', temp]
        duringTime = nexExp(inputs.serviceRate)
        leaveTim = arrTime+duringTime
        temp.endTime = leaveTim
        leave = [leaveTim,'leave',temp]
        pre = arrTime
        arrLeaveQuene.append(arr)
        arrLeaveQuene.append(leave)
    return arrLeaveQuene

def initEventQuene(arrLeaveQuene):
    tempQuene = copy.deepcopy(arrLeaveQuene)
    testq2 = PriorityQueue()
    for i in range(inputs.sliceNum*2):
        testq2.put(tempQuene[i])
    return testq2

