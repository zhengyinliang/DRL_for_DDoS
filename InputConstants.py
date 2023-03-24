#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on zhengyin liang 12 2021

@author: liang
"""
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

class Inputs:
    #eMBB, uRLLC, mMTC
    #type = (0,1,2)
    #====================代改
    aauNums = (1,1,2)
    resoucers = [[20,10,5],[4,2,1],[4,4,2]]
    transLatency = [[0.1,1,20],[0.05,0.05,0.5],[0.25,10,10]]
    bandWidth = [[25,15,15],[4,1,1],[4,1,1]]
    #isolation = (1,2,3,4)
    #reliability = (99,99.9,99.99,99.999)
    linkLatency = 0.005
    oeoLatency = 0.02
    vmLatcncy = 0.052

    waveCapabity = 25
    
    lightPathScore = 80

    VMCapability = 25
    LevelAndType = -1
    isolation = 4
    #10 8   500  1500 4500
    vmNumeberPerServer = 10
    aeNum = 2
    mnNum = 6
    meNum = 32
    serverNumAE  = np.ones([aeNum,vmNumeberPerServer]) * VMCapability
    serverNumMN = np.ones([mnNum,vmNumeberPerServer]) * VMCapability
    serverNumME = np.ones([meNum,vmNumeberPerServer]) * VMCapability
    vmLevelAE = np.ones([aeNum,vmNumeberPerServer])*LevelAndType
    vmLevelMN = np.ones([mnNum,vmNumeberPerServer])*LevelAndType
    vmLevelME = np.ones([meNum,vmNumeberPerServer])*LevelAndType
    vmTypeAE = np.ones([aeNum,vmNumeberPerServer])*LevelAndType
    vmTypeMN = np.ones([mnNum,vmNumeberPerServer])*LevelAndType
    vmTypeME = np.ones([meNum,vmNumeberPerServer])*LevelAndType

    nodeScore = {}
    nodeSlice = {}
    for i in range(1,76):
        nodeSlice[i] = {} 
        nodeScore[i] = aeNum*vmNumeberPerServer*VMCapability
    for i in range(101,106):
        nodeSlice[i] = {} 
        nodeScore[i] = mnNum*vmNumeberPerServer*VMCapability
    nodeScore[200] = meNum*vmNumeberPerServer*VMCapability
    nodeSlice[200] = {} 

    serviceRate = 1
    trafficLoad = 500
    # 5000 500 1000
    sliceNum = 2000
    markBlock = 0

    aeToae = 5
    mnTomn = 50
    wavelengths = np.ones(80)* 25
    num_BS = 6
    DosStrength = 0

    # 是否统计 攻击 后续切片影响
    countAfterDos = False

    # 已经到达的切片
    sliceArrived = 1000
    
    migTime = 0.052 + 0.25