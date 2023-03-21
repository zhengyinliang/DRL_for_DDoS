#import random
#random.seed(99999)
import graph
import functions as fun
import numpy as np
import InputConstants
inputs = InputConstants.Inputs()
import event
import copy
sliceQue = event.event()

class SliceEnv():
    #action_bound = [0,1]   #动作范围
    action_dim = 2 #两个动作
    state_dim = 81  #1个观测值
    #dosList = [58, 7, 61, 33, 24, 74, 18, 52, 25, 19, 37]
    dosList = [103]
    get_point = False

    def __init__(self):
        self.sliceDic = {}
        self.top = graph.topology()
        self.G = self.top.G
        self.dosId = self.dosList[0]
        self.count = 0
        self.eventQuene = event.initEventQuene(sliceQue)
        self.blockForDos = 0.0
        self.dosSliceCount = 0.0
        

    def step(self,action,dosId, sliceId,currentTime):
        #actionlist = np.clip(action, *self.action_bound)
        #action = round(actionlist[0])
        r,b = fun.rlDealSlice(self.G, self.top, dosId, self.sliceDic, sliceId, action, currentTime)
        # add mark
        self.sliceCount += 1
        if b == 0:
            self.blockForDos += 1
        s = self._get_state()
        return s, r, self.get_point, b
        
    def reset(self):
        self.get_point = False
        self.sliceDic = {}  # 记录slice 对象
        self.top = graph.topology()
        self.top.nodeScore = copy.deepcopy(inputs.nodeScore)
        self.G = self.top.G
        self.count == 0
        self.blockForDos = 0.0
        self.sliceCount = 0.0
        self.eventQuene = event.initEventQuene(sliceQue)
        self.top.nodeSlice = copy.deepcopy(inputs.nodeSlice)
        self.dosId = self.dosList[0]

        return self._get_state()

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        # point1 = self.top.nodeScore[101]
        # point2 = self.top.nodeScore[102]
        # point3 = self.top.nodeScore[103]
        # point4 = self.top.nodeScore[104]
        # point5 = self.top.nodeScore[105]
        # point6 = self.top.nodeScore[200]
        points = []
        for i in range(1,76):
            points.append(self.top.nodeScore[i])
            points.append(fun.getNodeLinkScore(self.top.G,i))
        for i in range(101,106):
            points.append(self.top.nodeScore[i])
            points.append(fun.getNodeLinkScore(self.top.G,i))
        points.append(self.top.nodeScore[200])
        points.append(fun.getNodeLinkScore(self.top.G,i))



        #return np.hstack([point1, point2, point3,point4,point5,point6])
        return points
