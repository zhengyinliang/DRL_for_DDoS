import random

import numpy as np
import tensorflow as tf

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
import graph
import functions as fun
import InputConstants

inputs = InputConstants.Inputs()
import event
import copy

sliceQue = event.event()


class SliceEnv():
    # action_bound = [0,1]   #动作范围
    action_dim = 160  # 动作
    state_dim = 54  # 54个观测值
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
        self.slice = None

    def set_slice(self, input_slice):
        self.slice = self.sliceDic[input_slice]

    def step(self, action, dosId, sliceId, currentTime):
        # actionlist = np.clip(action, *self.action_bound)
        # action = round(actionlist[0])
        r, b = fun.rlDealSlice(self.G, self.top, dosId, self.sliceDic, sliceId, action, currentTime)
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
        self.slice = None
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
        # state更新设置
        # 检查points是否正确==========
        # bw

        points += [0, fun.getNodeLinkScore(self.top.G, 101), 0, 0, 0, fun.getNodeLinkScore(self.top.G, 200)]
        points += [fun.getNodeLinkScore(self.top.G, 101), 0, fun.getNodeLinkScore(self.top.G, 102), 0, 0, 0]
        points += [0, fun.getNodeLinkScore(self.top.G, 102), 0, fun.getNodeLinkScore(self.top.G, 103), 0, 0]
        points += [0, 0, fun.getNodeLinkScore(self.top.G, 103), 0, fun.getNodeLinkScore(self.top.G, 104), 0]
        points += [0, 0, 0, fun.getNodeLinkScore(self.top.G, 104), 0, fun.getNodeLinkScore(self.top.G, 105), 0]
        points += [fun.getNodeLinkScore(self.top.G, 200), 0, 0, 0, fun.getNodeLinkScore(self.top.G, 105), 0]
        # node
        for i in range(101, 106):
            points.append(self.top.nodeScore[i]/self.top.max_node_score)
        points.append(self.top.nodeScore[200]/self.top.max_node_score)

        # slice
        if self.slice != None:
            points += self.slice.resource
            points += self.slice.bandwidth
            points += self.slice.transLatency
            if self.slice.DU[0] == self.dosId:
                points.append(1)
            else:
                points.append(0)
            if self.slice.CU[0] == self.dosId:
                points.append(1)
            else:
                points.append(0)
            if self.slice.MEC[0] == self.dosId:
                points.append(1)
            else:
                points.append(0)
        else:
            points += [0, 0, 0, 0,
                       0, 0, 0, 0,
                       0, 0, 0, 0]
        # print(points)
        # print(len(points))
        # return np.hstack([point1, point2, point3,point4,point5,point6])
        return points
