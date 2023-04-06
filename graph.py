import random

import numpy as np
import tensorflow as tf

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
'''
输出：拓扑
'''
import InputConstants
import copy
import networkx as nx

inputs = InputConstants.Inputs()


class topology:
    def generateGraph(self):
        G = nx.Graph()
        for j in range(0, 15):
            mnID = 101 + j // 3
            aeID = j * 5 + 1
            tempwavelengths = copy.deepcopy(self.wavelengths)
            tempPathScore = copy.deepcopy(self.lightPathScore)
            G.add_edge(mnID, aeID, weight=self.aeToae, wavelength=tempwavelengths, pathScore=tempPathScore)
            for k in range(1, 5):
                aeID = j * 5 + k
                tempwavelengths = copy.deepcopy(self.wavelengths)
                tempPathScore = copy.deepcopy(self.lightPathScore)
                G.add_edge(aeID, aeID + 1, weight=self.aeToae,
                           wavelength=tempwavelengths, pathScore=tempPathScore)
            tempwavelengths = copy.deepcopy(self.wavelengths)
            tempPathScore = copy.deepcopy(self.lightPathScore)
            G.add_edge(mnID, aeID + 1, weight=self.aeToae,
                       wavelength=tempwavelengths, pathScore=tempPathScore)
        for j in range(101, 105):
            tempwavelengths = copy.deepcopy(self.wavelengths)
            tempPathScore = copy.deepcopy(self.lightPathScore)
            G.add_edge(j, j + 1, weight=self.mnTomn, wavelength=tempwavelengths, pathScore=tempPathScore)
        tempwavelengths = copy.deepcopy(self.wavelengths)
        tempPathScore = copy.deepcopy(self.lightPathScore)
        G.add_edge(101, 200, weight=self.mnTomn, wavelength=tempwavelengths, pathScore=tempPathScore)
        tempwavelengths = copy.deepcopy(self.wavelengths)
        tempPathScore = copy.deepcopy(self.lightPathScore)
        G.add_edge(105, 200, weight=self.mnTomn, wavelength=tempwavelengths, pathScore=tempPathScore)
        # nx.draw(G, with_labels=True, edge_color='b', node_color='g')
        # plt.savefig("path.png")
        for i in range(1, 76):
            temp = copy.deepcopy(self.serverNumAE)
            tempVmlevel = copy.deepcopy(self.vmLevelAE)
            tempVmType = copy.deepcopy(self.vmTypeAE)
            G.nodes[i]['type'] = tempVmType
            G.nodes[i]['servers'] = temp
            G.nodes[i]['vmLevel'] = tempVmlevel
        for i in range(101, 106):
            temp = copy.deepcopy(self.serverNumMN)
            tempVmlevel = copy.deepcopy(self.vmLevelMN)
            tempVmType = copy.deepcopy(self.vmTypeMN)
            G.nodes[i]['type'] = tempVmType
            G.nodes[i]['servers'] = temp
            G.nodes[i]['vmLevel'] = tempVmlevel
        G.nodes[200]['type'] = copy.deepcopy(self.vmTypeME)
        G.nodes[200]['servers'] = copy.deepcopy(self.serverNumME)
        G.nodes[200]['vmLevel'] = copy.deepcopy(self.vmLevelME)
        return G

    def __init__(self):
        self.aeToae = inputs.aeToae
        self.mnTomn = inputs.mnTomn
        self.wavelengths = inputs.wavelengths
        self.serverNumAE = inputs.serverNumAE
        self.serverNumMN = inputs.serverNumMN
        self.serverNumME = inputs.serverNumME
        self.vmLevelAE = inputs.vmLevelAE
        self.vmLevelMN = inputs.vmLevelMN
        self.vmLevelME = inputs.vmLevelME
        self.vmTypeAE = inputs.vmTypeAE
        self.vmTypeMN = inputs.vmTypeMN
        self.vmTypeME = inputs.vmTypeME
        self.max_node_score = inputs.max_node_score
        # mark block
        self.blockNumForAAU = inputs.markBlock

        # markLinghtPathScore
        self.lightPathScore = inputs.lightPathScore
        # self.blockNumForLink = inputs.markBlock
        # self.blockNumForNode = inputs.markBlock
        # self.totalBlock = inputs.markBlock
        # self.idle_AAU_num = num_access_ring*num_AE*num_BS*sub_aau_num_each_aau
        self.sub_aau_num_each_aau = 3
        self.total_aau_num = 15 * 5 * 6 * 3
        self.idle_AAU_num = 15 * 5 * 6 * 3
        self.aau_map_slice_num = {}
        self.G = self.generateGraph()
        self.sliceQuene = []
        self.nodeScore = copy.deepcopy(inputs.nodeScore)
        self.nodeSlice = copy.deepcopy(inputs.nodeSlice)

        # 标记
        self.totalLoss = 0.0
        self.blockLoss = 0.0
        self.migLoss = 0.0

        self.l1MigNumber = 0.0
        self.l2MigNumber = 0.0
        self.l3MigNumber = 0.0
        self.l4MigNumber = 0.0

        self.l1MigNumberMn = 0.0
        self.l2MigNumberMn = 0.0
        self.l3MigNumberMn = 0.0
        self.l4MigNumberMn = 0.0

        self.l1TotalNumber = 0.0
        self.l2TotalNumber = 0.0
        self.l3TotalNumber = 0.0
        self.l4TotalNumber = 0.0

        self.migAeNum = 0.0
        self.migMnNum = 0.0
        self.notMigNum = 0.0

        # 标记，若正常映射，则为false
        # 正常业务到达数量
        self.sliceArrived = 0.0
        self.sliceBlocked = 0.0

        # 被攻击的切片
        self.totalSliceNumDosed = 0.0
        self.blockForDos = 0.0
        self.flag = False

        self.l1BpNumber = 0.0
        self.l2BpNumber = 0.0
        self.l3BpNumber = 0.0
        self.l4BpNumber = 0.0

        self.bpOfMappingForNode = 0.0
        self.bpOfMappingForBW = 0.0

        self.bpOfDosForNode = 0.0
        self.bpOfDosForBW = 0.0

    def updateLink(self, G, src, des, wNumber, capability):
        # print(src,des,G[src][des])

        tempflag = G[src][des]['wavelength'][wNumber]
        if tempflag == inputs.waveCapabity and capability != 0:
            G[src][des]['pathScore'] -= 1

        G[src][des]['wavelength'][wNumber] += capability

        if G[src][des]['wavelength'][wNumber] == inputs.waveCapabity and capability != 0:
            G[src][des]['pathScore'] += 1

        # print(src,des,G[src][des])

    def updateLightPath(self, G, lightPath, wNumber, capability):
        if wNumber == -1:
            return
        for i in range(len(lightPath) - 1):
            src = lightPath[i]
            des = lightPath[i + 1]
            self.updateLink(G, src, des, wNumber, capability)

    def updateServerVM(self, G, nodeId, serverId, vmId, capability, type, level):
        '''
        server 0/1: 0/5
        '''
        G.nodes[nodeId]['servers'][serverId, vmId] += capability
        self.nodeScore[nodeId] += capability
        if G.nodes[nodeId]['servers'][serverId, vmId] == inputs.VMCapability:
            self.resetServerVM(G, nodeId, serverId, vmId)
        elif G.nodes[nodeId]['type'][serverId, vmId] == inputs.LevelAndType:
            G.nodes[nodeId]['type'][serverId, vmId] = type
            G.nodes[nodeId]['vmLevel'][serverId, vmId] = level
        # print(G.nodes[nodeId]['vmLevel'][serverId, vmId])

    def searchLightPath(self, G, lightPath, capability):
        '''
        寻找满足波长一致性的lightPath
        返回：波长链表，已用的 和未用的  // 可能为空，阻塞
        -1 表示不需要链路
        '''
        if len(lightPath) < 2:
            return [-1]
        usedWave = []
        newWave = []
        slots = copy.deepcopy(G[lightPath[0]][lightPath[1]]['wavelength'])
        for i in range(len(lightPath) - 1):
            src = lightPath[i]
            des = lightPath[i + 1]
            curLinkSlots = G[src][des]['wavelength']
            for j in range(len(slots)):
                slots[j] = min(slots[j], curLinkSlots[j])
        for j in range(len(slots)):
            if slots[j] >= capability and slots[j] < inputs.waveCapabity:
                usedWave.append(j)
            elif slots[j] == inputs.waveCapabity:
                newWave.append(j)
        res = usedWave + newWave
        return res

    def searchServerVM(self, G, nodeId, capability, type, level):
        '''
        输入：节点列表
        输出：可用的node vm type编号
        for i in range(2):
        print(i)
        print(len(G.nodes[1]['servers']))
        print(G.nodes[1]['servers'][0])
        '''
        res = []
        count = 0
        serverNum = len(G.nodes[nodeId]['servers'])
        for server in range(serverNum):
            for vm in range(len(G.nodes[nodeId]['servers'][server])):
                if G.nodes[nodeId]['servers'][server, vm] >= capability:
                    if G.nodes[nodeId]['type'][server, vm] == inputs.LevelAndType or G.nodes[nodeId]['type'][
                        server, vm] == type:  # 相同类型
                        if G.nodes[nodeId]['vmLevel'][server, vm] == -1 or (
                                G.nodes[nodeId]['vmLevel'][server, vm] > -1 and abs(
                                G.nodes[nodeId]['vmLevel'][server, vm] - level) <= inputs.isolation):
                            if judgeIlleagle(G.nodes[nodeId]['type'][server, vm],
                                             G.nodes[nodeId]['vmLevel'][server, vm], type, level):
                                res.append([nodeId, server, vm])
                                count += 1
                                if count >= 5:
                                    break
            if count >= 5:
                break
        return res

    def resetServerVM(self, G, nodeId, serverId, vmId):
        G.nodes[nodeId]['servers'][serverId, vmId] = inputs.VMCapability
        G.nodes[nodeId]['type'][serverId, vmId] = inputs.LevelAndType
        G.nodes[nodeId]['vmLevel'][serverId, vmId] = inputs.LevelAndType


def judgeIlleagle(exisType, exisLevel, newType, newLevel):
    # type -1~2  level -1 ~ 4
    if exisType == -1:  # null
        return True
    if exisType == 0 and newType == 0:  # DU
        if exisLevel <= 3 and newLevel <= 3:
            return True
        else:
            return False
    if (exisType == 1 and newType == 1):  # CU
        if (exisLevel <= 2 and newLevel <= 2):
            return True
        else:
            return False
    if (exisType == 2 and newType == 2):  # MEC
        if (exisLevel <= 1 and newLevel <= 1):
            return True
        else:
            return False
    print("judgeIlleagle wrong")
    return False
