import math
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
import InputConstants
import networkx as nx
import os

inputs = InputConstants.Inputs()
# action dic
import copy
ACTION = []
count = 0

temp = [-1, -1, -1]


temp = [-1, -1, -1]
for i in range(1, 6):
    temp[1] = i
    for j in range(1, 6):
        temp[2] = j
        if (temp[2] != 3 or temp[2] != -1) and (temp[0] == 3 or temp[1] == 3):
            pass
        else:
            ACTION.append(copy.copy(temp))
            count += 1

temp = [-1, -1, -1]
for i in range(1, 6):
    temp[2] = i
    if (temp[2] != 3 or temp[2] != -1) and (temp[0] == 3 or temp[1] == 3):
        pass
    else:
        ACTION.append(copy.copy(temp))
        count += 1
temp = [-1, -1, -1]

for i in range(1, 6):
    temp[1] = i
    if (temp[2] != 3 or temp[2] != -1) and (temp[0] == 3 or temp[1] == 3):
        pass
    else:
        ACTION.append(copy.copy(temp))
        count += 1


ACTION_DIC = {1: 101, 2: 102, 3: 200, 4: 104, 5: 105}


def markBp(top, bpflag, tempSlice):
    if top.flag == False:  # 正常映射阻塞
        if bpflag == 0:
            top.bpOfMappingForNode += 1
        else:
            top.bpOfMappingForBW += 1
    else:
        # dos
        if bpflag == 0:  # node
            top.bpOfDosForNode += 1
        else:
            top.bpOfDosForBW += 1
        if tempSlice.level == 1:
            top.l1BpNumber += 1
        if tempSlice.level == 2:
            top.l2BpNumber += 1
        if tempSlice.level == 3:
            top.l3BpNumber += 1
        if tempSlice.level == 4:
            top.l4BpNumber += 1


def markFunction(dos_node, dos_bw, slice_info_ss, total_step,ss_action, res_path, bpBefore, bpAfter, dosLoss, bpDos, blockLoss, migLoss, migProbal1, migProbal2, migProbal3,
                 migProbal4, notMigNum, migAeNum, migMnNum, l1NotMig, l2NotMig, l3NotMig, l4NotMig, l1MigAe, l2MigAe,
                 l3MigAe, l4MigAe, l1MigMn, l2MigMn, l3MigMn, l4MigMn):
    # bp total block

    fp = open('./' + res_path + '/block_dos_for_node.dat', 'a')
    fp.write('%f\n' % dos_node)
    fp.close()

    fp = open('./' + res_path + '/block_dos_for_bw.dat', 'a')
    fp.write('%f\n' % dos_bw)
    fp.close()

    fp = open('./' + res_path + '/action_ss.dat', 'a')
    fp.write('epoch: '+str(total_step) + '\n')
    fp.write(str(ss_action)+'\n')
    fp.write('slice_type: ' + str(slice_info_ss) + '\n')
    fp.close()

    fp = open('./' + res_path + '/bpBeforeDos.dat', 'a')
    fp.write('%f\n' % bpBefore)
    fp.close()

    fp = open('./' + res_path + '/bpBeforeDos.dat', 'a')
    fp.write('%f\n' % bpBefore)
    fp.close()

    fp = open('./' + res_path + '/bpBeforeDos.dat', 'a')
    fp.write('%f\n' % bpBefore)
    fp.close()
    # after
    fp = open('./' + res_path + '/bpAfterDos.dat', 'a')
    fp.write('%f\n' % bpAfter)
    fp.close()
    # dos
    fp = open('./' + res_path + '/DosBp.dat', 'a')
    fp.write('%f\n' % bpDos)
    fp.close()
    # reward
    fp = open('./' + res_path + '/reward.dat', 'a')
    fp.write('%f\n' % dosLoss)
    fp.close()

    fp = open('./' + res_path + '/blockLoss.dat', 'a')
    fp.write('%f\n' % blockLoss)
    fp.close()

    fp = open('./' + res_path + '/migLoss.dat', 'a')
    fp.write('%f\n' % migLoss)
    fp.close()

    fp = open('./' + res_path + '/migProbal1.dat', 'a')
    fp.write('%f\n' % migProbal1)
    fp.close()

    fp = open('./' + res_path + '/migProbal2.dat', 'a')
    fp.write('%f\n' % migProbal2)
    fp.close()

    fp = open('./' + res_path + '/migProbal3.dat', 'a')
    fp.write('%f\n' % migProbal3)
    fp.close()

    fp = open('./' + res_path + '/migProbal4.dat', 'a')
    fp.write('%f\n' % migProbal4)
    fp.close()

    fp = open('./' + res_path + '/notMigNum.dat', 'a')
    fp.write('%f\n' % notMigNum)
    fp.close()
    fp = open('./' + res_path + '/migAeNum.dat', 'a')
    fp.write('%f\n' % migAeNum)
    fp.close()
    fp = open('./' + res_path + '/migMnNum.dat', 'a')
    fp.write('%f\n' % migMnNum)
    fp.close()
    fp = open('./' + res_path + '/l1NotMig.dat', 'a')
    fp.write('%f\n' % l1NotMig)
    fp.close()
    fp = open('./' + res_path + '/l2NotMig.dat', 'a')
    fp.write('%f\n' % l2NotMig)
    fp.close()
    fp = open('./' + res_path + '/l3NotMig.dat', 'a')
    fp.write('%f\n' % l3NotMig)
    fp.close()
    fp = open('./' + res_path + '/l4NotMig.dat', 'a')
    fp.write('%f\n' % l4NotMig)
    fp.close()

    fp = open('./' + res_path + '/l1MigAe.dat', 'a')
    fp.write('%f\n' % l1MigAe)
    fp.close()
    fp = open('./' + res_path + '/l2MigAe.dat', 'a')
    fp.write('%f\n' % l2MigAe)
    fp.close()
    fp = open('./' + res_path + '/l3MigAe.dat', 'a')
    fp.write('%f\n' % l3MigAe)
    fp.close()
    fp = open('./' + res_path + '/l4MigAe.dat', 'a')
    fp.write('%f\n' % l4MigAe)
    fp.close()
    fp = open('./' + res_path + '/l1MigMn.dat', 'a')
    fp.write('%f\n' % l1MigMn)
    fp.close()
    fp = open('./' + res_path + '/l2MigMn.dat', 'a')
    fp.write('%f\n' % l2MigMn)
    fp.close()
    fp = open('./' + res_path + '/l3MigMn.dat', 'a')
    fp.write('%f\n' % l3MigMn)
    fp.close()
    fp = open('./' + res_path + '/l4MigMn.dat', 'a')
    fp.write('%f\n' % l4MigMn)
    fp.close()


def draw_res(res_path):
    name_list1 = ['DosBp', 'reward','bpBeforeDos','bpAfterDos']

    name_list = ['DosBp', 'entropy', 'policy_loss', 'value_loss']

    def draw(res_path, name_list, type):
        Ylist_bp = []
        Ylist_reward = []
        Ylist_entro = []
        Ylist_migLoss = []
        for name in name_list:
            f = open(r'./' + res_path + '/{}.dat'.format(name), encoding='utf-8')
            if name == name_list[0]:
                for line in f:
                    s = line.strip().split('\t')
                    Ylist_bp.append(float(s[0]))
                f.close()
            if name == name_list[1]:
                for line in f:
                    s = line.strip().split('\t')
                    Ylist_entro.append(float(s[0]))
                f.close()
            if name == name_list[2]:
                for line in f:
                    s = line.strip().split('\t')
                    Ylist_reward.append(float(s[0]))
                f.close()
            if name == name_list[3]:
                for line in f:
                    s = line.strip().split('\t')
                    Ylist_migLoss.append(float(s[0]))
                f.close()
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        line = 2
        for a in ax.reshape(-1, 1):
            a[0].set_xlabel("epochs")
        ax[0][0].plot(Ylist_bp, linewidth=line, color='black', label=name_list[0])
        ax[0][0].legend()
        ax[1][0].plot(Ylist_entro, linewidth=line, color='black', label=name_list[1])
        ax[1][0].legend()
        ax[0][1].plot(Ylist_reward, linewidth=line, color='black', label=name_list[2])
        ax[0][1].legend()

        ax[1][1].plot(Ylist_migLoss, linewidth=line, color='black', label=name_list[3])
        ax[1][1].legend()
        plt.savefig("./" + res_path + "/" + type + "res")
        # plt.show()

    draw(res_path, name_list1, "loss")
    draw(res_path, name_list, "value")


def markNodeSliceDel(top, nodeid, sliceId):
    if not top.nodeSlice[nodeid][sliceId] or top.nodeSlice[nodeid][sliceId] <= 0:
        print('markNodeSlice error, out of index')
    else:
        top.nodeSlice[nodeid][sliceId] -= 1
        if top.nodeSlice[nodeid][sliceId] == 0:
            del top.nodeSlice[nodeid][sliceId]


def markNodeSliceAdd(top, nodeid, sliceId):
    if sliceId not in top.nodeSlice[nodeid] or top.nodeSlice[nodeid][sliceId] == 0:
        top.nodeSlice[nodeid][sliceId] = 1
    else:
        top.nodeSlice[nodeid][sliceId] += 1


'''
每一个节点服务器中加入 slice hash表来控制，当攻击时，获得服务器中所有切片信息
'''


def searchNode(latency, path):
    '''
    input: latency， path
    output: path
    '''
    if not path:
        return []
    res = [path[0]]
    transTime = 0.0
    for index in range(0, len(path) - 1):
        if path[index] < 100:
            transTime += 0.005 * 5
        elif path[index] > 100:
            transTime += 0.005 * 50
        if transTime < latency:
            res.append(path[index + 1])
        else:
            return res
    return res


def linkSearch(exisLink, start, G, top, end, bandwidth):
    waveList = []
    pathAeDU = nx.shortest_path(G, start, end, weight='weight')
    linkTemp = top.searchLightPath(G, pathAeDU, bandwidth)
    for i in linkTemp:
        if i not in exisLink or i == -1:
            waveList.append(i)
    if waveList:  # 如果link存在
        if waveList[0] == -1:
            return [-1, pathAeDU]
        return [waveList[0], pathAeDU]
    else:
        return [-10, pathAeDU]


def nodeLinkSearch(exisNode, exisLink, start, G, top, node, slice, type, bandwidth):
    '''
    type = du\ cu\ mec,0,1,2
    output: 不重叠的node，server，vm ，lightpath,
    '''
    flag = 0
    if top.nodeScore[node] < 0:
        return [None, None, None, flag]
    serverVmList = []
    waveList = []
    # 寻找DU节点
    temp = top.searchServerVM(G, node, slice.resource[type], type, slice.level)
    for i in temp:
        if i not in exisNode:
            serverVmList.append(i)
    if serverVmList:  # 如果节点存在
        flag = 1
        first = serverVmList[0]
        # 寻找前传链路
        pathAeDU = nx.shortest_path(G, start, first[0], weight='weight')
        linkTemp = top.searchLightPath(G, pathAeDU, bandwidth)
        for i in linkTemp:
            if i not in exisLink or i == -1:
                waveList.append(i)
        if waveList:  # 如果前传存在
            if waveList[0] == -1:
                return [serverVmList[0], -1, pathAeDU, flag]
            return [serverVmList[0], waveList[0], pathAeDU, flag]
        # 占用节点，链路资源 并标记, 返回DU之后节点列表, 记录 firstDu 如果阻塞需要释放
        # 返回标记
        # top.updateServerVM(G, first[0], first[1], first[2], slice.resource[type], type, slice.level)
        # top.updateLightPath(pathAeDU, fWave, bandwidth)
        # else:
        # top.blockNumForLink += 1
    return [None, None, None, flag]




def nodeQuene(top, arr, funType, slice_type):
    '''
    funType: 0,1,2 --- DU,CU,MEC,不同的分数
    '''
    tempScore = {}
    # 500 相当于权重，对不同fun，偏向不同的节点
    s_ae = inputs.aeNum*inputs.VMCapability*inputs.vmNumeberPerServer
    s_mn = inputs.mnNum*inputs.VMCapability*inputs.vmNumeberPerServer
    s_me = inputs.meNum*inputs.VMCapability*inputs.vmNumeberPerServer

    for i in range(len(arr)):
        if slice_type != 1:
            if funType == 0:
                if arr[i] < 100:
                    # tempScore[arr[i]] =  top.nodeScore[arr[i]]
                    tempScore[arr[i]] = s_me-s_ae + top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
                else:
                    tempScore[arr[i]] = top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
            elif funType == 1:
                if arr[i] >= 100 and arr[i] != 200:
                    # tempScore[arr[i]] =  top.nodeScore[arr[i]]
                    tempScore[arr[i]] = s_me-s_mn + top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
                else:
                    tempScore[arr[i]] = top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
            else:
                # type 2；
                tempScore[arr[i]] = top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
    # print(arr)
        else:
            if funType == 0 or funType == 1:
                if arr[i] < 100:
                    # tempScore[arr[i]] =  top.nodeScore[arr[i]]
                    tempScore[arr[i]] = s_me-s_ae + top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
                else:
                    tempScore[arr[i]] = top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
            else:
                # type 2；
                tempScore[arr[i]] = top.nodeScore[arr[i]]  # + getNodeLinkScore(top.G,arr[i])
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and tempScore[key] > tempScore[arr[j]]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    # print('排序后',arr)
    return arr


def delServiceChain(G, top, slice):
    '''
    输入：切片映射请求
    映射
    '''
    # 寻找资源 AAU, DU, CU, MEC
    slice.generateAAU(top)
    aauList = slice.list_AAU
    if aauList:
        aauId = aauList[0]
    else:
        # 因AAU阻塞
        top.blockNumForAAU += 1
        # print('block for aau')
        return 0
    fLatency = slice.transLatency[0]
    mLatency = slice.transLatency[1]
    bLatency = slice.transLatency[2]
    fBandwidh = slice.bandwidth[0]
    mBandwidh = slice.bandwidth[1]
    bBandwidh = slice.bandwidth[2]

    exisNode = []
    exisWave = []
    totalPath = []
    AE = math.ceil(1.0 * aauId / inputs.num_BS)
    # print('aauID %d, AE %d' %(aauId, AE))

    # AeMe链路
    pathAeMe = nx.shortest_path(G, AE, 200, weight='weight')
    duNodeList = searchNode(fLatency, pathAeMe)
    duNodeList = nodeQuene(top, duNodeList, 0, slice.type)
    for nodeDU in duNodeList:
        serverDU, pathfront, lightfront, bpFlag = nodeLinkSearch(exisNode, exisWave, AE, G, top, nodeDU, slice, 0,
                                                                 fBandwidh)
        if serverDU:
            break
    if not serverDU:
        markBp(top, bpFlag, slice)
        # 阻塞
        # top.blockNumForNode += 1
        return 0
    exisNode.append(serverDU)
    exisWave.append(pathfront)
    totalPath.append(lightfront)
    # DuME
    pathDUMe = nx.shortest_path(G, serverDU[0], 200, weight='weight')
    cuNodeList = searchNode(mLatency, pathDUMe)
    cuNodeList = nodeQuene(top, cuNodeList, 1, slice.type)
    for nodeCU in cuNodeList:
        serverCU, pathmront, lightmid, bpFlag = nodeLinkSearch(exisNode, exisWave, serverDU[0], G, top, nodeCU, slice,
                                                               1, mBandwidh)
        if serverCU:
            break
    if not serverCU:
        # 阻塞CU
        markBp(top, bpFlag, slice)
        # top.blockNumForNode += 1
        # print('line 109 block for CU or midhaul')
        return 0
    exisNode.append(serverCU)
    exisWave.append(pathmront)
    totalPath.append(lightmid)
    # CuME
    pathCUMe = nx.shortest_path(G, serverCU[0], 200, weight='weight')
    mecNodeList = searchNode(bLatency, pathCUMe)
    mecNodeList = nodeQuene(top, mecNodeList, 2, slice.type)
    for nodeMEC in mecNodeList:
        serverMEC, pathbront, lightback, bpFlag = nodeLinkSearch(exisNode, exisWave, serverCU[0], G, top, nodeMEC,
                                                                 slice, 2, bBandwidh)
        if serverMEC:
            break
    if not serverMEC:
        # 阻塞MEC
        markBp(top, bpFlag, slice)
        # top.blockNumForNode += 1
        # print('line 133 block for MEC or backhaul')
        return 0
    exisNode.append(serverMEC)
    exisWave.append(pathbront)
    totalPath.append(lightback)
    i = 0
    # print('DU,CU,MEC node:')
    # print('=======节点=====更新前',G.nodes[exisNode[0][0]])
    for node in exisNode:
        # print(node)
        top.updateServerVM(G, node[0], node[1], node[2], -slice.resource[i], i, slice.level)
        markNodeSliceAdd(top, node[0], slice.id)
        i += 1
    # print('========节点====更新后',G.nodes[exisNode[0][0]])
    slice.DU = exisNode[0]
    slice.CU = exisNode[1]
    slice.MEC = exisNode[2]

    # print('f,m,b paht:')
    # for i in range(3):
    #    print(totalPath[i],exisWave[i])
    # print('arrive： link before ')
    top.updateLightPath(G, totalPath[0], exisWave[0], -fBandwidh)
    top.updateLightPath(G, totalPath[1], exisWave[1], -mBandwidh)
    top.updateLightPath(G, totalPath[2], exisWave[2], -bBandwidh)
    slice.front = [totalPath[0], exisWave[0], fBandwidh]
    slice.mid = [totalPath[1], exisWave[1], mBandwidh]
    slice.back = [totalPath[2], exisWave[2], bBandwidh]
    # print('arrive： link after ')
    return 1


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def getNodeSlice(top, dosId):
    tempSliceDic = {}
    nodeSliceDic = top.nodeSlice[dosId]
    for key, value in nodeSliceDic.items():
        # print(key,':',value)
        if value > 0:
            tempSliceDic[key] = value
    return tempSliceDic


def expNum(num):
    return 1 / (1 + math.exp(-num))


def rankSlice(tempSliceDic, sliceDic, currTime):
    '''
    tempDic: slice in node
    output ranked slice
    '''
    sliceScore = {}
    for fkey, value in tempSliceDic.items():
        if value > 0:
            tempSlice = sliceDic[fkey]
            c1 = expNum(tempSlice.reliability)
            c2 = expNum(tempSlice.level)
            c3 = expNum(tempSlice.transLatency[0]) + expNum(tempSlice.transLatency[1]) + expNum(
                tempSlice.transLatency[2])
            c4 = expNum(tempSlice.bandwidth[0]) + expNum(tempSlice.bandwidth[1]) + expNum(tempSlice.bandwidth[2])
            c5 = expNum(len(tempSlice.list_AAU)) + expNum(tempSlice.resource[0]) + expNum(
                tempSlice.resource[0]) + expNum(tempSlice.resource[2])
            ht = tempSlice.endTime - tempSlice.arrTime
            st = currTime - tempSlice.arrTime
            # loss = 0
            loss = c2
            # loss =(ht-st)/ht
            # loss = 1 * (ht-st)/ht * 0.2*(c1+c2+c3+c4+c5)
            sliceScore[fkey] = loss
    sliceScore = sorted(sliceScore.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)

    res = []
    # print(sliceScore)
    for key in sliceScore:
        res.append(key[0])
    # for key,value in sliceScore.items():
    #     res.append(key)
    #     print(key)

    return res[::-1]


def clearNode(G, top, dosId):
    vmNum = 0
    serverNum = inputs.vmNumeberPerServer
    if dosId < 100:
        temp = copy.deepcopy(top.serverNumAE)
        tempVmlevel = copy.deepcopy(top.vmLevelAE)
        tempVmType = copy.deepcopy(top.vmTypeAE)
        vmNum = inputs.aeNum
    elif dosId < 200 and dosId > 100:
        temp = copy.deepcopy(top.serverNumMN)
        tempVmlevel = copy.deepcopy(top.vmLevelMN)
        tempVmType = copy.deepcopy(top.vmTypeMN)
        vmNum = inputs.mnNum
    else:
        temp = copy.deepcopy(top.serverNumME)
        tempVmlevel = copy.deepcopy(top.vmLevelME)
        tempVmType = copy.deepcopy(top.vmTypeME)
        vmNum = inputs.meNum
    G.nodes[dosId]['type'] = tempVmType
    G.nodes[dosId]['servers'] = temp
    G.nodes[dosId]['vmLevel'] = tempVmlevel
    top.nodeSlice[dosId] = {}
    top.nodeScore[dosId] = inputs.DosStrength * inputs.nodeScore[dosId]
    influncedVmSize = (int)((1 - inputs.DosStrength) * vmNum * serverNum)
    markDosVm(G, dosId, influncedVmSize)


def markDosVm(G, nodeId, influncedSize):
    '''
    输入：受影响服务器
    输出：遍历服务器vm ，置空

    '''
    count = 0
    serverNum = len(G.nodes[nodeId]['servers'])
    for server in range(serverNum):
        for vm in range(len(G.nodes[nodeId]['servers'][server])):
            if (count == influncedSize):
                break
            count += 1
            G.nodes[nodeId]['servers'][server, vm] = 0
        if (count == influncedSize):
            break


def cleanLink(G, top, dosId, sliceDic, sliceList):
    for id in sliceList:
        tempSlice = sliceDic[id]
        if tempSlice.DU[0] == dosId:
            front = tempSlice.front
            top.updateLightPath(G, front[0], front[1], front[2])
            tempSlice.front = None
            if tempSlice.CU[0] != dosId:
                mid = tempSlice.mid
                top.updateLightPath(G, mid[0], mid[1], mid[2])
                tempSlice.mid = None
        if tempSlice.CU[0] == dosId:
            mid = tempSlice.mid
            top.updateLightPath(G, mid[0], mid[1], mid[2])
            tempSlice.mid = None
            if tempSlice.MEC[0] != dosId:
                back = tempSlice.back
                top.updateLightPath(G, back[0], back[1], back[2])
                tempSlice.back = None
        if tempSlice.MEC[0] == dosId:
            back = tempSlice.back
            top.updateLightPath(G, back[0], back[1], back[2])
            tempSlice.back = None


def clearNodeLink(G, top, dosId, sliceDic, sliceList):
    clearNode(G, top, dosId)
    cleanLink(G, top, dosId, sliceDic, sliceList)


def migrate_rl(G, top, tempSlice, dosId, mnOrAe, action):
    # (1,1,5)
    place_code = ACTION[action]
    migrate_place = [ACTION_DIC.get(place_code[0]), ACTION_DIC.get(place_code[1]), ACTION_DIC.get(place_code[2])]

    pathStart = 0
    # mnorAe == 1 Mn, 0 - ae
    exisNode = []
    exisWave = []
    totalPath = []
    updateType = []  # 统计哪些更新了
    serverDU = None
    left_flag = -1

    if tempSlice.DU[0] == dosId:
        # ===============if ！=   du, cu(DOS), mec// cu正常, mec节点
        # 寻找节点和链路资源，并占位
        updateType.append(0)
        AE = math.ceil(1.0 * tempSlice.list_AAU[0] / inputs.num_BS)
        pathAeMeBefore = nx.shortest_path(G, AE, 200, weight='weight')
        pathAeMe = pathFilter(pathAeMeBefore, mnOrAe)
        duNodeList = searchNode(tempSlice.transLatency[0], pathAeMe)
        duNodeList = nodeQuene(top, duNodeList, 0, tempSlice.type)
        if migrate_place[0] not in duNodeList:
            left_flag = -1
            # markBp(top, 0, tempSlice)
            # return 0
        else:
            duNodeList = [migrate_place[0]]
        bpFlag = 0
        for nodeDU in duNodeList:
            if nodeDU is not None:
                if nodeDU != dosId:
                    serverDU, pathfront, lightfront, bpFlag = nodeLinkSearch(exisNode, exisWave, AE, G, top, nodeDU,
                                                                             tempSlice, 0, tempSlice.bandwidth[0])
                    if serverDU: break
            else:
                serverDU = None
        if not serverDU:
            # markBp(top, bpFlag, tempSlice)
            # return 0
            left_flag =1
        else:
            exisNode.append(serverDU)
            exisWave.append(pathfront)
            totalPath.append(lightfront)

    serverCU = None

    if tempSlice.CU[0] == dosId:
        #
        # 寻找节点和链路资源，并占位
        updateType.append(1)
        if tempSlice.DU[0] == dosId:
            pathStart = serverDU[0]
        else:
            pathStart = tempSlice.DU[0]
        pathDUMeBefore = nx.shortest_path(G, pathStart, 200, weight='weight')
        pathDUMe = pathFilter(pathDUMeBefore, mnOrAe)
        cuNodeList = searchNode(tempSlice.transLatency[1], pathDUMe)
        cuNodeList = nodeQuene(top, cuNodeList, 1, tempSlice.type)
        if migrate_place[1] not in cuNodeList:
            # markBp(top, 0, tempSlice)
            # return 0
            left_flag =1
        else:
            cuNodeList = [migrate_place[1]]
        bpFlag = 0
        for nodeCU in cuNodeList:
            if nodeCU is not None:
                if nodeCU != dosId:
                    serverCU, pathmront, lightmid, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top,
                                                                           nodeCU, tempSlice, 1, tempSlice.bandwidth[1])
                    if serverCU: break
            else:
                serverCU = None
        if not serverCU:
            # markBp(top, bpFlag, tempSlice)
            # return 0
            left_flag =1
        else:
            exisNode.append(serverCU)
            exisWave.append(pathmront)
            totalPath.append(lightmid)
    elif tempSlice.DU[0] == dosId and tempSlice.CU[0] != dosId:
        # upadte b link
        # search link
        updateType.append(1)
        midwave, midpath = linkSearch(exisWave, serverDU[0], G, top, tempSlice.CU[0], tempSlice.bandwidth[1])
        if midwave == -10:
            # print('366, block for link')
            left_flag = 1
            # return 0
        else:
            # updateLink.append(1)
            exisNode.append(-1)
            exisWave.append(midwave)
            totalPath.append(midpath)
    if left_flag == 1:
        pass
    else:
        if tempSlice.MEC[0] == dosId:
            # 寻找节点和链路资源，并占位，更新
            updateType.append(2)
            if tempSlice.CU[0] == dosId:
                pathStart = serverCU[0]
            else:
                pathStart = tempSlice.CU[0]

            pathCUMeBefore = nx.shortest_path(G, pathStart, 200, weight='weight')
            pathCUMe = pathFilter(pathCUMeBefore, mnOrAe)
            # pathCUMe = nx.shortest_path(G, pathStart, 200, weight='weight')
            mecNodeList = searchNode(tempSlice.transLatency[2], pathCUMe)
            mecNodeList = nodeQuene(top, mecNodeList, 2, tempSlice.type)
            if migrate_place[2] not in mecNodeList:
                # markBp(top, 0, tempSlice)
                # return 0
                left_flag = 1
            else:
                mecNodeList = [migrate_place[2]]
            serverMEC = None
            bpFlag = 0
            for nodeMEC in mecNodeList:
                if nodeMEC is not None:
                    if nodeMEC != dosId:
                        serverMEC, pathbront, lightback, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top,
                                                                                 nodeMEC, tempSlice, 2,
                                                                                 tempSlice.bandwidth[2])
                        if serverMEC: break
            if not serverMEC:
                # 阻塞MEC
                # markBp(top, bpFlag, tempSlice)
                # return 0
                left_flag =1
            else:
                exisNode.append(serverMEC)
                exisWave.append(pathbront)
                totalPath.append(lightback)
        elif tempSlice.CU[0] == dosId and tempSlice.MEC[0] != dosId:
            # upadte b link
            # search link
            if left_flag == 1:
                pass
            else:
                updateType.append(2)
                backwave, backpath = linkSearch(exisWave, serverCU[0], G, top, tempSlice.MEC[0], tempSlice.bandwidth[2])
                if backwave == -10:
                    # print('403, block for link')
                    left_flag =1
                    # markBp(top, 1, tempSlice)
                    # return 0
                else:
                    # updateLink.append(1)
                    exisNode.append(-1)
                    exisWave.append(backwave)
                    totalPath.append(backpath)
    if left_flag == -1:
        i = 0
        # print('DU,CU,MEC node:')
        # print('=======节点=====更新前',G.nodes[exisNode[0][0]])
        for node in exisNode:
            # print(node)
            if node != -1:
                markNodeSliceAdd(top, node[0], tempSlice.id)
                if updateType[i] == 0:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[0], updateType[i], tempSlice.level)
                    tempSlice.DU = exisNode[i]
                if updateType[i] == 1:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[1], updateType[i], tempSlice.level)
                    tempSlice.CU = exisNode[i]
                if updateType[i] == 2:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[2], updateType[i], tempSlice.level)
                    tempSlice.MEC = exisNode[i]
            i += 1

        # print('========节点====更新后',G.nodes[exisNode[0][0]])

        # print('f,m,b paht:')
        # for i in range(len(updateType)):
        # print(totalPath[i],exisWave[i])
        # print('arrive： link before ')
        for j in range(len(updateType)):
            if updateType[j] == 0:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[0])
                tempSlice.front = [totalPath[j], exisWave[j], tempSlice.bandwidth[0]]
            if updateType[j] == 1:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[1])
                tempSlice.mid = [totalPath[j], exisWave[j], tempSlice.bandwidth[1]]
            if updateType[j] == 2:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[2])
                tempSlice.back = [totalPath[j], exisWave[j], tempSlice.bandwidth[2]]
        return 1
    else:

        exisNode = []
        exisWave = []
        totalPath = []
        updateType = []  # 统计哪些更新了
        serverDU = None

        if tempSlice.DU[0] == dosId:
            # ===============if ！=   du, cu(DOS), mec// cu正常, mec节点
            # 寻找节点和链路资源，并占位
            updateType.append(0)
            AE = math.ceil(1.0 * tempSlice.list_AAU[0] / inputs.num_BS)
            pathAeMeBefore = nx.shortest_path(G, AE, 200, weight='weight')
            pathAeMe = pathFilter(pathAeMeBefore, mnOrAe)
            duNodeList = searchNode(tempSlice.transLatency[0], pathAeMe)
            duNodeList = nodeQuene(top, duNodeList, 0, tempSlice.type)
            if migrate_place[0] not in duNodeList:
                markBp(top, 0, tempSlice)
                return 0
            else:
                duNodeList = [migrate_place[0]]
            bpFlag = 0
            for nodeDU in duNodeList:
                if nodeDU is not None:
                    if nodeDU != dosId:
                        serverDU, pathfront, lightfront, bpFlag = nodeLinkSearch(exisNode, exisWave, AE, G, top, nodeDU,
                                                                                 tempSlice, 0, tempSlice.bandwidth[0])
                        if serverDU: break
                else:
                    serverDU = None
            if not serverDU:
                markBp(top, bpFlag, tempSlice)
                return 0

            exisNode.append(serverDU)
            exisWave.append(pathfront)
            totalPath.append(lightfront)

        serverCU = None

        if tempSlice.CU[0] == dosId:
            #
            # 寻找节点和链路资源，并占位
            updateType.append(1)
            if tempSlice.DU[0] == dosId:
                pathStart = serverDU[0]
            else:
                pathStart = tempSlice.DU[0]
            pathDUMe = [103,104,105,200]
            cuNodeList = searchNode(tempSlice.transLatency[1], pathDUMe)
            cuNodeList = nodeQuene(top, cuNodeList, 1, tempSlice.type)
            if migrate_place[1] not in cuNodeList:
                markBp(top, 0, tempSlice)
                return 0
            else:
                cuNodeList = [migrate_place[1]]
            bpFlag = 0
            for nodeCU in cuNodeList:
                if nodeCU is not None:
                    if nodeCU != dosId:
                        serverCU, pathmront, lightmid, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top,
                                                                               nodeCU, tempSlice, 1, tempSlice.bandwidth[1])
                        if serverCU: break
                else:
                    serverCU = None
            if not serverCU:
                markBp(top, bpFlag, tempSlice)
                return 0
            exisNode.append(serverCU)
            exisWave.append(pathmront)
            totalPath.append(lightmid)
        elif tempSlice.DU[0] == dosId and tempSlice.CU[0] != dosId:
            # upadte b link
            # search link
            updateType.append(1)
            midwave, midpath = linkSearch(exisWave, serverDU[0], G, top, tempSlice.CU[0], tempSlice.bandwidth[1])
            if midwave == -10:
                # print('366, block for link')
                return 0
            else:
                # updateLink.append(1)
                exisNode.append(-1)
                exisWave.append(midwave)
                totalPath.append(midpath)

        if tempSlice.MEC[0] == dosId:
            # 寻找节点和链路资源，并占位，更新
            updateType.append(2)
            if tempSlice.CU[0] == dosId:
                pathStart = serverCU[0]
            else:
                pathStart = tempSlice.CU[0]

            pathCUMe = []
            if pathStart == 200:
                pathCUMe = [200]
            if pathStart == 105:
                pathCUMe = [105,200]
            if pathStart == 104:
                pathCUMe = [104,105,200]
            if pathStart == 102:
                pathCUMe = [102,103,104,105,200]
            if pathStart == 101:
                pathCUMe = [101,102,103,104,105,200]
            if pathStart < 100:
                pathCUMe = [103,104,105,200]
            # pathCUMe = nx.shortest_path(G, pathStart, 200, weight='weight')
            mecNodeList = searchNode(tempSlice.transLatency[2], pathCUMe)
            mecNodeList = nodeQuene(top, mecNodeList, 2, tempSlice.type)
            if migrate_place[2] not in mecNodeList:
                markBp(top, 0, tempSlice)
                return 0
            else:
                mecNodeList = [migrate_place[2]]
            serverMEC = None
            bpFlag = 0
            for nodeMEC in mecNodeList:
                if nodeMEC is not None:
                    if nodeMEC != dosId:
                        serverMEC, pathbront, lightback, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top,
                                                                                 nodeMEC, tempSlice, 2,
                                                                                 tempSlice.bandwidth[2])
                        if serverMEC: break
            if not serverMEC:
                # 阻塞MEC
                markBp(top, bpFlag, tempSlice)
                return 0
            exisNode.append(serverMEC)
            exisWave.append(pathbront)
            totalPath.append(lightback)
        elif tempSlice.CU[0] == dosId and tempSlice.MEC[0] != dosId:
            # upadte b link
            # search link
            if serverCU is None:
                pass
            else:
                updateType.append(2)
                backwave, backpath = linkSearch(exisWave, serverCU[0], G, top, tempSlice.MEC[0], tempSlice.bandwidth[2])
                if backwave == -10:
                    # print('403, block for link')
                    markBp(top, 1, tempSlice)
                    return 0
                else:
                    # updateLink.append(1)
                    exisNode.append(-1)
                    exisWave.append(backwave)
                    totalPath.append(backpath)

        i = 0
        # print('DU,CU,MEC node:')
        # print('=======节点=====更新前',G.nodes[exisNode[0][0]])
        for node in exisNode:
            # print(node)
            if node != -1:
                markNodeSliceAdd(top, node[0], tempSlice.id)
                if updateType[i] == 0:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[0], updateType[i], tempSlice.level)
                    tempSlice.DU = exisNode[i]
                if updateType[i] == 1:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[1], updateType[i], tempSlice.level)
                    tempSlice.CU = exisNode[i]
                if updateType[i] == 2:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[2], updateType[i], tempSlice.level)
                    tempSlice.MEC = exisNode[i]
            i += 1

        # print('========节点====更新后',G.nodes[exisNode[0][0]])

        # print('f,m,b paht:')
        # for i in range(len(updateType)):
        # print(totalPath[i],exisWave[i])
        # print('arrive： link before ')
        for j in range(len(updateType)):
            if updateType[j] == 0:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[0])
                tempSlice.front = [totalPath[j], exisWave[j], tempSlice.bandwidth[0]]
            if updateType[j] == 1:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[1])
                tempSlice.mid = [totalPath[j], exisWave[j], tempSlice.bandwidth[1]]
            if updateType[j] == 2:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[2])
                tempSlice.back = [totalPath[j], exisWave[j], tempSlice.bandwidth[2]]
        return 1


def migrate_left(G, top, tempSlice, dosId, mnOrAe):
    # (1,1,5)

    # mnorAe == 1 Mn, 0 - ae
    exisNode = []
    exisWave = []
    totalPath = []
    updateType = []  # 统计哪些更新了
    serverDU = None
    left_flag = -1
    if tempSlice.DU[0] == dosId:
        # ===============if ！=   du, cu(DOS), mec// cu正常, mec节点
        # 寻找节点和链路资源，并占位
        updateType.append(0)
        AE = math.ceil(1.0 * tempSlice.list_AAU[0] / inputs.num_BS)

        pathAeMeBefore = nx.shortest_path(G, AE, 200, weight='weight')
        pathAeMe = pathFilter(pathAeMeBefore, mnOrAe)
        duNodeList = searchNode(tempSlice.transLatency[0], pathAeMe)
        duNodeList = nodeQuene(top, duNodeList, 0, tempSlice.type)
        bpFlag = 0
        for nodeDU in duNodeList:
            if nodeDU != dosId:
                serverDU, pathfront, lightfront, bpFlag = nodeLinkSearch(exisNode, exisWave, AE, G, top, nodeDU,
                                                                         tempSlice, 0, tempSlice.bandwidth[0])
                if serverDU: break
        if not serverDU:
            # markBp(top, bpFlag, tempSlice)
            left_flag = 1
            # return 0
        else:
            exisNode.append(serverDU)
            exisWave.append(pathfront)
            totalPath.append(lightfront)
    serverCU = None
    if tempSlice.CU[0] == dosId:
        # 寻找节点和链路资源，并占位
        updateType.append(1)
        if tempSlice.DU[0] == dosId:
            pathStart = serverDU[0]
        else:
            pathStart = tempSlice.DU[0]
        pathDUMeBefore = nx.shortest_path(G, pathStart, 200, weight='weight')
        pathDUMe = pathFilter(pathDUMeBefore, mnOrAe)
        cuNodeList = searchNode(tempSlice.transLatency[1], pathDUMe)
        cuNodeList = nodeQuene(top, cuNodeList, 1, tempSlice.type)
        bpFlag = 0
        for nodeCU in cuNodeList:
            if nodeCU != dosId:
                serverCU, pathmront, lightmid, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top, nodeCU,
                                                                       tempSlice, 1, tempSlice.bandwidth[1])
                if serverCU: break
        if not serverCU:
            left_flag = 1
            # markBp(top, bpFlag, tempSlice)
            # return 0
        else:
            exisNode.append(serverCU)
            exisWave.append(pathmront)
            totalPath.append(lightmid)
    elif tempSlice.DU[0] == dosId and tempSlice.CU[0] != dosId:
        # upadte b link
        # search link
        updateType.append(1)
        midwave, midpath = linkSearch(exisWave, serverDU[0], G, top, tempSlice.CU[0], tempSlice.bandwidth[1])
        if midwave == -10:
            # print('366, block for link')
            left_flag = 1
            # return 0
        else:
            # updateLink.append(1)
            exisNode.append(-1)
            exisWave.append(midwave)
            totalPath.append(midpath)

    if tempSlice.MEC[0] == dosId:
        # 寻找节点和链路资源，并占位，更新
        if left_flag == 1:
            pass
        else:
            updateType.append(2)
            if tempSlice.CU[0] == dosId:
                pathStart = serverCU[0]
            else:
                pathStart = tempSlice.CU[0]

            pathCUMeBefore = nx.shortest_path(G, pathStart, 200, weight='weight')
            pathCUMe = pathFilter(pathCUMeBefore, mnOrAe)
            # pathCUMe = nx.shortest_path(G, pathStart, 200, weight='weight')
            mecNodeList = searchNode(tempSlice.transLatency[2], pathCUMe)
            mecNodeList = nodeQuene(top, mecNodeList, 2, tempSlice.type)
            serverMEC = None
            bpFlag = 0
            for nodeMEC in mecNodeList:
                if nodeMEC != dosId:
                    serverMEC, pathbront, lightback, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top, nodeMEC,
                                                                             tempSlice, 2, tempSlice.bandwidth[2])
                    if serverMEC: break
            if not serverMEC:
                # 阻塞MEC
                left_flag = 1
                # markBp(top, bpFlag, tempSlice)
                # return 0
            else:
                exisNode.append(serverMEC)
                exisWave.append(pathbront)
                totalPath.append(lightback)
    elif tempSlice.CU[0] == dosId and tempSlice.MEC[0] != dosId:
        # upadte b link
        # search link
        if serverCU is None:
            pass
        else:
            updateType.append(2)
            backwave, backpath = linkSearch(exisWave, serverCU[0], G, top, tempSlice.MEC[0], tempSlice.bandwidth[2])
            if backwave == -10:
                # print('403, block for link')
                left_flag = 1

                # markBp(top, 1, tempSlice)
                # return 0
            else:
                # updateLink.append(1)
                exisNode.append(-1)
                exisWave.append(backwave)
                totalPath.append(backpath)
    if left_flag == -1:
        i = 0
        # print('DU,CU,MEC node:')
        # print('=======节点=====更新前',G.nodes[exisNode[0][0]])
        for node in exisNode:
            # print(node)
            if node != -1:
                markNodeSliceAdd(top, node[0], tempSlice.id)
                if updateType[i] == 0:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[0], updateType[i], tempSlice.level)
                    tempSlice.DU = exisNode[i]
                if updateType[i] == 1:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[1], updateType[i], tempSlice.level)
                    tempSlice.CU = exisNode[i]
                if updateType[i] == 2:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[2], updateType[i], tempSlice.level)
                    tempSlice.MEC = exisNode[i]
            i += 1

        # print('========节点====更新后',G.nodes[exisNode[0][0]])

        # print('f,m,b paht:')
        # for i in range(len(updateType)):
        # print(totalPath[i],exisWave[i])
        # print('arrive： link before ')
        for j in range(len(updateType)):
            if updateType[j] == 0:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[0])
                tempSlice.front = [totalPath[j], exisWave[j], tempSlice.bandwidth[0]]
            if updateType[j] == 1:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[1])
                tempSlice.mid = [totalPath[j], exisWave[j], tempSlice.bandwidth[1]]
            if updateType[j] == 2:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[2])
                tempSlice.back = [totalPath[j], exisWave[j], tempSlice.bandwidth[2]]
        return 1
    else:
        #right
        exisNode = []
        exisWave = []
        totalPath = []
        updateType = []  # 统计哪些更新了
        serverDU = None
        if tempSlice.DU[0] == dosId:
            # ===============if ！=   du, cu(DOS), mec// cu正常, mec节点
            # 寻找节点和链路资源，并占位
            updateType.append(0)
            AE = math.ceil(1.0 * tempSlice.list_AAU[0] / inputs.num_BS)

            pathAeMeBefore = nx.shortest_path(G, AE, 200, weight='weight')
            pathAeMe = pathFilter(pathAeMeBefore, mnOrAe)
            duNodeList = searchNode(tempSlice.transLatency[0], pathAeMe)
            duNodeList = nodeQuene(top, duNodeList, 0, tempSlice.type)
            bpFlag = 0
            for nodeDU in duNodeList:
                if nodeDU != dosId:
                    serverDU, pathfront, lightfront, bpFlag = nodeLinkSearch(exisNode, exisWave, AE, G, top, nodeDU,
                                                                             tempSlice, 0, tempSlice.bandwidth[0])
                    if serverDU: break
            if not serverDU:
                markBp(top, bpFlag, tempSlice)
                return 0

            exisNode.append(serverDU)
            exisWave.append(pathfront)
            totalPath.append(lightfront)
        serverCU = None
        if tempSlice.CU[0] == dosId:
            # 寻找节点和链路资源，并占位
            updateType.append(1)
            if tempSlice.DU[0] == dosId:
                pathStart = serverDU[0]
            else:
                pathStart = tempSlice.DU[0]
            pathDUMe = [103,104,105,200]
            cuNodeList = searchNode(tempSlice.transLatency[1], pathDUMe)
            cuNodeList = nodeQuene(top, cuNodeList, 1, tempSlice.type)
            bpFlag = 0
            for nodeCU in cuNodeList:
                if nodeCU != dosId:
                    serverCU, pathmront, lightmid, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top, nodeCU,
                                                                           tempSlice, 1, tempSlice.bandwidth[1])
                    if serverCU: break
            if not serverCU:
                markBp(top, bpFlag, tempSlice)
                return 0
            exisNode.append(serverCU)
            exisWave.append(pathmront)
            totalPath.append(lightmid)
        elif tempSlice.DU[0] == dosId and tempSlice.CU[0] != dosId:
            # upadte b link
            # search link
            updateType.append(1)
            midwave, midpath = linkSearch(exisWave, serverDU[0], G, top, tempSlice.CU[0], tempSlice.bandwidth[1])
            if midwave == -10:
                # print('366, block for link')
                return 0
            else:
                # updateLink.append(1)
                exisNode.append(-1)
                exisWave.append(midwave)
                totalPath.append(midpath)

        if tempSlice.MEC[0] == dosId:
            # 寻找节点和链路资源，并占位，更新
            updateType.append(2)
            if tempSlice.CU[0] == dosId:
                pathStart = serverCU[0]
            else:
                pathStart = tempSlice.CU[0]

            pathCUMe = []
            if pathStart == 200:
                pathCUMe = [200]
            if pathStart == 105:
                pathCUMe = [105,200]
            if pathStart == 104:
                pathCUMe = [104,105,200]
            if pathStart == 102:
                pathCUMe = [102,103,104,105,200]
            if pathStart == 101:
                pathCUMe = [101,102,103,104,105,200]
            if pathStart < 100:
                pathCUMe = [103,104,105,200]
            # pathCUMe = nx.shortest_path(G, pathStart, 200, weight='weight')
            mecNodeList = searchNode(tempSlice.transLatency[2], pathCUMe)
            mecNodeList = nodeQuene(top, mecNodeList, 2, tempSlice.type)
            serverMEC = None
            bpFlag = 0
            for nodeMEC in mecNodeList:
                if nodeMEC != dosId:
                    serverMEC, pathbront, lightback, bpFlag = nodeLinkSearch(exisNode, exisWave, pathStart, G, top, nodeMEC,
                                                                             tempSlice, 2, tempSlice.bandwidth[2])
                    if serverMEC: break
            if not serverMEC:
                # 阻塞MEC
                markBp(top, bpFlag, tempSlice)
                return 0
            exisNode.append(serverMEC)
            exisWave.append(pathbront)
            totalPath.append(lightback)
        elif tempSlice.CU[0] == dosId and tempSlice.MEC[0] != dosId:
            # upadte b link
            # search link
            updateType.append(2)
            backwave, backpath = linkSearch(exisWave, serverCU[0], G, top, tempSlice.MEC[0], tempSlice.bandwidth[2])
            if backwave == -10:
                # print('403, block for link')
                markBp(top, 1, tempSlice)
                return 0
            else:
                # updateLink.append(1)
                exisNode.append(-1)
                exisWave.append(backwave)
                totalPath.append(backpath)

        i = 0
        # print('DU,CU,MEC node:')
        # print('=======节点=====更新前',G.nodes[exisNode[0][0]])
        for node in exisNode:
            # print(node)
            if node != -1:
                markNodeSliceAdd(top, node[0], tempSlice.id)
                if updateType[i] == 0:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[0], updateType[i], tempSlice.level)
                    tempSlice.DU = exisNode[i]
                if updateType[i] == 1:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[1], updateType[i], tempSlice.level)
                    tempSlice.CU = exisNode[i]
                if updateType[i] == 2:
                    top.updateServerVM(G, node[0], node[1], node[2], -tempSlice.resource[2], updateType[i], tempSlice.level)
                    tempSlice.MEC = exisNode[i]
            i += 1

        # print('========节点====更新后',G.nodes[exisNode[0][0]])

        # print('f,m,b paht:')
        # for i in range(len(updateType)):
        # print(totalPath[i],exisWave[i])
        # print('arrive： link before ')
        for j in range(len(updateType)):
            if updateType[j] == 0:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[0])
                tempSlice.front = [totalPath[j], exisWave[j], tempSlice.bandwidth[0]]
            if updateType[j] == 1:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[1])
                tempSlice.mid = [totalPath[j], exisWave[j], tempSlice.bandwidth[1]]
            if updateType[j] == 2:
                top.updateLightPath(G, totalPath[j], exisWave[j], -tempSlice.bandwidth[2])
                tempSlice.back = [totalPath[j], exisWave[j], tempSlice.bandwidth[2]]
        return 1

def releaseSlice(G, top, dosId, sliceDic, sliceId):
    tempSlice = sliceDic[sliceId]
    # 后续释放忽略
    tempSlice.real = False
    # AAU:
    aau = tempSlice.list_AAU
    for char in aau:
        top.aau_map_slice_num[char] -= 1
    top.idle_AAU_num += tempSlice.aau

    # link
    if tempSlice.front:
        front = tempSlice.front
        top.updateLightPath(G, front[0], front[1], front[2])
        tempSlice.front = None
    if tempSlice.mid:
        mid = tempSlice.mid
        top.updateLightPath(G, mid[0], mid[1], mid[2])
        tempSlice.mid = None
    if tempSlice.back:
        back = tempSlice.back
        top.updateLightPath(G, back[0], back[1], back[2])
        tempSlice.back = None
    # node 
    du = tempSlice.DU
    cu = tempSlice.CU
    mec = tempSlice.MEC
    if tempSlice.DU[0] != dosId:
        top.updateServerVM(G, du[0], du[1], du[2], tempSlice.resource[0], 0, tempSlice.level)
        markNodeSliceDel(top, du[0], tempSlice.id)
    if tempSlice.CU[0] != dosId:
        top.updateServerVM(G, cu[0], cu[1], cu[2], tempSlice.resource[1], 1, tempSlice.level)
        markNodeSliceDel(top, cu[0], tempSlice.id)
    if tempSlice.MEC[0] != dosId:
        top.updateServerVM(G, mec[0], mec[1], mec[2], tempSlice.resource[2], 2, tempSlice.level)
        markNodeSliceDel(top, mec[0], tempSlice.id)


def dealSlice(G, top, dosId, sliceDic, sliceList, currentTime):
    ss_action = []
    #fl_action = []
    #a_action = []
    #slice_info_all = []
    slice_info_ss = []
    #slice_info_fl = []
    for id in sliceList:
        res = 2
        top.totalSliceNumDosed += 1
        tempSlice = sliceDic[id]
        info = -1
        # 迁MN
        info = migrate_left(G, top, tempSlice, dosId, 1)
        if info == 0:
            top.blockForDos += 1
            # 迁移阻塞，释放并标记资源
            releaseSlice(G, top, dosId, sliceDic, id)
        reward = countReward(info, tempSlice, currentTime)
        lossCount(top, tempSlice, reward, res, info)



        if info != 0:
            slice_info_ss.append(tempSlice.type)
            # mara action
            one_action = []
            if tempSlice.DU[0] > 100:
                one_action.append(tempSlice.DU[0])
            else:
                one_action.append(-1)
            if tempSlice.CU[0] > 100:
                one_action.append(tempSlice.CU[0])
            else:
                one_action.append(-1)
            if tempSlice.MEC[0] > 100:
                one_action.append(tempSlice.MEC[0])
            else:
                one_action.append(-1)
            ss_action.append(copy.deepcopy(one_action))

    fp = open('./action_ss_base.dat', 'a')
    fp.write(str(ss_action) + '\n')
    fp.write('slice_type: ' + str(slice_info_ss) + '\n')
    fp.close()



    return top.blockForDos / top.totalSliceNumDosed


def lossCount(top, slice, reward, migOrNot, blockOrNot):
    if migOrNot == 0: top.migAeNum += 1

    if migOrNot == 1: top.notMigNum += 1

    if migOrNot == 2: top.migMnNum += 1

    if slice.level == 1:
        top.l1TotalNumber += 1
    if slice.level == 2:
        top.l2TotalNumber += 1
    if slice.level == 3:
        top.l3TotalNumber += 1
    if slice.level == 4:
        top.l4TotalNumber += 1

    if migOrNot == 2 and slice.level == 1:
        top.l1MigNumberMn += 1
    if migOrNot == 2 and slice.level == 2:
        top.l2MigNumberMn += 1
    if migOrNot == 2 and slice.level == 3:
        top.l3MigNumberMn += 1
    if migOrNot == 2 and slice.level == 4:
        top.l4MigNumberMn += 1

    if migOrNot == 0 and slice.level == 1:
        top.l1MigNumber += 1
    if migOrNot == 0 and slice.level == 2:
        top.l2MigNumber += 1
    if migOrNot == 0 and slice.level == 3:
        top.l3MigNumber += 1
    if migOrNot == 0 and slice.level == 4:
        top.l4MigNumber += 1

    if blockOrNot == 0:
        top.blockLoss += reward
    elif migOrNot == 0:
        top.migLoss += reward
    top.totalLoss += reward


def rlDealSlice(G, top, dosId, sliceDic, id, action, currentTime):
    # action变为DU、CU、MEC位置
    # action == 0,1,2  0，不迁移；  1， MN；  2， AE
    # for id in sliceList:

    top.totalSliceNumDosed += 1

    tempSlice = sliceDic[id]
    # RL 决定 迁移到到哪个位置
    info = migrate_rl(G, top, tempSlice, dosId, 1, action)

    if info == 0:
        top.blockForDos += 1
        # 迁移阻塞，释放并标记资源
        releaseSlice(G, top, dosId, sliceDic, id)
    reward = countReward(info, tempSlice, currentTime)
    lossCount(top, tempSlice, reward, action, info)
    return reward, info


def countReward(blockOrNot, slice, currentTime):
    #
    res = 0
    leftTime = slice.endTime - currentTime
    leftTime /= (slice.endTime - slice.arrTime)
    cr = slice.resource[0] + slice.resource[1] + slice.resource[2]
    bw = slice.bandwidth[0] + slice.bandwidth[1] + slice.bandwidth[2]
    rwd = 0.5 * expNum(cr) + 0.5 * expNum(bw)
    if blockOrNot == 0:
        # 阻塞
        res = -rwd * 10
    else:
        res = rwd

    return res


def rlDelDDoS(G, top, dosId, sliceDic, currTime):
    # 遍历获取Dos 节点的切片信息
    tempSliceDic = getNodeSlice(top, dosId)
    if not tempSliceDic:
        return
    # print('node ',dosId,' is being attacked by DDoS')

    # 排序决定重构顺序
    sliceList = rankSlice(tempSliceDic, sliceDic, currTime)
    # print(G.nodes[dosId])
    # 清除Dos节点,链路
    clearNodeLink(G, top, dosId, sliceDic, sliceList)

    return sliceList

    # 强化学习，决定切片迁移或者留在原节点
    # dealSlice(G,top, dosId, sliceDic, sliceList)


def delDDoS(G, top, dosId, sliceDic, currTime):
    # 遍历获取Dos 节点的切片信息
    tempSliceDic = getNodeSlice(top, dosId)
    if not tempSliceDic:
        return
    # print('node ',dosId,' is being attacked by DDoS')

    # 排序决定重构顺序
    sliceList = rankSlice(tempSliceDic, sliceDic, currTime)
    # print(G.nodes[dosId])
    # 清除Dos节点,链路
    clearNodeLink(G, top, dosId, sliceDic, sliceList)

    dosblock = dealSlice(G, top, dosId, sliceDic, sliceList, currTime)
    return dosblock


def pathFilter(lightPath, aeOrMn):
    # input nodepath   aeOrMn  0 - ae,1 - mn
    # outout newPath
    res = []
    if aeOrMn == 0:
        for i in lightPath:
            if i < 100:
                res.append(i)
    else:
        for i in lightPath:
            if i > 100:
                res.append(i)
    return res


def getNodeLinkScore(G, nodeId):
    # return nodeLinkScore
    res = 0

    if nodeId == 101:
        res = min(G[101][102]['pathScore'], G[102][101]['pathScore'])/80  # 检查score值
    if nodeId == 102:
        res = min(G[102][103]['pathScore'], G[103][102]['pathScore'])/80  # 检查score值
    if nodeId == 103:
        res = min(G[103][104]['pathScore'], G[104][103]['pathScore'])/80  # 检查score值

    if nodeId == 104:
        res = min(G[104][105]['pathScore'], G[105][104]['pathScore'])/80  # 检查score值

    if nodeId == 105:
        res = min(G[105][200]['pathScore'], G[200][105]['pathScore'])/80  # 检查score值

    if nodeId == 200:
        res = min(G[200][101]['pathScore'], G[101][200]['pathScore'])/80  # 检查score值

    return res
