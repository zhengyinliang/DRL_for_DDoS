'''
步骤：
1、创建图拓扑，并存储，配置每个节点的资源
2、设定切片属性，用业务生成器确定动态的业务
3、按业务进行资源分配
'''
import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
import graph
import event
import functions as fun
import InputConstants
input = InputConstants.Inputs()
import copy


if __name__ == '__main__':
#创建top，网路图G
    #增加一个traffic load 的值迭代后续，传入event
    sliceList = event.event()
    for ep in range(3):
        random.seed(1)
        print('第', ep ,'轮') 
        sliceDic = {}  # 记录slice 对象
        top = graph.topology()
        G = top.G
        eventQuene  = event.initEventQuene(sliceList)
        count = 0
        block = 0
        dosBlock = 0

        # mark block befor Dos
        bpOfNodeBeforeDos = 0
        bpOfBwBeforeDos = 0

        count_leave = 0

        while not eventQuene.empty():
            next_item = eventQuene.get()
            #如果业务到达
            #print('count ', count)
            if next_item[1] == 'arrive':
                top.sliceArrived += 1
                count += 1
                slice = next_item[2]
                sliceDic[slice.id] = slice
                # 3000
                if count == input.sliceArrived:
                    top.flag = True
                    dosId = 103
                    bpOfNodeBeforeDos = copy.deepcopy(top.bpOfMappingForNode)
                    bpOfBwBeforeDos = copy.deepcopy(top.bpOfMappingForBW)
                    dosBlock = fun.delDDoS(G, top, dosId, sliceDic, slice.arrTime)
                    
                    if input.countAfterDos:
                        top.flag = False
                    else:
                        break
                str = "slice %d 业务到达，开始网络映射" %slice.id
                #print(str)
                #解决映射
                res = fun.delServiceChain(G,top,slice)
                if res == 0:
                    top.sliceBlocked +=1
                    #print(G.nodes.data())
                    if slice.list_AAU:
                        top.idle_AAU_num += slice.aau
                    for index in slice.list_AAU:
                        top.aau_map_slice_num[index] -= 1
                    continue
                slice.real = True
            
            #如果业务离开
            elif next_item[1] == 'leave':
                slice = next_item[2]
                count_leave += 1
                #判断业务是否部署， 是则释放，否责跳过
                str = "slice %d 业务离开，开始释放切片资源" %slice.id
                #print(str)
                if slice.real == True:
                    aau = slice.list_AAU
                    fpath = slice.front[0]
                    fWave = slice.front[1]
                    fBandwidh = slice.front[2]
                    mpath = slice.mid[0]
                    mWave = slice.mid[1]
                    mBandwidh = slice.mid[2]
                    bpath = slice.back[0]
                    bWave = slice.back[1]
                    bBandwidh = slice.back[2]
                    du = slice.DU
                    cu = slice.CU
                    mec = slice.MEC
                    #print('=======节点=====更新前',G.nodes[du[0]])

                    top.updateLightPath(G,fpath, fWave, fBandwidh)
                    top.updateServerVM(G, du[0], du[1], du[2], slice.resource[0], 0, slice.level)
                    fun.markNodeSliceDel(top,du[0],slice.id)
                    top.updateLightPath(G,mpath, mWave, mBandwidh)
                    top.updateServerVM(G, cu[0], cu[1], cu[2], slice.resource[1], 1, slice.level)
                    fun.markNodeSliceDel(top,cu[0],slice.id)
                    top.updateLightPath(G,bpath, bWave, bBandwidh)
                    top.updateServerVM(G, mec[0], mec[1], mec[2], slice.resource[2], 2, slice.level)
                    fun.markNodeSliceDel(top,mec[0],slice.id)
                    #print('=======节点=====更新后',G.nodes[du[0]])
                
                    for char in aau:
                        top.aau_map_slice_num[char] -= 1
                    top.idle_AAU_num += slice.aau
                
            else:
                print('event wrong')
        # print('总阻塞率：',block/input.sliceNum)
        # print('AAU阻塞率：',top.blockNumForAAU/input.sliceNum)
        # print('link阻塞率：',top.blockNumForLink/input.sliceNum)
        # print('Node阻塞率：',top.blockNumForNode/input.sliceNum)
        glbReward = top.blockLoss + top.migLoss 
        bp = top.sliceBlocked / top.sliceArrived
        top.blockLoss
        top.migLoss

        print("before Dos")
        print(input.sliceArrived," slice mapping")
        print("sliceArrived: ",input.sliceArrived)
        beforeBP = (bpOfNodeBeforeDos + bpOfBwBeforeDos) / input.sliceArrived

        print(input.sliceArrived,'slcie bp:',beforeBP)
        print('blockOfMappingForNode :',bpOfNodeBeforeDos)
        print('blockOfMappingForBW :',bpOfBwBeforeDos)

        print('====================')
        print('DDoS start')
        print('dosedNumber: ',top.totalSliceNumDosed)
        print("totalLoss:",glbReward)
        print('top.blockLoss:',top.blockLoss)
        print('top.migLoss:',top.migLoss)
#        print('dosBlock: ',top.blockForDos/top.totalSliceNumDosed)
        print('bpOfDosForNode :',top.bpOfDosForNode)
        print('bpOfDosForBW :',top.bpOfDosForBW)

        # migProbal1 = (top.l1MigNumber+top.l1MigNumberMn) /top.l1TotalNumber
        # migProbal2 = (top.l2MigNumber+top.l2MigNumberMn) /top.l2TotalNumber
        # migProbal3 = (top.l3MigNumber+top.l3MigNumberMn) /top.l3TotalNumber
        # migProbal4 = (top.l4MigNumber+top.l4MigNumberMn) /top.l4TotalNumber

        print('l1_MigNumberAe: ', top.l1MigNumber)
        print('l2_MigNumberAe: ', top.l2MigNumber)
        print('l3_MigNumberAe: ', top.l3MigNumber)
        print('l4_MigNumberAe: ', top.l4MigNumber)

        print('l1_MigNumberMn: ', top.l1MigNumberMn)
        print('l2_MigNumberMn: ', top.l2MigNumberMn)
        print('l3_MigNumberMn: ', top.l3MigNumberMn)
        print('l4_MigNumberMn: ', top.l4MigNumberMn)

        print('l1_TotalNumber: ', top.l1TotalNumber)
        print('l2_TotalNumber: ', top.l2TotalNumber)
        print('l3_TotalNumber: ', top.l3TotalNumber)
        print('l4_TotalNumber: ', top.l4TotalNumber)

        # print('migProbal1:',migProbal1)
        # print('migProbal2:',migProbal2)
        # print('migProbal3:',migProbal3)
        # print('migProbal4:',migProbal4)

        #print('flag :',top.flag)
        print('l1BpNumber :',top.l1BpNumber)
        print('l2BpNumber :',top.l2BpNumber)
        print('l3BpNumber :',top.l3BpNumber)
        print('l4BpNumbe :',top.l4BpNumber)


        print("===== after Dos ========")
        afterSlice = input.sliceNum-input.sliceArrived
        afterBP =((top.bpOfMappingForNode + top.bpOfMappingForBW) - (bpOfNodeBeforeDos + bpOfBwBeforeDos) )/ afterSlice
        
        
        print(afterSlice," slice mapping")
        print("sliceArrived: ",afterSlice)
        print(afterSlice,'slcie bp:',afterBP)
        print('blockOfMappingForNode :',(top.bpOfMappingForNode-bpOfNodeBeforeDos))
        print('blockOfMappingForBW :',(top.bpOfMappingForBW - bpOfBwBeforeDos))