bs = [[-1, 102, 200], [-1, 102, 200], [-1, -1, 102], [-1, 102, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 102, 200], [-1, 102, 200], [-1, 102, 200], [-1, 200, 200], [-1, 200, 200], [-1, -1, 102], [-1, 200, 200], [-1, -1, 102], [-1, 200, 200], [-1, 200, 200], [-1, 200, 200], [-1, 200, 200], [-1, 200, 200], [-1, 200, 200], [-1, 200, 200], [-1, -1, 102], [-1, 200, 200], [-1, -1, 102], [-1, -1, 102], [-1, 200, 200]]

bs_type = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]


rl = [[-1, 101, 200], [-1, 104, 200], [-1, -1, 102], [-1, 104, 200], [-1, -1, 102], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 104, 200], [-1, 101, 200], [-1, 101, 200], [-1, 104, 200], [-1, 101, 200], [-1, 105, 200], [-1, -1, 102], [-1, -1, 102], [-1, 104, 200], [-1, 101, 200], [-1, 104, 200], [-1, 101, 200], [-1, 104, 200], [-1, 104, 200], [-1, 104, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 101, 200], [-1, 105, 200], [-1, 104, 200], [-1, 101, 200], [-1, 105, 200], [-1, -1, 102], [-1, 101, 200], [-1, -1, 102], [-1, 104, 200], [-1, 104, 200], [-1, 101, 200], [-1, 101, 200], [-1, 104, 200], [-1, 101, 200], [-1, 101, 200], [-1, -1, 102], [-1, 104, 200], [-1, -1, 102], [-1, -1, 102], [-1, 101, 200]]
rl_type = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]


ba_urllc = 0
ba_eMBB = 0
ba_101_eMBB = 0
ba_102_eMBB = 0
ba_104_eMBB = 0
ba_105_eMBB = 0
ba_200_eMBB = 0
ba_101_uRLLC = 0
ba_102_uRLLC = 0
ba_104_uRLLC = 0
ba_105_uRLLC = 0
ba_200_uRLLC = 0

print(len(bs_type))
for i in bs_type:
    if i == 1:
        ba_urllc +=1
    else:
        ba_eMBB += 1
for index in range(len(bs_type)):
    if bs_type[index] == 0:
        if bs[index][1] == 101:
            ba_101_eMBB += 1
        if bs[index][2] == 101:
            ba_101_eMBB += 1
        if bs[index][1] == 102:
            ba_102_eMBB += 1
        if bs[index][2] == 102:
            ba_102_eMBB += 1
        if bs[index][1] == 104:
            ba_104_eMBB += 1
        if bs[index][2] == 104:
            ba_104_eMBB += 1
        if bs[index][1] == 105:
            ba_105_eMBB += 1
        if bs[index][2] == 105:
            ba_105_eMBB += 1
        if bs[index][1] == 200:
            ba_200_eMBB += 1
        if bs[index][2] == 200:
            ba_200_eMBB += 1
    else:
        if bs[index][1] == 101:
            ba_101_uRLLC += 1
        if bs[index][2] == 101:
            ba_101_uRLLC += 1
        if bs[index][1] == 102:
            ba_102_uRLLC += 1
        if bs[index][2] == 102:
            ba_102_uRLLC += 1
        if bs[index][1] == 104:
            ba_104_uRLLC += 1
        if bs[index][2] == 104:
            ba_104_uRLLC += 1
        if bs[index][1] == 105:
            ba_105_uRLLC += 1
        if bs[index][2] == 105:
            ba_105_uRLLC += 1
        if bs[index][1] == 200:
            ba_200_uRLLC += 1
        if bs[index][2] == 200:
            ba_200_uRLLC += 1
ba_mig_eMBB = [ba_101_eMBB,ba_102_eMBB,ba_104_eMBB,ba_105_eMBB,ba_200_eMBB]
ba_mig_uRLLC = [ba_101_uRLLC,ba_102_uRLLC,ba_104_uRLLC,ba_105_uRLLC,ba_200_uRLLC]
print("=========base========")
print("eMBB: "+str(ba_mig_eMBB))
print("uRLLC: "+str(ba_mig_uRLLC))
print(ba_urllc)
print(ba_eMBB)

print("===========")

print(len(rl_type))
rl_101_eMBB = 0
rl_102_eMBB = 0
rl_104_eMBB = 0
rl_105_eMBB = 0
rl_200_eMBB = 0
rl_101_uRLLC = 0
rl_102_uRLLC = 0
rl_104_uRLLC = 0
rl_105_uRLLC = 0
rl_200_uRLLC = 0

rl_urllc = 0
rl_eMBB = 0


for index in range(len(rl_type)):
    if rl_type[index] == 0:
        rl_eMBB += 1
        if rl[index][1] == 101:
            rl_101_eMBB += 1
        if rl[index][2] == 101:
            rl_101_eMBB += 1
        if rl[index][1] == 102:
            rl_102_eMBB += 1
        if rl[index][2] == 102:
            rl_102_eMBB += 1
        if rl[index][1] == 104:
            rl_104_eMBB += 1
        if rl[index][2] == 104:
            rl_104_eMBB += 1
        if rl[index][1] == 105:
            rl_105_eMBB += 1
        if rl[index][2] == 105:
            rl_105_eMBB += 1
        if rl[index][1] == 200:
            rl_200_eMBB += 1
        if rl[index][2] == 200:
            rl_200_eMBB += 1
    else:
        rl_urllc +=1
        if rl[index][1] == 101:
            rl_101_uRLLC += 1
        if rl[index][2] == 101:
            rl_101_uRLLC += 1
        if rl[index][1] == 102:
            rl_102_uRLLC += 1
        if rl[index][2] == 102:
            rl_102_uRLLC += 1
        if rl[index][1] == 104:
            rl_104_uRLLC += 1
        if rl[index][2] == 104:
            rl_104_uRLLC += 1
        if rl[index][1] == 105:
            rl_105_uRLLC += 1
        if rl[index][2] == 105:
            rl_105_uRLLC += 1
        if rl[index][1] == 200:
            rl_200_uRLLC += 1
        if rl[index][2] == 200:
            rl_200_uRLLC += 1
rl_mig_eMBB = [rl_101_eMBB,rl_102_eMBB,rl_104_eMBB,rl_105_eMBB,rl_200_eMBB]
rl_mig_uRLLC = [rl_101_uRLLC,rl_102_uRLLC,rl_104_uRLLC,rl_105_uRLLC,rl_200_uRLLC]
print(rl_mig_eMBB)
print(rl_mig_uRLLC)

print(rl_urllc)
print(rl_eMBB)
