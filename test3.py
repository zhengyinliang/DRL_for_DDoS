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
