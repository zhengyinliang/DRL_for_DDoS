# name_list=['BP','entropy','policy_loss','value_loss','value']
name_list = ['bpBeforeDos', 'bpAfterDos', 'DosBp', 'reward', 'blockLoss', 'migLoss', 'migProbal1', 'migProbal2',
             'migProbal3', 'migProbal4',
             'policy_loss', 'entropy', 'value_loss', 'value', 'notMigNum', 'migAeNum', 'migMnNum', 'l1NotMig',
             'l2NotMig', 'l3NotMig', 'l4NotMig', 'l1MigAe', 'l2MigAe', 'l3MigAe', 'l4MigAe', 'l1MigMn', 'l2MigMn',
             'l3MigMn', 'l4MigMn']
for name in name_list:
    f = open('./{}.dat'.format(name), 'r+')
    f.truncate()
    f.close()
