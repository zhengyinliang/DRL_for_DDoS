import InputConstants

def store_args(self, res_path):
    inputs = InputConstants.Inputs()

    # topology

    aauNums = inputs.aauNums
    resoucers = inputs.resoucers
    transLatency = inputs.transLatency
    bandWidth = inputs.bandWidth
    linkLatency = inputs.linkLatency
    oeoLatency = inputs.oeoLatency
    waveCapabity = inputs.waveCapabity
    VMCapability = inputs.VMCapability
    vmNumeberPerServer = inputs.vmNumeberPerServer
    aeNum = inputs.aeNum
    mnNum = inputs.mnNum
    meNum = inputs.meNum
    # 2x25x10=500,1500,9000
    serviceRate = inputs.serviceRate
    trafficLoad = inputs.trafficLoad
    # 5000 500 1000
    sliceNum = inputs.sliceNum
    aeToae = inputs.aeToae
    mnTomn = inputs.mnTomn
    wavelengths = inputs.wavelengths
    num_BS = inputs.num_BS
    DosStrength = inputs.DosStrength
    sliceArrived = inputs.sliceArrived
    #
    # # RL
    epsilon_arg = inputs.epsilon_arg

    n_actions = self.n_actions
    # we do not input ttl
    x_dim_p = self.x_dim_p  # node number
    x_dim_v = self.x_dim_v

    num_layers = self.num_layers
    layer_size = self.layer_size
    regu_scalar = self.regu_scalar


    gamma = self.gamma  # penalty on future reward
    batch_size = self.batch_size # probably smaller value, e.g., 50, would be better for higher blocking probability (see JLT)
    total_epoNum = self.total_epoNum


    fp = open('./' + res_path + '/train_args', 'a')
    fp.write('==========================\n')
    stri = 'sliceArrived :' + str(sliceArrived)
    fp.write(stri + '\n')

    stri = 'serviceRate :' + str(serviceRate)
    fp.write(stri + '\n')

    stri = 'trafficLoad :' + str(trafficLoad)
    fp.write(stri + '\n')

    stri = 'sliceNum :' + str(sliceNum)
    fp.write(stri + '\n')

    stri = 'aauNums: '+str(aauNums)
    fp.write(stri+'\n')
    stri = 'resoucers :' + str(resoucers)
    fp.write(stri + '\n')

    stri = 'transLatency :' + str(transLatency)
    fp.write(stri + '\n')

    stri = 'bandWidth :' + str(bandWidth)
    fp.write(stri + '\n')

    stri = 'linkLatency :' + str(linkLatency)
    fp.write(stri + '\n')

    stri = 'oeoLatency :' + str(oeoLatency)
    fp.write(stri + '\n')

    stri = 'waveCapabity :' + str(waveCapabity)
    fp.write(stri + '\n')

    stri = 'vmLatcncy :' + str(bandWidth)
    fp.write(stri + '\n')

    stri = 'VMCapability :' + str(VMCapability)
    fp.write(stri + '\n')

    stri = 'vmNumeberPerServer :' + str(vmNumeberPerServer)
    fp.write(stri + '\n')

    stri = 'aeNum :' + str(aeNum)
    fp.write(stri + '\n')

    stri = 'mnNum :' + str(mnNum)
    fp.write(stri + '\n')

    stri = 'meNum :' + str(meNum)
    fp.write(stri + '\n')


    stri = 'aeToae :' + str(aeToae)
    fp.write(stri + '\n')

    stri = 'mnTomn :' + str(mnTomn)
    fp.write(stri + '\n')

    stri = 'wavelengths :' + str(wavelengths)
    fp.write(stri + '\n')

    stri = 'num_BS :' + str(num_BS)
    fp.write(stri + '\n')

    stri = 'DosStrength :' + str(DosStrength)
    fp.write(stri + '\n')



    fp.write('===========RL==================\n')
    # # RL


    stri = 'n_actions :' + str(n_actions)
    fp.write(stri + '\n')


    stri = 'x_dim_p :' + str(x_dim_p)
    fp.write(stri + '\n')


    stri = 'x_dim_v :' + str(x_dim_v)
    fp.write(stri + '\n')


    stri = 'num_layers :' + str(num_layers)
    fp.write(stri + '\n')


    stri = 'layer_size :' + str(layer_size)
    fp.write(stri + '\n')

    stri = 'regu_scalar :' + str(regu_scalar)
    fp.write(stri + '\n')

    stri = 'gamma :' + str(gamma)
    fp.write(stri + '\n')

    stri = 'batch_size :' + str(batch_size)
    fp.write(stri + '\n')

    stri = 'total_epoNum :' + str(total_epoNum)
    fp.write(stri + '\n')

    stri = 'epsilon_arg :' + str(epsilon_arg)
    fp.write(stri + '\n')
    fp.close()
