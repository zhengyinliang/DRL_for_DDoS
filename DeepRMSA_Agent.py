from __future__ import division

import random

import numpy as np
import tensorflow as tf

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
import numpy as np
import copy
from AC_Net import AC_Net
import tensorflow as tf
import scipy.signal
from time import time
import time
import os
import train_args

from env1 import SliceEnv
import functions as fun
import InputConstants

input = InputConstants.Inputs()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DeepRMSA_Agent():

    def __init__(self, total_epoNum,
                 id,
                 trainer,
                 gamma,
                 batch_size,
                 model_path,
                 global_episodes,
                 regu_scalar,
                 x_dim_p,
                 x_dim_v,
                 n_actions,
                 num_layers,
                 layer_size,
                 ):
        self.env = SliceEnv()
        self.name = 'agent_' + str(id)
        self.trainer = trainer
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.regu_scalar = regu_scalar

        self.global_episodes = global_episodes  #
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + self.name)
        #
        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        self.local_network = AC_Net(scope=self.name,
                                    trainer=self.trainer,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)
        self.update_local_ops = self.update_target_graph('global', self.name)

        #
        self.total_epoNum = total_epoNum

        # self.all_ones = [[1 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        # self.all_negones = [[0 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)

        #

    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    # Discounting function used to calculate discounted returns.
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def train(self, espisode_buff, sess, value_est):
        espisode_buff = np.array(espisode_buff)
        input_p = espisode_buff[:self.batch_size, 0]
        input_v = espisode_buff[:self.batch_size, 1]
        actions = espisode_buff[:self.batch_size, 2]
        rewards = espisode_buff[:, 3]
        values = espisode_buff[:, 4]

        self.rewards_plus = np.asarray(rewards.tolist() + [value_est])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        discounted_rewards = np.append(discounted_rewards, 0)  # --
        discounted_rewards_batch = discounted_rewards[:self.batch_size] - (
                    self.gamma ** self.batch_size) * discounted_rewards[self.batch_size:]  # --
        self.value_plus = np.asarray(values.tolist() + [value_est])
        '''advantages = rewards + self.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages)
        advantages = np.append(advantages, 0) # --
        advantages = advantages[:self.batch_size] - (self.gamma**self.batch_size)*advantages[self.batch_size:] # --'''
        advantages = discounted_rewards_batch - self.value_plus[:self.batch_size]

        # a filtering scheme, filter out 20% largest and smallest elements from 'discounted_rewards_batch'
        '''sorted_reward = np.argsort(discounted_rewards_batch)
        input_p = input_p[sorted_reward[10:-10]]
        input_v = input_v[sorted_reward[10:-10]]
        discounted_rewards_batch = discounted_rewards_batch[sorted_reward[10:-10]]
        actions = actions[sorted_reward[10:-10]]
        advantages = advantages[sorted_reward[10:-10]]'''

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_network.target_v: discounted_rewards_batch,
                     self.local_network.Input_p: np.vstack(input_p),
                     self.local_network.Input_v: np.vstack(input_v),
                     self.local_network.actions: actions,
                     self.local_network.advantages: advantages}
        sum_value_losss, sum_policy_loss, sum_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, _, _, regu_loss_policy, regu_loss_value = sess.run(
            [self.local_network.loss_value,
             self.local_network.loss_policy,
             self.local_network.entropy,
             self.local_network.grad_norms_policy,
             self.local_network.grad_norms_value,
             self.local_network.var_norms_policy,
             self.local_network.var_norms_value,
             self.local_network.apply_grads_policy,
             self.local_network.apply_grads_value,
             self.local_network.regu_loss_policy,
             self.local_network.regu_loss_value],
            feed_dict=feed_dict)
        return sum_value_losss / self.batch_size, sum_policy_loss / self.batch_size, sum_entropy / self.batch_size, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, regu_loss_policy / self.batch_size, regu_loss_value / self.batch_size

    def rmsa(self, sess, coord, saver):
        ep_count = 0
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        episode_buffer = []

        action_onehot = [x for x in range(self.n_actions)]

        # update local dnn with the global one
        sess.run(self.update_local_ops)

        epsilon = 1

        res_path = "./res"
        time_tag = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        res_path += time_tag
        fun.mkdir(res_path)  # 调用函数

        print('Starting ' + self.name)

        train_args.store_args(self, res_path)

        with sess.as_default(), sess.graph.as_default():

            total_step = 0
            markCount = 0
            while not coord.should_stop():
                ss_action = []
                slice_info_ss = []
                mean_value_losss = 0
                mean_policy_loss = 0
                mean_entropy = 0
                total_step += 1
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                num_blocks = 0
                # my env
                random.seed(1)
                s = self.env.reset()
                top = self.env.top
                G = self.env.G
                eventQuene = self.env.eventQuene
                count = 0
                markDosId = []
                markCount += 1
                markDosNode = -1
                # mark block befor Dos
                bpOfNodeBeforeDos = 0
                bpOfBwBeforeDos = 0

                bp = -1
                totalReward = -1
                dosBlock = -1

                # begin an episode
                while not eventQuene.empty():
                    next_item = eventQuene.get()
                    # 如果业务到达
                    # print('count ', count)
                    if next_item[1] == 'arrive':
                        count += 1
                        top.sliceArrived += 1
                        slice = next_item[2]
                        self.env.sliceDic[slice.id] = slice
                        currentTime = slice.arrTime
                        if count == input.sliceArrived:
                            top.flag = True
                            markDosNode += 1
                            # print(markDosNode)
                            self.env.dosId = self.env.dosList[markDosNode]
                            # 随机选一个节点或者多个节点发生DDoS
                            dosId = self.env.dosId
                            # print('第',markCount,'轮训练','   dosId',dosId)
                            markDosId.append(dosId)
                            bpOfNodeBeforeDos = copy.deepcopy(top.bpOfMappingForNode)
                            bpOfBwBeforeDos = copy.deepcopy(top.bpOfMappingForBW)
                            sliceList = fun.rlDelDDoS(G, top, dosId, self.env.sliceDic, currentTime)

                            ep_r = 0

                            if sliceList:
                                episode_size = len(sliceList)
                                for index in range(len(sliceList)):
                                    # self.batch_size = len(sliceList)
                                    # state
                                    Input_feature = []
                                    ep_t = sliceList[index]

                                    # check in
                                    self.env.set_slice(ep_t)
                                    Input_feature = self.env._get_state()
                                    Input_feature = np.array(Input_feature)
                                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))

                                    blocking = 0
                                    # 观察value
                                    # Take an action using probabilities from policy network output.
                                    prob_dist, value, entro = sess.run(
                                        [self.local_network.policy, self.local_network.value,
                                         self.local_network.entropy],
                                        feed_dict={self.local_network.Input_p: Input_feature,
                                                   self.local_network.Input_v: Input_feature})
                                    pp = prob_dist[0]
                                    # [0.00086836 0.00357914 0.00357802 0.00478219 0.00254058 0.0030082, 0.01609759 0.00319894 0.00434318 0.00051378 0.00250096 0.00049609, 0.00181239 0.00334174 0.00105248 0.03642332 0.00180417 0.01148524, 0.00236244 0.00289154 0.00122864 0.00996341 0.00518862
                                    assert not np.isnan(entro)

                                    action_id = -1
                                    # epsilon = 1
                                    if random.random() < epsilon:
                                        # 随机探索
                                        # action_id = np.random.choice(action_onehot, p = pp)
                                        #action_id = np.random.randint(0, len(pp))
                                        action_id = np.random.choice(action_onehot, p=pp)

                                    else:
                                        # 取概率最大的action，返回[  ]值最大的index
                                        action_id = np.argmax(pp)
                                        #action_id = np.random.choice(action_onehot, p=pp)

                                    s_, r, done, b = self.env.step(action_id, dosId, ep_t, currentTime)

                                    # reward
                                    r_t = r
                                    if b == 0:
                                        num_blocks += 1
                                        print("index:", index, " action:", action_id, " blocked")
                                    else:
                                        slice_mark = self.env.sliceDic[ep_t]
                                        one_action = []
                                        if slice_mark.DU[0] > 100:
                                            one_action.append(slice_mark.DU[0])
                                        else:
                                            one_action.append(-1)
                                        if slice_mark.CU[0] > 100:
                                            one_action.append(slice_mark.CU[0])
                                        else:
                                            one_action.append(-1)
                                        if slice_mark.MEC[0] > 100:
                                            one_action.append(slice_mark.MEC[0])
                                        else:
                                            one_action.append(-1)
                                        ss_action.append(copy.deepcopy(one_action))
                                        slice_info_ss.append(self.env.sliceDic[ep_t].type)
                                        print("index:", index, " action:", action_id)

                                    episode_reward += r_t
                                    total_steps += 1
                                    episode_step_count += 1

                                    if episode_count < len(sliceList)/self.batch_size:  # for warm-up
                                        continue

                                    # store experience
                                    episode_buffer.append([Input_feature, Input_feature, action_id, r_t, value[0, 0]])
                                    episode_values.append(value[0, 0])

                                    if len(episode_buffer) == 2 * self.batch_size - 1:
                                        mean_value_losss, mean_policy_loss, mean_entropy, grad_norms_policy, grad_norms_value, \
                                        var_norms_policy, var_norms_value, regu_loss_policy, regu_loss_value = self.train(
                                            episode_buffer, sess, 0.0)
                                        del (episode_buffer[:self.batch_size])
                                        sess.run(self.update_local_ops)  # if we want to synchronize local with global every a training is performed
                                        # epsilon = np.max([epsilon - 1e-5, 0.05])    # 100轮后随机概率减少
                                        epsilon = np.max([epsilon - input.epsilon_arg, 0.05])  # 100轮后随机概率减少

                                # end of an episode
                                episode_count += 1


                                if episode_count <= len(sliceList) / self.batch_size + 2:  # for warm-up
                                    break

                                # sess.run(self.update_local_ops) # if we want to synchronize local with global every episode is finished

                                # if episode_count <= (600//episode_size): # for warm-up
                                #    continue
                                bp = num_blocks / episode_size
                                bpTotal = top.sliceBlocked / top.sliceArrived
                                dosBlock = bp

                                ep_count += 1
                                self.episode_blocking.append(bp)
                                self.episode_rewards.append(episode_reward)
                                self.episode_mean_values.append(np.mean(episode_values))

                                # Periodically save model parameters, and summary statistics.
                                sample_step = int(500 / episode_size)
                                # if episode_count % sample_step == 0 and episode_count != 0:

                                if self.name == 'agent_0':
                                    sess.run(self.increment)
                                if ep_count >= self.total_epoNum:
                                    coord.request_stop()

                            bpOfNodeBeforeDos = top.bpOfMappingForNode
                            bpOfBwBeforeDos = top.bpOfMappingForBW

                            # break
                            if input.countAfterDos:
                                top.flag = False
                            else:
                                break
                        # 解决映射
                        res = fun.delServiceChain(G, top, slice)
                        if res == 0:
                            top.sliceBlocked += 1
                            # print(G.nodes.data())
                            if slice.list_AAU:
                                top.idle_AAU_num += slice.aau
                            for index in slice.list_AAU:
                                top.aau_map_slice_num[index] -= 1
                            continue
                        slice.real = True

                    # 如果业务离开
                    elif next_item[1] == 'leave':
                        slice = next_item[2]
                        # 判断业务是否部署， 是则释放，否责跳过
                        str = "slice %d 业务离开，开始释放切片资源" % slice.id
                        # print(str)
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
                            # print('=======节点=====更新前',G.nodes[du[0]])

                            top.updateLightPath(G, fpath, fWave, fBandwidh)
                            top.updateServerVM(G, du[0], du[1], du[2], slice.resource[0], 0, slice.level)
                            fun.markNodeSliceDel(top, du[0], slice.id)
                            top.updateLightPath(G, mpath, mWave, mBandwidh)
                            top.updateServerVM(G, cu[0], cu[1], cu[2], slice.resource[1], 1, slice.level)
                            fun.markNodeSliceDel(top, cu[0], slice.id)
                            top.updateLightPath(G, bpath, bWave, bBandwidh)
                            top.updateServerVM(G, mec[0], mec[1], mec[2], slice.resource[2], 2, slice.level)
                            fun.markNodeSliceDel(top, mec[0], slice.id)
                            # print('=======节点=====更新后',G.nodes[du[0]])

                            for char in aau:
                                top.aau_map_slice_num[char] -= 1
                            top.idle_AAU_num += slice.aau
                    else:
                        print('event wrong')

                if episode_count <= len(sliceList) / self.batch_size +2:  # for warm-up
                    continue

                print("before Dos")
                print(input.sliceArrived, " slice mapping")
                print("sliceArrived: ", input.sliceArrived)
                beforeBP = (bpOfNodeBeforeDos + bpOfBwBeforeDos) / input.sliceArrived

                print(input.sliceArrived, 'slcie bp:', beforeBP)
                print('blockOfMappingForNode :', bpOfNodeBeforeDos)
                print('blockOfMappingForBW :', bpOfBwBeforeDos)

                print('====================')
                print('DDoS start')
                print('dosedNumber: ', top.totalSliceNumDosed)
                print('top.blockLoss:', top.blockLoss)
                print('dosBlock: ', top.blockForDos / top.totalSliceNumDosed)
                print('bpOfDosForNode :', top.bpOfDosForNode)
                print('bpOfDosForBW :', top.bpOfDosForBW)
                dos_node = top.bpOfDosForNode/top.totalSliceNumDosed
                dos_bw = top.bpOfDosForBW/top.totalSliceNumDosed

                migProbal1 = (top.l1MigNumber + top.l1MigNumberMn) / top.l1TotalNumber
                migProbal2 = (top.l2MigNumber + top.l2MigNumberMn) / top.l2TotalNumber
                migProbal3 = (top.l3MigNumber + top.l3MigNumberMn) / top.l3TotalNumber
                migProbal4 = (top.l4MigNumber + top.l4MigNumberMn) / top.l4TotalNumber

                print('l1_TotalNumber: ', top.l1TotalNumber)
                print('l2_TotalNumber: ', top.l2TotalNumber)
                print('l3_TotalNumber: ', top.l3TotalNumber)
                print('l4_TotalNumber: ', top.l4TotalNumber)

                # print('flag :',top.flag)
                print('l1BpNumber :', top.l1BpNumber)
                print('l2BpNumber :', top.l2BpNumber)
                print('l3BpNumber :', top.l3BpNumber)
                print('l4BpNumbe :', top.l4BpNumber)

                print("===== after Dos ========")
                afterSlice = input.sliceNum - input.sliceArrived
                afterBP = ((top.bpOfMappingForNode + top.bpOfMappingForBW) - (
                            bpOfNodeBeforeDos + bpOfBwBeforeDos)) / afterSlice

                print(afterSlice, " slice mapping")
                print("sliceArrived: ", afterSlice)
                print(afterSlice, 'slcie bp:', afterBP)
                print('blockOfMappingForNode :', (top.bpOfMappingForNode - bpOfNodeBeforeDos))
                print('blockOfMappingForBW :', (top.bpOfMappingForBW - bpOfBwBeforeDos))

                bp = top.blockForDos / top.totalSliceNumDosed
                totalReward = top.blockLoss + top.migLoss

                if self.name == 'agent_0':
                    fun.markFunction(dos_node, dos_bw, slice_info_ss, total_step,ss_action, res_path, beforeBP, afterBP, totalReward, dosBlock, top.blockLoss, top.migLoss,
                                     migProbal1, migProbal2, migProbal3, migProbal4, top.notMigNum, top.migAeNum,
                                     top.migMnNum,
                                     (top.l1TotalNumber - top.l1MigNumber - top.l1MigNumberMn),
                                     (top.l2TotalNumber - top.l2MigNumber - top.l2MigNumberMn),
                                     (top.l3TotalNumber - top.l3MigNumber - top.l3MigNumberMn),
                                     (top.l4TotalNumber - top.l4MigNumber - top.l4MigNumberMn), top.l1MigNumber,
                                     top.l2MigNumber, top.l3MigNumber, top.l4MigNumber,
                                     top.l1MigNumberMn, top.l2MigNumberMn, top.l3MigNumberMn, top.l4MigNumberMn)
                    print('Blocking Probability = ', bp)
                    # print('Action Distribution', actionss.count(0)/len(actionss))
                    # print('Mean Resource Utilization =', np.mean(resource_util))
                    # store value prediction
                    fp = open('./' + res_path + '/value.dat', 'a')
                    fp.write('%f\n' % np.mean(episode_values))
                    fp.close()
                    # store value loss
                    fp = open('./' + res_path + '/value_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_value_losss))
                    fp.close()
                    # store policy loss
                    fp = open('./' + res_path + '/policy_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_policy_loss))
                    fp.close()
                    # store entroy
                    fp = open('./' + res_path + '/entropy.dat', 'a')
                    fp.write('%f\n' % float(mean_entropy))
                    fp.close()

        # 绘制结果图
        fun.draw_res(res_path)
