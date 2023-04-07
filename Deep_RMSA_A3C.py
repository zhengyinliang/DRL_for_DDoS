import random

import logging
logging.basicConfig(filename="rl.log", level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',filemode='w')

import numpy as np
import tensorflow as tf

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)
import sys, os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import threading
from AC_Net import AC_Net
from DeepRMSA_Agent import DeepRMSA_Agent
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# author Xiaoliang Chen, xlichen@ucdavis.edu
# copyright NGNS lab @ucdavis

# key features: uniform/nonuniform traffic distribution; window-based training; policy embedded with epsilon-greedy approach

# -----------------------------------------------------------


n_actions = 29

# we do not input ttl
x_dim_p = 55  # node number
x_dim_v = 55
num_layers = 5
layer_size = 128
regu_scalar = 1e-4

max_cpu = 4

gamma = 0.95  # penalty on future reward
batch_size = 50  # probably smaller value, e.g., 50, would be better for higher blocking probability (see JLT)
total_epoNum = 10

load_model = False  # True
model_path = 'model'

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5)
    # trainer = tf.train.RMSPropOptimizer(learning_rate = 1e-5, decay = 0.99, epsilon = 0.0001)
    master_network = AC_Net(scope='global',
                            trainer=None,
                            x_dim_p=x_dim_p,
                            x_dim_v=x_dim_v,
                            n_actions=n_actions,
                            num_layers=num_layers,
                            layer_size=layer_size,
                            regu_scalar=regu_scalar)  # Generate global network

    # num_agents = multiprocessing.cpu_count() # Set workers to number of available CPU threads

    num_agents = 1  # Set workers to number of available CPU threads
    print("CPU  ", num_agents)
    if num_agents > max_cpu:
        num_agents = max_cpu  # as most assign max_cpu CPUs
    agents = []
    # Create worker classes
    for i in range(num_agents):
        agents.append(DeepRMSA_Agent(total_epoNum, i, trainer, gamma, batch_size, model_path, global_episodes,
                                     regu_scalar, x_dim_p, x_dim_v, n_actions, num_layers, layer_size))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # Start the "rmsa" process for each agent in a separate threat.
    agent_threads = []
    for agent in agents:
        agent_rmsa = lambda: agent.rmsa(sess, coord, saver)
        t = threading.Thread(target=(agent_rmsa))
        t.start()
        agent_threads.append(t)
    coord.join(agent_threads)
