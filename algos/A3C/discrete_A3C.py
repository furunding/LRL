"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import copy
import torch
import torch.nn as nn
from .utils import v_wrap, set_init, push_and_pull, record
import torch.multiprocessing as mp
from .shared_adam import SharedAdam
import os

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

# env = gym.make('CartPole-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n
class Worker(mp.Process):
    def __init__(self, env, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        # self.lnet = copy.deepcopy(gnet)           # local network
        from .net import DNet
        self.lnet = DNet(4, 2)
        # self.env = gym.make('CartPole-v0').unwrapped
        # print(self.name)
        self.env = env(name)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                # if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    v_loss, pi_loss, entropy = push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, v_loss, pi_loss, entropy, ep_r)
                        break

                # if done:  # update global and assign to local net
                #     # sync
                #     v_loss, pi_loss, entropy = push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                #     buffer_s, buffer_a, buffer_r = [], [], []
                #     record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, v_loss, pi_loss, entropy, ep_r)
                #     break
                s = s_
                total_step += 1
        self.res_queue.put(None)