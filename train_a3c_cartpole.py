import datetime
import tensorboardX
import torch.multiprocessing as mp

import gym

from algos.A3C import *

summary_writer = tensorboardX.SummaryWriter('log/cartpole.{}'.\
    format(datetime.datetime.now().strftime(r'%d%H%M')))
env_maker = lambda x: gym.make("CartPole-v0")

N_S = 4
N_A = 2

if __name__ == "__main__":
    gnet = DNet(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(env_maker, gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    # while True:
    #     r = res_queue.get()
    #     if r is not None:
    #         g_ep = r
    #         # g_ep, reward, v_loss, pi_loss, entropy = r
    #         # print(g_ep)
    #         # summary_writer.add_scalar("reward", reward, g_ep)
    #         # summary_writer.add_scalar("v_loss", v_loss, g_ep)
    #         # summary_writer.add_scalar("pi_loss", pi_loss, g_ep)
    #         # summary_writer.add_scalar("entropy", entropy, g_ep)
    #     else:
    #         break
    [w.join() for w in workers]

    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()