from algos.A3C import *
import torch.multiprocessing as mp

from env.catsim import CatsimSA

N_S = 6
N_A = 4

if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(CatsimSA, gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(20)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()