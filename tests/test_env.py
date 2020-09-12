import time
import gym

if __name__ == "__main__":
    env = CatsimSA(0)
    s = env.reset()
    print(s)
    time.sleep(10)
    s_, r, done, _ = env.step(0)
    print(s_)