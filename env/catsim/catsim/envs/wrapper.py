import time
import random
import subprocess
import math
import gym

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .interface.env_cmd import EnvCmd
from .interface.env_manager import EnvManager
from .interface.env_client import EnvClient
from .interface.env_def import BLUE_AIRPORT_ID, RED_AIRPORT_ID, UnitType
from .config import *

class CatSimSA(gym.Env):

    '''single agent version of catsim environment'''

    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7, ))
        self.action_space = spaces.Discrete(4)
        self.sim_speed = 50
        self.decision_period = 20
        self.env_manager = EnvManager(
            self._get_env_id(),
            base_map_port,
            scene_name,
            image_name=image_name,
            docker_name_prefix=docker_name_prefix,
            sim_server_ip='127.0.0.1'
        )
        self._start_docker()

    def __del__(self):
        self.env_manager.stop_docker()

    def _get_env_id(self):
        p = subprocess.Popen("docker ps | grep {} | grep {} | awk '{print $(NF)}' | grep -o '[0-9]\+'".format(image_name, docker_name_prefix), stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        # decode()将bytes对象转成str对象, strip()删除头尾字符空白符
        # split()默认以分隔符, 包括空格, 换行符\n, 制表符\t来对字符串进行分割
        out_str = out.decode().strip().split()

    def _start_docker(self):
        # 查找是否存在同名的旧容器, 有的话先删除再启动新环境
        # 捕获终端输出结果
        p = subprocess.Popen('docker ps -a --filter name=^/{}$'.format(self.env_manager.docker_name), stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        # decode()将bytes对象转成str对象, strip()删除头尾字符空白符
        # split()默认以分隔符, 包括空格, 换行符\n, 制表符\t来对字符串进行分割
        out_str = out.decode()
        str_split = out_str.strip().split()
        if self.env_manager.docker_name in str_split:
            print('删除同名容器\n', out_str)
            self.env_manager.stop_docker()
        self.env_manager.start_docker()  # 启动新环境
        # time.sleep(10)
        print("start docker success!")

    def _connect_docker(self, rpyc_port):
        """根据映射出来的宿主机端口号rpyc_port，与容器内的仿真平台建立rpyc连接"""
        while True:
            try:
                env_client = EnvClient(sim_server_ip, rpyc_port)
                self.observation = env_client.get_observation()
                if len(self.observation['red']['units']) != 0:
                    return env_client

            except Exception as e:
                print(e)
                print("rpyc connect failed")

            time.sleep(3)

    @staticmethod
    def calc_distance(pos1, pos2):
        return math.sqrt((pos1["X"]-pos2["X"])**2 + (pos1["Y"]-pos2["Y"])**2 + (pos1["Z"]-pos2["Z"])**2)
 
    def _make_state(self):
        if self.red_fighter is None or self.blue_fighter is None:
            return np.array([0]*6)
        else:
            return np.array([self.red_fighter["X"] / 100000, self.red_fighter["Y"] / 100000, \
                            self.blue_fighter["X"] / 100000, self.blue_fighter["Y"] / 100000, \
                                self.calc_distance(self.red_fighter, self.blue_fighter) / 100000, int(len(self.obs["red"]["rockets"]) > 0)])

    def _make_reward_done(self):
        # self.red_fighter = next((entity for entity in self.obs["red"]["units"] if entity["ID"]==self.red_fighter_id), None)
        # self.blue_fighter = next((entity for entity in self.obs["red"]["qb"] if entity["ID"]==self.blue_fighter_id), None)
        if self.blue_fighter is None and self.red_fighter is None:
            reward = 0
            done = True
            print("tie!")
        elif self.red_fighter is None:
            reward = -1
            done = True
            print("loss!")
        elif self.blue_fighter is None:
            reward = 1
            done = True
            print("win!")
        else:
            reward = 0.01
            done = (self.obs["sim_time"] > 1800)
            if done: print("tie!")
        return reward, done

    def update_state(self):
        self.obs = self.env_client.get_observation()
        self.red_fighter = next((entity for entity in self.obs["red"]["units"] if entity["ID"]==self.red_fighter_id), None)
        self.blue_fighter = next((entity for entity in self.obs["red"]["qb"] if entity["ID"]==self.blue_fighter_id), None)

    def reset(self):
        self.env_manager.reset()
        self.env_client = self._connect_docker(self.env_manager.get_server_port())
        self.env_client.take_action([EnvCmd.make_simulation("SPEED", "", self.sim_speed)])
        time.sleep(1)
        # deploy
        self.obs = self.env_client.get_observation()
        blue_aew_id = next(entity["ID"] for entity in self.obs["blue"]["units"] if entity["LX"]==UnitType.AWACS)
        red_aew_id = next(entity["ID"] for entity in self.obs["red"]["units"] if entity["LX"]==UnitType.AWACS)
        self.env_client.take_action([
            # blue deploy
            EnvCmd.make_awcs_areapatrol(blue_aew_id, -20000, 0, 8000, 0, 20000, 20000, 250, 7200, 0), 
            EnvCmd.make_takeoff_areapatrol(BLUE_AIRPORT_ID, 1, UnitType.A2A, -60000, 80000 * random.choice([-1, 0, 1]), 8000, 0, 20000, 20000, 250, 7200),
            # red deploy
            EnvCmd.make_awcs_areapatrol(red_aew_id, 20000, 0, 8000, 0, 20000, 20000, 250, 7200, 0),
            EnvCmd.make_takeoff_areapatrol(RED_AIRPORT_ID, 1, UnitType.A2A, 100000, 80000 * random.choice([-1, 0, 1]), 8000, 0, 20000, 20000, 250, 7200)
        ])
        time.sleep(600 / self.sim_speed)
        self.obs = self.env_client.get_observation()
        self.red_fighter = next(entity for entity in self.obs["red"]["units"] if entity["LX"]==UnitType.A2A)
        self.blue_fighter = next(entity for entity in self.obs["red"]["qb"] if entity["LX"]==UnitType.A2A) 
        self.red_fighter_id = self.red_fighter["ID"]
        self.blue_fighter_id = self.blue_fighter["ID"]
        
        self.env_client.take_action([EnvCmd.make_airattack(self.blue_fighter_id, self.red_fighter_id, random.choice([0, 1]))])
        return np.array([self.red_fighter["X"] / 100000, self.red_fighter["Y"] / 100000, \
                            self.blue_fighter["X"] / 100000, self.blue_fighter["Y"] / 100000, \
                                self.calc_distance(self.red_fighter, self.blue_fighter) / 100000, int(len(self.obs["red"]["rockets"]) > 0)])

    def step(self, action):
        step_start_time = time.time()
        if action == 0:
            cmd = EnvCmd.make_airattack(self.red_fighter_id, self.blue_fighter_id, 0)
        elif action == 1:
            cmd = EnvCmd.make_areapatrol(self.red_fighter_id, 160000, self.red_fighter["Y"], 8000, 0, 20000, 20000, 250, 7200, 0)
        elif action == 2:
            cmd = EnvCmd.make_areapatrol(self.red_fighter_id, self.red_fighter["X"], 120000, 8000, 0, 20000, 20000, 250, 7200, 0)
        elif action == 3:
            cmd = EnvCmd.make_areapatrol(self.red_fighter_id, self.red_fighter["X"], -120000, 8000, 0, 20000, 20000, 250, 7200, 0)
        else:
            raise ValueError("illegal action!")
        self.env_client.take_action([cmd])
        time.sleep(max(0, step_start_time + self.decision_period / self.sim_speed - time.time()))

        self.update_state()
        if self.red_fighter and self.red_fighter["WP"]["170"] < 6:
            self.env_client.take_action([EnvCmd.make_returntobase(self.red_fighter_id, RED_AIRPORT_ID)])
            time.sleep(100 / self.sim_speed)
            self.update_state()

        next_state = self._make_state()
        reward, done = self._make_reward_done()
        return next_state, reward, done, {}

    def render(self, mode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

class CatsimMA:  

    '''multi agent version of catsim environment'''

    def __init__(self):
        pass