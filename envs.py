import gym
import cv2
import random 
import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

import nle
import re
import random
from PIL import Image, ImageDraw

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass
def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        #self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8) 
        # store dictionaries in observation buffer 
        self._obs_buffer = np.array([{},{},{},{}])
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            #if i == self._skip - 2:
            #    self._obs_buffer[0] = obs
            #if i == self._skip - 1:
            #    self._obs_buffer[1] = obs
            self._obs_buffer[i] = obs
            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer[random.randint(0,3)]
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=790,
            w=370,
            life_done=True,
            sticky_action=True,
            p=0.25):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, info = self.env.step(action)

            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                    info.get('episode', {}).get('visited_rooms', {})))
                self.history = self.reset()
            self.child_conn.send([self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(s)
        return self.history[:, :, :]

    def pre_proc(self, frame):
        if type(frame) == type(dict()):
            frame = frame
        else:
            frame = frame[0]
        img = nle.nethack.tty_render(frame['tty_chars'],frame['tty_colors'],frame['tty_cursor'])
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        img = ansi_escape.sub('', img).split('\n')[2:-2]
        frame = ''
        for l in img:
            frame = frame+l+'\n'
        img = Image.new(mode='RGB',size=(790,370))
        text = ImageDraw.Draw(img)
        text.text((0, 0),frame, fill=(255,255,255))
        return np.array(img.convert('L')).reshape(1,790,370)

    def get_init_state(self, s):
        for i in range(self.history_size):
            #print(self.history.shape)
            #print(self.pre_proc(s).shape)
            self.history[i,:,:] = self.pre_proc(s)
