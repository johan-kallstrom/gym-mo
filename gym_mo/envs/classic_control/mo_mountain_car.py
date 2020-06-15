import math

import numpy as np

from gym import spaces
from gym.envs.classic_control import MountainCarEnv

class MoMountainCarEnv(MountainCarEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_reward = -1.0
        self.max_reward = 1.0

        self.reward_low = np.array([self.min_reward, self.min_reward, self.min_reward])
        self.reward_high = np.array([self.max_reward, self.max_speed, self.max_reward])
        self.reward_space = spaces.Box(self.reward_low, self.reward_high, dtype=np.float32)

        super().__init__()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = np.zeros(self.reward_space.shape, dtype=np.float32)
        reward[0] =  0.0 if done else -1.0          # Time penalty
        reward[1] = -1.0 if (action == 0) else 0.0  # Braking penalty
        reward[2] = -1.0 if (action == 2) else 0.0  # Acceleration penalty

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}
