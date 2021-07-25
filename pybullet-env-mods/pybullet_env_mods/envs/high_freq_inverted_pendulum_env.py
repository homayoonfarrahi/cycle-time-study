"""
The following classes extend the respective classes from pybullet_envs (https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs) and provide two modifications. Correcting the position of the central joint in the returned state and correcting the negative reward that is incurred when the elbow joint is at either of its limits.
"""

import numpy as np

from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv
from pybullet_envs.gym_pendulum_envs import InvertedDoublePendulumBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

class HighFreqInvertedPendulumBulletEnv(InvertedPendulumBulletEnv):
    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.004, frame_skip=1)

    def step(self, a):
        state, reward, done, info = super().step(a)
        reward *= self.scene.dt / 0.016
        self.HUD(state, a, False)
        return state, reward, done, info

class HighFreqInvertedDoublePendulumBulletEnv(InvertedDoublePendulumBulletEnv):
    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.004, frame_skip=1)

    def step(self, a):
        state, reward, done, info = super().step(a)
        reward *= self.scene.dt / 0.016
        self.HUD(state, a, False)
        return state, reward, done, info
