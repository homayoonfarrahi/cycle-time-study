"""
The following classes extend the respective classes from pybullet_envs (https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs) and provide two modifications. Correcting the position of the central joint in the returned state and correcting the negative reward that is incurred when the elbow joint is at either of its limits.
"""

import numpy as np

from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv
from pybullet_envs.robot_manipulators import Reacher
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

class SelfAwareReacher(Reacher):
    def calc_state(self):
        """ theta is calculated using current_relative_position() in the original code.
            This is problematic since the central_joint is unlimited and its min and
            max are 1 and 0 respectively. The below code changes the returned state
            to use the correct theta.
        """
        theta, self.theta_dot = self.central_joint.current_position()
        self.theta_dot *= 0.1   # to be consistent w/ how velocity was calculated before this change
        state = super().calc_state()
        state[4:7] = np.array([np.cos(theta), np.sin(theta), self.theta_dot])
        return state

class SelfAwareReacherBulletEnv(ReacherBulletEnv):
    def __init__(self, render=False):
        # robot is set to the reacher with the correct state defined above
        self.robot = SelfAwareReacher()

        ####################
        # Original Code
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
        ####################

    def step(self, a):
        ####################
        # Original Code
        potential_old = self.potential
        ####################

        # step the environment, the reward is changed later
        state, reward, done, info = super().step(a)  # updates self.potential

        ####################
        # Original Code
        electricity_cost = (
            -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
                    )  # work torque*angular_velocity
            - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )
        ####################

        """ In the original code, stuck_joint_cost would be set to -0.1 if gamma (elbow_joint's
            position) was in the range (-1.01, -0.99) or (0.99, 1.01). However, the physics engine
            allows the joint to go slightly above 1.01 or below -1.01 even though the joint is
            limited to (-1, 1). The stuck_joint_cost below is modified to account for the joint
            going beyond these limits.
        """
        stuck_joint_cost = -0.1 if np.abs(self.robot.gamma) > 0.99 else 0.0

        ####################
        # Original Code
        self.rewards = [
            float(self.potential - potential_old),
            float(electricity_cost),
            float(stuck_joint_cost)
        ]
        self.HUD(state, a, False)
        return state, sum(self.rewards), done, info
        ####################

class HighFreqReacher(SelfAwareReacher):
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

class HighFreqReacherBulletEnv(SelfAwareReacherBulletEnv):
    def __init__(self, render=False):
        # robot is set to the reacher with the correct state defined above
        self.robot = HighFreqReacher()

        ####################
        # Original Code
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
        ####################

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.002, frame_skip=1)

    def step(self, a):
        ####################
        # Original Code
        potential_old = self.potential
        ####################

        # step the environment, the reward is changed later
        state, reward, done, info = super().step(a)  # updates self.potential
        reward_scale = self.scene.dt / 0.016
        electricity_cost = reward_scale * (
            -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
                    )  # work torque*angular_velocity
            - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )

        """ In the original code, stuck_joint_cost would be set to -0.1 if gamma (elbow_joint's
            position) was in the range (-1.01, -0.99) or (0.99, 1.01). However, the physics engine
            allows the joint to go slightly above 1.01 or below -1.01 even though the joint is
            limited to (-1, 1). The stuck_joint_cost below is modified to account for the joint
            going beyond these limits.
        """
        stuck_joint_cost = reward_scale * (-0.1 if np.abs(self.robot.gamma) > 0.99 else 0.0)

        ####################
        # Original Code
        self.rewards = [
            float(self.potential - potential_old),
            float(electricity_cost),
            float(stuck_joint_cost)
        ]
        self.HUD(state, a, False)
        info['reward_components'] = self.rewards
        return state, sum(self.rewards), done, info
        ####################
