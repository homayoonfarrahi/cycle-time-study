
import torch
import numpy as np
import time


class Agent(object):
    def __init__(self, pol, learner, device='cpu'):
        self.pol = pol
        self.learner = learner
        self.device = device

    def get_action(self, o):
        """
        :param o: np. array of shape (1,)
        :return: a two tuple
        - np.array of shape (1,)
        - np.array of shape (1,)
        """
        action, lprob, dist = self.pol.action(torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0))
        return action[0].cpu().numpy(), lprob.cpu().numpy(), self.pol.dist_to(dist, to_device='cpu')

    def log_update(self, o, a, r, op, logp, dist, done):
        return self.learner.log_update(o, a, r, op, logp, dist, done)

class UR5ReacherScriptedAgent(Agent):
    def __init__(self):
        self.reset()

    def reset(self):
        self.i = 0
        self.prev_error = None

    def get_action(self, o):
        q = o[:2]
        qd = o[2:4]
        target = o[-2:]
        error = target - q

        if self.prev_error is None:
            self.prev_error = error


        # s_1.0_a_2.0
        Kp = np.array([3, 3])
        Ki = np.array([0, 0])
        Kd = np.array([-0.1, -0.1])

        # s_0.3_a_1.4
        # Kp = np.array([10, 10])
        # Ki = np.array([0, 0])
        # Kd = np.array([-0.1, -0.1])

        p =  error * Kp
        self.i += error * Ki
        d = qd * Kd

        action = p + self.i + d
        return action, None, None

class UR5ReacherEStopScripter(Agent):
    def __init__(self):
        self.reset()

    def reset(self):
        self.i = 0

    def get_action(self, o):
        action = np.array([0, 0])
        speed = 100
        start = 0
        period = 20
        if start <= self.i < start + period:
            action = np.array([speed, 0])
        elif self.i < start + (2 * period):
            # action = np.array([-speed, 0])
            pass


        self.i += 1

        return action, None, None

        # q = o[:2]
        # qd = o[2:4]
        # target = o[-2:]
        # error = target - q

        # if self.prev_error is None:
        #     self.prev_error = error


        # Kp = np.array([3, 3])
        # Ki = np.array([0, 0])
        # Kd = np.array([-0.25, -0.25])

        # p =  error * Kp
        # self.i += error * Ki
        # d = qd * Kd

        # action = p + self.i + d
        # return action, None, None
