import numpy as np
import copy
from stable_baselines.common.running_mean_std import RunningMeanStd
from vecEnv import VecEnv

class bVecNormalize(VecEnv):
    def __init__(self, venv, ob=True, st=True, ret=True, clipob=10., clipst=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnv.__init__(self,
                        observation_space=venv.observation_space,
                        state_space=venv.state_space,
                        action_space=venv.action_space)
        print('bullet vec normalize initialization. ')
        self.venv = venv
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.st_rms = RunningMeanStd(shape=self.state_space.shape) if st else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.clipst = clipst
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action, z, skel):
        return self.step_norm(action, z, skel)

    def step_norm(self, action, z, skel):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, state, rews, done, infos = self.venv.step(action, z, skel)     # 각 robot에서 정의된 step()이 호출됨
        true_rews = copy.deepcopy(rews)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        state = self._stfilt(state)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return obs, state, rews, done, infos, true_rews

    def step_broadcast(self, action):
        res, obs, state, rews, done, infos = self.venv.step_broadcast(action)  # 각 robot에서 정의된 step()이 호출됨
        true_rews = copy.deepcopy(rews)
        for a in range(self.venv.num_agent):
            self.ret = self.ret * self.gamma + rews[a]
            obs[a] = self._obfilt(obs[a])
            state[a] = self._stfilt(state[a])

            if self.ret_rms:
                self.ret_rms.update(self.ret)
                rews[a] = np.clip(rews[a] / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return res, obs, state, rews, done, infos, true_rews

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs) if self.ret_rms else None
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def _stfilt(self, state):
        if self.st_rms:
            self.st_rms.update(state) if self.ret_rms else None
            state = np.clip((state - self.st_rms.mean) / np.sqrt(self.st_rms.var + self.epsilon), -self.clipst, self.clipst)
            return state
        else:
            return state

    def reset(self, z, skel):
        obs, state = self.venv.reset(z, skel)
        return self._obfilt(obs), self._stfilt(state)

    def reset_broadcast(self):
        obs, state = self.venv.reset_broadcast()
        for i in range(self.venv.num_agent):
            obs[i] = self._obfilt(obs[i])
            state[i] = self._stfilt(state[i])
        return obs, state

    def get_vrep_scene_path(self):
        return self.venv.get_vrep_scene_path()

    def initialize_robot(self, clientID):
        self.venv.initialize_robot(clientID)