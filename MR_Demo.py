"""
Description: A class for Motion Retargeting Demo
Author: Tae-woo Kim
Contact: twkim0812@gmail.com
"""

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import argparse
import os

import numpy as np
import torch
from model import MLPPolicy
from vae_model import VAE, VAE_SK

from robots.NAO_MIMIC import NAO_MIMIC
from robots.BAXTER_MIMIC import BAXTER_MIMIC
from robots.C3PO_MIMIC import C3PO_MIMIC

# imports for V-REP
from ExpCollector import ExpCollector
from arguments import get_args

robot_dict = {'NAO': NAO_MIMIC(), 'BAXTER': BAXTER_MIMIC(), 'C3PO': C3PO_MIMIC()}
env__list = ['CHREO', 'VREP']


class MR_Demo:
    def __init__(self, target_robot, target_env, tcp_port):
        if target_env not in env__list:
            raise Exception('Unknown env. Please choose the env among ', env__list)

        if target_robot not in robot_dict.keys():
            raise Exception('Unknown robot. Please choose the robot among ', robot_dict.keys())

        self.target_env = target_env
        self.target_robot = target_robot
        self.myrobot = robot_dict[target_robot]

        self.nao_tcp_port = tcp_port

        self.env = ExpCollector(self.myrobot, portNum=23876, comm=None, enjoy_mode=True, motion_sampling=False)
        self.args = self.init_args()

        self.actor_critic = self.load_policy()
        self.sModel = self.load_skeleton_VAE()
        self.rModel = self.load_robot_VAE()

        self.current_obs = None
        self.current_state = None
        self.masks = None

        if self.target_env == 'VREP':
            self.vrep = self.init_vrep()

    def init_args(self):
        args = get_args()
        args.algo = 'ppo'
        args.num_stack = 1
        args.env_name = self.myrobot.robot_name + '_UNI' '(1M_3L_512_ReLU_inf)'
        args.vis = True
        args.cuda = False

        args.cyclic_policy = True
        args.symm_policy = False
        args.LbD_use = False
        args.phase = 'phase3' if args.LbD_use else 'phase2'

        args.env_name += '(cyclic)' if args.cyclic_policy else '(acyclic)'
        args.env_name += '(sym)' if args.symm_policy else '(asym)'
        args.env_name += '(local_3)'  # choose among: [single, local_#, global]
        args.env_name += '(LbD)' if args.LbD_use else ''
        args.tr_itr = '(487)'

        # shape initialization
        obs_shape = self.env.robot.observation_space.shape
        args.obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
        state_shape = self.env.robot.state_space.shape
        args.state_shape = (state_shape[0] * args.num_stack, *state_shape[1:])
        args.action_shape = self.env.robot.action_space.shape
        args.full_state_shape = (state_shape[0] * args.num_stack, obs_shape[1] + state_shape[1])

        if args.symm_policy:
            args.full_state_shape = state_shape
        return args

    def update_current_state(self, state, current_state):
        shape_dim0 = current_state.shape[-1]
        state = torch.from_numpy(state).float()
        current_state[:, -shape_dim0:] = state

    def update_current_obs(self, _obs, current_obs):
        shape_dim0 = current_obs.shape[-1]
        obs = torch.from_numpy(_obs).float()
        if self.args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
        return current_obs

    def load_skeleton_VAE(self):
        load_path = os.path.join('./trained_models', 'vae')
        load_name = 'Skel_vae_Superset'
        state_dict = torch.load(os.path.join(load_path, load_name + '.pt'), map_location='cpu')
        sModel = VAE_SK(75, 75, use_batch_norm=False, activation='ReLU')
        sModel.load_state_dict(state_dict)
        sModel.eval()

        return sModel

    def load_robot_VAE(self):
        if self.target_robot == 'BAXTER':
            load_name = '[Baxter]Motion_vae_Superset'
        elif self.target_robot == 'NAO':
            load_name = '[NAO]Motion_vae_Superset'
        elif self.target_robot == 'C3PO':
            load_name = '[C3PO]Motion_vae_Superset'
        else:
            print('Unknown robot..')
            raise ValueError

        load_path = os.path.join('./trained_models', 'vae')
        state_dict = torch.load(os.path.join(load_path, load_name + '.pt'), map_location='cpu')
        rModel = VAE(14, latent_dim=np.prod(self.args.action_shape), use_batch_norm=False, activation='Tanh')
        rModel.load_state_dict(state_dict)
        rModel.eval()
        print('Motion VAE Load Success!')

        if not self.args.symm_policy:
            self.myrobot.rModel = rModel
        return rModel

    def load_policy(self):
        actor_critic = MLPPolicy(self.args.obs_shape[1], self.args.full_state_shape[1], self.env.robot.action_space,
                                 symm_policy=self.args.symm_policy)
        print(os.path.join(self.args.load_dir + self.args.algo, self.args.phase, self.args.env_name, self.args.env_name
                           + self.args.tr_itr + ".pt"))
        state_dict, ob_rms, st_rms, ret_rms = \
            torch.load(
                os.path.join(self.args.load_dir + self.args.algo, self.args.phase, self.args.env_name,
                             self.args.env_name
                             + self.args.tr_itr + ".pt"),
                map_location='cpu')

        actor_critic.load_state_dict(state_dict)
        actor_critic.train(False)
        actor_critic.eval()
        self.env.robot.ob_rms = ob_rms
        return actor_critic

    def init_vrep(self):
        return self.env.simStart(gui_on=True, autoStart=True, autoQuit=True, epiNum=1000000)

    def do_retargeting(self, skeleton_frame):
        skels = np.array(skeleton_frame)  # the last element is the class number

        # Initialized only once
        if self.current_obs is None:
            with torch.no_grad():
                mu, logvar = self.sModel.encode(torch.from_numpy(skels).unsqueeze(0).float())
                z = self.sModel.reparameterize(mu, logvar)
                z = z.cpu().numpy()

            obs = self.env.robot._obfilt(z)

            self.current_obs = torch.zeros(1, *self.args.obs_shape)
            self.current_state = torch.zeros(1, *self.args.full_state_shape)

            self.masks = torch.zeros(1, 1)
            self.update_current_obs(obs, self.current_obs)

        if self.args.cyclic_policy:
            with torch.no_grad():
                mu, logvar = self.sModel.encode(torch.from_numpy(skels).unsqueeze(0).float())
                z = self.sModel.reparameterize(mu, logvar)
                skels = self.sModel.decode(z).reshape(75, -1).squeeze().numpy()
        else:
            skels = skels[:-1]  # the last number is a phase value

        # get latent vector
        with torch.no_grad():
            mu, logvar = self.sModel.encode(torch.from_numpy(skels[:]).unsqueeze(0).float())
            z = self.sModel.reparameterize(mu, logvar)
            z = z.cpu().numpy()

        with torch.no_grad():
            value, action, action_log_prob, states, _, _ = self.actor_critic.act(
                self.current_obs,
                self.current_state,
                self.masks,
                deterministic=True)

        # get trajectory of NAO by decoding
        with torch.no_grad():
            x_hat_rj = self.rModel.decode(action).reshape(-1, 14)

        next_obs = self.env.robot._obfilt(z)
        self.update_current_obs(next_obs, self.current_obs)

        return x_hat_rj.squeeze(0).cpu().numpy().tolist()

    def do_retargeting_vrep(self, skeleton_frame):
        if self.vrep == -1: return False
        skels = np.array(skeleton_frame)  # the last element is the class number

        # Initialized only once
        if self.current_obs is None:
            with torch.no_grad():
                mu, logvar = self.sModel.encode(torch.from_numpy(skels).unsqueeze(0).float())
                z = self.sModel.reparameterize(mu, logvar)
                z = z.cpu().numpy()

            obs, state = self.env.reset(z, None)

            self.current_obs = torch.zeros(1, *self.args.obs_shape)
            self.current_state = torch.zeros(1, *self.args.full_state_shape)
            self.masks = torch.zeros(1, 1)

            self.update_current_obs(obs, self.current_obs)

            full_state = np.concatenate((np.random.normal(0.0, 0.1, self.args.obs_shape[1]),  # initial obs
                                         state[0]))  # initial state
            if self.args.symm_policy:
                full_state = state
            self.update_current_state(full_state, self.current_state)

        # Main loop
        if self.args.cyclic_policy:
            with torch.no_grad():
                mu, logvar = self.sModel.encode(torch.from_numpy(skels).unsqueeze(0).float())
                z = self.sModel.reparameterize(mu, logvar)
                skels = self.sModel.decode(z).reshape(75, -1).squeeze().numpy()
        else:
            skels = skels[:-1]  # the last number is a phase value

        # get a latent vector
        with torch.no_grad():
            mu, logvar = self.sModel.encode(torch.from_numpy(skels[:]).unsqueeze(0).float())
            z = self.sModel.reparameterize(mu, logvar)
            z = z.cpu().numpy()

        with torch.no_grad():
            value, action, action_log_prob, states, _, _ = self.actor_critic.act(
                self.current_obs,
                self.current_state,
                self.masks,
                deterministic=True)

        # get trajectory of NAO by decoding
        with torch.no_grad():
            x_hat_rj = self.rModel.decode(action).reshape(-1, 14)
        obs, next_state, reward, done, info, true_rew = self.env.step(x_hat_rj, z, skels)  # 0.02 sec, 50Hz

        self.masks.fill_(0.0 if done else 1.0)
        if self.current_obs.dim() == 4:
            self.current_obs *= self.masks.unsqueeze(2).unsqueeze(2)
        else:
            self.current_obs *= self.masks
            self.current_state *= self.masks
        self.update_current_obs(obs, self.current_obs)

        # make full state
        full_state = np.concatenate((self.current_obs.numpy().flatten(),
                                     next_state[0].flatten()))
        if self.args.symm_policy:
            full_state = state

        self.update_current_state(full_state, self.current_state)

        return True
