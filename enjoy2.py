from CommonObject import *
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

# TODO, select robots!
from robots.NAO_MIMIC import NAO_MIMIC
from robots.BAXTER_MIMIC import BAXTER_MIMIC
from robots.C3PO_MIMIC import C3PO_MIMIC

# imports for V-REP
from ExpCollector import ExpCollector

####################################
# visualization
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter('runs/C-3PO_Task')
now = str(datetime.now().strftime('%Y-%m-%d@%H:%M:%S'))


"""
Parameter Settings
"""
action_class_list = [22, 23, 31, 37, 38, 39]
str_action_class = 'A' + 'A'.join('{0:03}'.format(action_class_list[i]) for i in range(len(action_class_list)))
str_action_class = 'UNI' if len(action_class_list) >= 6 else str_action_class
print(str_action_class)

ntu_socket_comm = True   # enjoy1.py
nao_socket_comm = False  # nao_action_parser.py
ntu_tcp_port = 4000
nao_tcp_port = 5001
args.port = 8097

"""
=================================================================
User Parameters for robot selection.
Pick a robot name in the "robotDict" and set it to "robot_name"
================================================================= 
"""
robotDict = {'NAO': NAO_MIMIC(), 'Baxter': BAXTER_MIMIC(), 'C3PO': C3PO_MIMIC()}
robot_name = 'NAO'

try:
    myrobot = robotDict[robot_name]
except KeyError:
    print('The robot is NOT supported..')
    exit()

args.num_stack = 1
args.env_name = myrobot.robot_name + '_' + str_action_class + '(1M_3L_512_ReLU_inf)'
args.algo = 'ppo'
args.vis = True
args.cuda = False

args.cyclic_policy = True
args.symm_policy = False
args.LbD_use = False
args.phase = 'phase3' if args.LbD_use else 'phase2'
# TODO --------------------------------

env = ExpCollector(myrobot, portNum=77777, comm=None, enjoy_mode=True, motion_sampling=False)

args.env_name += '(cyclic)' if args.cyclic_policy else '(acyclic)'
args.env_name += '(sym)' if args.symm_policy else '(asym)'
args.env_name += '(local_3)'    # [single, local_#, global]
args.env_name += '(LbD)' if args.LbD_use else ''
# args.env_name += '(roman)'
args.tr_itr = '(487)'   # (487), (512*6=3072) for single, 220, local 3-> 487 and local 5-> 110
print('Load File: ', args.env_name + args.tr_itr)


obs_shape = env.robot.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
state_shape = env.robot.state_space.shape
state_shape = (state_shape[0] * args.num_stack, *state_shape[1:])
action_shape = env.robot.action_space.shape
full_state_shape = (state_shape[0] * args.num_stack, obs_shape[1] + state_shape[1])

if args.symm_policy:
    full_state_shape = state_shape


"""
----[ Load Policy ]----
"""
actor_critic = MLPPolicy(obs_shape[1], full_state_shape[1], env.robot.action_space, symm_policy=args.symm_policy)

print(os.path.join(args.load_dir + args.algo, args.phase, args.env_name, args.env_name + args.tr_itr + ".pt"))
# state_dict, ob_rms = \
state_dict, ob_rms, st_rms, ret_rms = \
    torch.load(os.path.join(args.load_dir + args.algo, args.phase, args.env_name, args.env_name + args.tr_itr + ".pt"),
               map_location='cpu')

actor_critic.load_state_dict(state_dict)
actor_critic.train(False)
actor_critic.eval()  # TODO

print('ob_rms: ', ob_rms)
# print('av_ob_rms: ', av_ob_rms)
# print('ret_rms: ', ret_rms)
# print('av_ret_rms: ', av_ret_rms)

env.robot.ob_rms = ob_rms


epi_rewards = 0
######################
# Load skeleton data
# def load_train_data(file_path, _shape):
#     load_data = np.loadtxt(file_path).reshape(_shape)
#     print('load data shape: ', load_data.shape)
#     return load_data, load_data.shape
#
# skel_data, _ = load_train_data('./data/A038TrainSkeletonDataTr(32859_76).txt', (32859, 76))
# print('skel: ', skel_data, ' shape: ', skel_data.shape)

# Load skeleton vae model
from vae_model import VAE, VAE_SK
load_path = os.path.join('./trained_models', 'vae')
load_name = 'Skel_vae_Superset'
state_dict = torch.load(os.path.join(load_path, load_name + '.pt'))
sModel = VAE_SK(75, 75, use_batch_norm=False, activation='ReLU')
sModel.load_state_dict(state_dict)
sModel.eval()

# Load Motion vae Model
if myrobot.robot_name == 'BAXTER_MIMIC':
    load_name = '[Baxter]Motion_vae_Superset'
elif myrobot.robot_name == 'NAO_MIMIC':
    load_name = '[NAO]Motion_vae_Superset'
elif myrobot.robot_name == 'C3PO_MIMIC':
    load_name = '[C3PO]Motion_vae_Superset'

print('Motion Vae Name: ', load_name)
state_dict = torch.load(os.path.join(load_path, load_name + '.pt'))
rModel = VAE(14, latent_dim=np.prod(action_shape), use_batch_norm=False, activation='Tanh')   # action: 10*20
rModel.load_state_dict(state_dict)
rModel.eval()

if not args.symm_policy:
    myrobot.rModel = rModel

# -----[ For socket Communication, Client ] -----
if ntu_socket_comm:
    ntu_s = SocketCom('localhost', ntu_tcp_port)
    ntu_s.socket_connect()

num_eval_frames = 5000
reward_list = []

# Low Pass Filter
alpha = 0.9
prev_actions = torch.zeros([1, 14])


if nao_socket_comm:
    nao_s = SocketCom('localhost', nao_tcp_port)
    nao_s.socket_connect()

    # Connection Validation
    nao_s.write_socket({'header': 'Start NAO!', 'data': []})  # start signal to NAO

    if ntu_socket_comm:
        # recv -> bytes -> list -> numpy
        ntu_s.write_socket({'header': 'start NTU!', 'data': []})  # start signal
        # (2) get skel. data
        recv_data = ntu_s.read_socket('{', '}')
        data = recv_data['skeleton']
        skels = np.array(data[:-1])     # the last element is the class number

    with torch.no_grad():
        mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
        z = sModel.reparameterize(mu, logvar)
        z = z.cpu().numpy()
        # z = np.append(z.data.numpy(), recv_data[-1])  # appending phase to z

    obs = env.robot._obfilt(z)

    current_obs = torch.zeros(1, *obs_shape)
    current_state = torch.zeros(1, *full_state_shape)

    masks = torch.zeros(1, 1)
    update_current_obs(obs, current_obs)
    print('cu obs: ', current_obs, ' shape: ', current_obs.shape)

    for i in range(num_eval_frames):
        if i % 100 == 0:
            print('i: ', i)

        # data = sdm.get_random_scene_frame()
        if ntu_socket_comm:
            ntu_s.write_socket({'header': 'ContNTU', 'data': []})  # continue signal
            # (2) get skel. data
            recv_data = ntu_s.read_socket('{', '}')
            data = recv_data['skeleton']
            skels = np.array(data[:-1])  # the last element is the class number

            # cylic policy...
            if args.cyclic_policy:
                with torch.no_grad():
                    mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
                    z = sModel.reparameterize(mu, logvar)
                    skels = sModel.decode(z).reshape(75, -1).squeeze().numpy()
            else:
                skels = skels[:-1]  # the second last number is a phase value

        # get latent vector
        with torch.no_grad():
            # mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
            mu, logvar = sModel.encode(torch.from_numpy(skels[:]).unsqueeze(0).float())
            z = sModel.reparameterize(mu, logvar)
            z = z.cpu().numpy()

        with torch.no_grad():
            value, action, action_log_prob, states, _, _ = actor_critic.act(
                current_obs,
                current_state,
                masks,
                deterministic=True)

        # get trajectory of NAO by decoding
        with torch.no_grad():
            x_hat_rj = rModel.decode(action).reshape(-1, 14)

        # send motor values to Real NAO
        # torch -> numpy -> list -> bytes -> sendall
        filtered_action = alpha * x_hat_rj + (1 - alpha) * prev_actions  # TODO
        send_data = filtered_action.squeeze(0).cpu().numpy().tolist()
        nao_s.write_socket({'header': 'SetMotor', 'data': send_data})
        print('send: nao_angle (deg): ', np.rad2deg(send_data))

        # time.sleep(1/20)    # 20 Hz

        next_obs = env.robot._obfilt(z)
        update_current_obs(next_obs, current_obs)

        prev_actions = filtered_action

    exit()


"""
####################################################################################
"""


if env.simStart(gui_on=True, autoStart=True, autoQuit=True, epiNum=10000) != -1:
    print('V-REP evaluation start!!----------')

    if ntu_socket_comm:
        # recv -> bytes -> list -> numpy
        ntu_s.write_socket({'header': 'start NTU!', 'data': []})  # start signal
        # (2) get skel. data
        recv_data = ntu_s.read_socket('{', '}')
        data = recv_data['skeleton']
        skels = np.array(data[:-1])     # the last element is the class number

    with torch.no_grad():
        mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
        z = sModel.reparameterize(mu, logvar)
        z = z.cpu().numpy()
        # z = np.append(z.data.numpy(), recv_data[-1])  # appending phase to z

    print('z: ', z.shape, ' data: ', skels.shape)
    obs, state = env.reset(z, data)
    current_obs = torch.zeros(1, *obs_shape)
    current_state = torch.zeros(1, *full_state_shape)
    full_state = np.concatenate((np.random.normal(0.0, 0.1, obs_shape[1]),  # initial obs
                                 state[0]))  # initial state
                                 # np.random.normal(0.0, 0.1, action_shape[0])))  # initial action
    if args.symm_policy:
        full_state = state

    masks = torch.zeros(1, 1)
    update_current_obs(obs, current_obs)
    update_current_state(full_state, current_state)

    print('cu obs: ', current_obs, ' shape: ', current_obs.shape)

    for i in range(num_eval_frames):
        if i % 100 == 0:
            print('i: ', i)

        # data = sdm.get_random_scene_frame()
        if ntu_socket_comm:
            ntu_s.write_socket({'header': 'ContNTU', 'data': []})  # continue signal
            # (2) get skel. data
            recv_data = ntu_s.read_socket('{', '}')
            data = recv_data['skeleton']
            skels = np.array(data[:-1])  # the last element is the class number

            # cylic policy...
            if args.cyclic_policy:
                with torch.no_grad():
                    mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
                    z = sModel.reparameterize(mu, logvar)
                    skels = sModel.decode(z).reshape(75, -1).squeeze().numpy()
            else:
                skels = skels[:-1]

        # get latent vector
        with torch.no_grad():
            # mu, logvar = sModel.encode(torch.from_numpy(skels[:-1]).unsqueeze(0).float())
            mu, logvar = sModel.encode(torch.from_numpy(skels[:]).unsqueeze(0).float())
            z = sModel.reparameterize(mu, logvar)
            z = z.cpu().numpy()

        with torch.no_grad():
            value, action, action_log_prob, states, _, _ = actor_critic.act(
                current_obs,
                current_state,
                masks,
                deterministic=True)

        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # get trajectory of NAO by decoding
        with torch.no_grad():
            traj = rModel.decode(action).reshape(-1, 14)    # rad

        filtered_action = alpha*traj + (1-alpha)*prev_actions    # TODO
        obs, next_state, reward, done, info, true_rew = env.step(filtered_action, z, skels)  # 0.02 sec, 50Hz

        masks.fill_(0.0 if done else 1.0)
        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
            current_state *= masks
        update_current_obs(obs, current_obs)

        # make full state
        full_state = np.concatenate((current_obs.numpy().flatten(),
                                     next_state[0].flatten()))
                                     # action.numpy().flatten()))
        if args.symm_policy:
            full_state = state

        update_current_state(full_state, current_state)

        reward_list.append(true_rew[0])
        prev_actions = filtered_action

    # Print the statistics of rewards
    print('Model: ', args.env_name + args.tr_itr)
    print('Total Evaluation Frames: ', num_eval_frames)
    print('Total Rewards Sum: ', sum(reward_list))
    print('Average: ', np.mean(reward_list))
    print('Std: ', np.std(reward_list))
    print('Min: ', min(reward_list), ' Max: ', max(reward_list))

    # # save list file
    # with open('Rewards_Eval_' + args.env_name + '.txt', 'w') as f:
    #     for item in reward_list:
    #         f.write("%s\n" % item)


# socket close
if ntu_socket_comm:
    ntu_s.socket_close()
if nao_socket_comm:
    nao_s.socket_close()

print('Finish!')