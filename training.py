"""
Description: Main training code
Author: Tae-woo Kim
Contact: twkim0812@gmail.com
"""

from CommonObject import *
import glob
import argparse
import os
import time
import datetime
import copy
import torch

# GPU selection in case of multi-gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
facNum = 1
dev_num = (facNum-1) % 2
device = torch.device("cuda:" + str(dev_num))  # torch.cuda.set_device(facNum % 2 - 1)
print('device:: ', device, ' dev_num: ', dev_num)
torch.cuda.set_device(dev_num)


# imports for V-REP
from ExpCollector import ExpCollector

try:
    import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')

from model import MLPPolicy
from storage import RolloutStorage
import algo

from robots.NAO_MIMIC import NAO_MIMIC
from robots.BAXTER_MIMIC import BAXTER_MIMIC
from robots.C3PO_MIMIC import C3PO_MIMIC


#### settings for RL initial values ####
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:  # False is default
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

action_class_list = [22, 23, 31, 37, 38, 39]    # 23: hand waving, 38: salute
str_action_class = 'A' + 'A'.join('{0:03}'.format(action_class_list[i]) for i in range(len(action_class_list)))
str_action_class = 'UNI' if len(action_class_list) >= 6 else str_action_class
print(str_action_class)

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

args.env_name = myrobot.robot_name + '_' + str_action_class
args.num_processes = 1
args.num_stack = 1
args.algo = 'ppo'
args.use_gae = True  # Generalized Advantage Estimation
args.vis_interval = 100
args.log_interval = 10  # TODO
args.num_steps = 2048  # This should be greater than num_mini_batch
args.a_lr = 1e-4
args.c_lr = 2e-4
args.entropy_coef = 0.005
args.value_loss_coef = 1
args.cuda = True
args.ppo_epoch = 5
args.num_mini_batch = 32
args.gamma = 0.98
args.tau = .95
args.port = 6006  # tensorboard port [default: 6006]
args.vis = True
args.num_frames = 1000000
args.save_interval = 10

args.cont_learning = False
args.itr_num = 487

# Learning Mode variables
cyclic_policy = True
symm_policy = False
args.discount_mode = 'local'   # [single, local, global]
args.wnd_size = 3   # [1, 3, 5]
print('args: ', args)

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    print('cuda is vailable..1')
    torch.cuda.manual_seed(args.seed)


# print("#######")
# print("WARNING: We note that all rewards are clipped and normalized)
# print("#######")

# sym, asym, c3po
obs_shape = myrobot.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
state_shape = myrobot.state_space.shape
state_shape = (state_shape[0] * args.num_stack, *state_shape[1:])
action_shape = myrobot.action_space.shape
full_state_shape = (state_shape[0] * args.num_stack, obs_shape[1] + state_shape[1])
if symm_policy:
    full_state_shape = obs_shape
    state_shape = obs_shape

from gym.spaces import Box
myrobot.state_space = Box(low=float('-inf'), high=float('inf'), shape=state_shape, dtype=np.float32)
print('obs shape: ', obs_shape)
print('state shape: ', state_shape)
print('action shape: ', action_shape)
print('full state shape: ', full_state_shape)

portNum = 88880+facNum
env = ExpCollector(myrobot, portNum, 0, motion_sampling=False)

# load or new
if args.cont_learning == True:
    print('#### continue learning... ####')
    actor_critic = MLPPolicy(obs_shape[1], full_state_shape[1], env.robot.action_space, symm_policy=symm_policy, use_seq=False, cuda_use=args.cuda)
    exist_mode_name = '(1M_3L_512_ReLU)'
    exist_mode_name += '(cyclic)' if cyclic_policy else '(acyclic)'
    exist_mode_name += '(sym)' if symm_policy else '(asym)'
    print('Load Policy:::: ', os.path.join(args.load_dir + args.algo
                                           + '/', args.env_name + exist_mode_name
                                           + '(' + str(args.itr_num) + ')' + ".pt"))

    state_dict, ob_rms, st_rms, ret_rms = \
        torch.load(os.path.join(args.load_dir + args.algo
                                + '/', args.env_name + exist_mode_name
                                + '(' + str(args.itr_num) + ')' + ".pt"))
    actor_critic.load_state_dict(state_dict)

    env.robot.ob_rms = ob_rms
    env.robot.st_rms = st_rms
    env.robot.ret_rms = ret_rms
    args.env_name = exist_mode_name

else:
    print('$$$$ new learning... $$$$')
    actor_critic = MLPPolicy(obs_shape[1], full_state_shape[1], env.robot.action_space, symm_policy=symm_policy, use_seq=False, cuda_use=args.cuda)

    print('model desc: ', actor_critic)
    args.env_name += '(2M_3L_512_ReLU_inf)'
    args.env_name += '(cyclic)' if cyclic_policy else '(acyclic)'
    args.env_name += '(sym)' if symm_policy else '(asym)'
    if args.discount_mode == 'single':
        args.env_name += '(single)'
        args.gamma = 0.0
        args.use_gae = False
    elif args.discount_mode == 'local':
        args.env_name += '(local_' + str(args.wnd_size) + ')'
        args.use_gae = False
    elif args.discount_mode == 'global':
        args.env_name += '(global)'
        args.use_gae = True
    else:
        raise ValueError

if hasattr(actor_critic, 'actor_lstm'):
    args.env_name += '(LSTM)'

print(args.env_name)

####################################
# visualization
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter('runs/C-3PO_Task')
now = str(datetime.now().strftime('%Y-%m-%d@%H:%M:%S'))

if args.cuda:
    actor_critic.cuda()
    print('cuda is available..2')


agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, a_lr=args.a_lr, c_lr=args.a_lr,
                     eps=args.eps, max_grad_norm=args.max_grad_norm)

rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, full_state_shape,
                          env.robot.action_space)

current_obs = torch.zeros(args.num_processes, *obs_shape)
current_state = torch.zeros(args.num_processes, *full_state_shape)

######################
# Load skeleton data
from SkeletonDataManager import SkeletonDataManager

all_train_skeleton_data = np.empty((0, 76))
all_train_skeleton_fc = []
for i in action_class_list:
    np_train_skel_data, shape, fc_list = load_skeleton_data(skeleton_train_dict[i])
    all_train_skeleton_data = np.vstack((all_train_skeleton_data, np_train_skel_data))
    all_train_skeleton_fc += fc_list
    print('Finished the frame data Load of: ', i)

sdm = SkeletonDataManager(all_train_skeleton_data, all_train_skeleton_fc)
print('shape: ', all_train_skeleton_data.shape, 'skel: ', all_train_skeleton_data)
print('len: ', len(all_train_skeleton_fc), 'frame count: ', all_train_skeleton_fc)

# Load skeleton vae model
from vae_model import VAE, VAE_SK
load_path = os.path.join('./trained_models', 'vae')
load_name = 'Skel_vae_Superset'
state_dict = torch.load(os.path.join(load_path, load_name + '.pt'))

# Skeleton Model(input dim, output dim)
sModel = VAE_SK(75, 75, use_batch_norm=False, activation='ReLU')
sModel.load_state_dict(state_dict)
sModel.eval()

# Robot Model
# load_name = 'Motion_vae_' + str_action_class  # make_class_prefix([22, 23, 31, 37, 38, 39])
if myrobot.robot_name == 'BAXTER_MIMIC':
    load_name = '[Baxter]Motion_vae_Superset'
elif myrobot.robot_name == 'NAO_MIMIC':
    load_name = '[NAO]Motion_vae_Superset'
elif myrobot.robot_name == 'C3PO_MIMIC':
    load_name = '[C3PO]Motion_vae_Superset'
state_dict = torch.load(os.path.join(load_path, load_name + '.pt'))
rModel = VAE(14, latent_dim=np.prod(action_shape), use_batch_norm=False, activation='Tanh')   # action: 10*20
rModel.load_state_dict(state_dict)
rModel.eval()

# Set rModel to myrobot
if not symm_policy:
    myrobot.rModel = rModel


reward_list = []
if env.simStart(gui_on=True, autoStart=False, autoQuit=True, epiNum=args.num_steps) != -1:
    print('sim start!!----------')
    time.sleep(1)

    data, last_scene, first_scene = sdm.get_random_scene_frame()
    with torch.no_grad():
        mu, logvar = sModel.encode(torch.from_numpy(data[:-1]).unsqueeze(0).float())
        zs = sModel.reparameterize(mu, logvar)

    if cyclic_policy:
        with torch.no_grad():
            rSkel = sModel.decode(zs).reshape(75, -1).squeeze()
            rSkel = rSkel.numpy()
    else:
        rSkel = data

    # zs = zs.cpu().numpy()
    obs, st = env.reset(zs.cpu().numpy(), rSkel)
    update_current_obs(obs, current_obs)
    full_state = np.concatenate((current_obs.numpy().flatten(),  # initial obs
                                 st[0]))  # initial state
    if symm_policy:
        full_state = st
    update_current_state(full_state, current_state)
    rollouts.observations[0].copy_(current_obs)
    rollouts.states[0].copy_(current_state)

    if args.cuda:
        print('cuda is available..3')
        current_obs = current_obs.cuda()
        current_state = current_state.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        print('num_updates: ', j)

        episode_rewards = torch.zeros([args.num_processes, 1])
        final_rewards = torch.zeros([args.num_processes, 1])
        itr_time = time.time()
        for step in range(args.num_steps):  # PPO steps

            with torch.no_grad():
                value, action, action_log_prob, states, _, _ = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])

            # get next observation
            data, last_scene, first_scene = sdm.get_random_scene_frame()

            # get latent vector
            with torch.no_grad():
                mu, logvar = sModel.encode(torch.from_numpy(data[:-1]).unsqueeze(0).float())
                zs = sModel.reparameterize(mu, logvar)

            # cpu_actions = action.data.squeeze(1).cpu().numpy()
            # get robot motor value by decoding
            with torch.no_grad():
                x_hat = rModel.decode(action.cpu()).reshape(-1, 14)

            if cyclic_policy:
                with torch.no_grad():
                    rSkel = sModel.decode(zs).reshape(75, -1).squeeze()
                    rSkel = rSkel.numpy()
            else:
                rSkel = data

            next_obs, next_state, reward, done, info, true_rew = env.step(x_hat, zs.cpu().numpy(), rSkel)

            # To separate the Q-value of the current scene from others
            # if last_scene:
            #     done[0] = 1     # done is np array, [0]
            #     obs, next_state = env.reset(zs, rSkel)

            # local q-function
            if args.discount_mode == 'local' and j % args.wnd_size == 0:
                done[0] = 1  # done is np array, [0]

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            true_rew = torch.from_numpy(np.expand_dims(np.stack(true_rew), 1)).float()
            final_rewards += true_rew
            episode_rewards += reward  # scalar value

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            # final_rewards *= masks
            # final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim == 4:  # This means image
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
                current_state *= masks

            update_current_obs(next_obs, current_obs)
            # make full state
            full_state = np.concatenate((current_obs.cpu().numpy().flatten(),
                                         next_state.flatten()))  # current state!!
            if symm_policy:
                full_state = states.cpu().numpy()
            update_current_state(full_state, current_state)
            rollouts.insert(current_obs, current_state, action.data, action_log_prob.data, value.data, reward, masks)

        rollout_time = time.time() - itr_time
        itr_time = time.time()
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau, args.discount_mode)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()  # observation, state, mask의 0번째 값을 마지막에 얻은 값으로 대체

        print('itr rewards: ', final_rewards, ' rollout_time: ', rollout_time, ' update_time: ', time.time() - itr_time)
        reward_list.append(final_rewards.item())
        try:
            writer.add_scalar(args.env_name + '/' + now, reward_list[-1], len(reward_list) + 1)
            # reconst = reconst.squeeze(0)
            # b, g, r = torch.split(reconst, 1, dim=0)
            # reconst = torch.cat((r, g, b))
            # writer.add_image(args.env_name + '_IMG' + '/' + now, reconst, global_step=len(reward_list)+1)
            pass
        except NameError:
            pass

        # save learned network only for root node
        if (j % args.save_interval == 0 and args.save_dir != "") or j >= num_updates-1:
            print('save model!!')
            save_path = os.path.join(args.save_dir, args.algo, 'phase2', args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            save_state_dict = actor_critic.state_dict()
            # if args.cuda:
            #     save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_state_dict,
                          hasattr(env.robot, 'ob_rms') and env.robot.ob_rms or None,
                          hasattr(env.robot, 'st_rms') and env.robot.st_rms or None,
                          hasattr(env.robot, 'ret_rms') and env.robot.ret_rms or None]

            # save_model = [save_model, None]
            str_itr_num = '(' + str(args.itr_num + j) + ')' if args.cont_learning else '(' + str(j) + ')'
            torch.save(save_model, os.path.join(save_path, args.env_name + str_itr_num + ".pt"))

            # save list file
            with open(os.path.join(save_path, 'Rewards_Train_' + args.env_name + '.txt'), 'w') as f:
                for item in reward_list:
                    f.write("%s\n" % item)

    print('total elapsed time {:.2f}'.format(time.time()-start))
