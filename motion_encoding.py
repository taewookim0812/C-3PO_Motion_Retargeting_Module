
import time, os
import numpy as np
from robots.NAO_MIMIC import NAO_MIMIC
from robots.BAXTER_MIMIC import BAXTER_MIMIC
from robots.C3PO_MIMIC import C3PO_MIMIC

####### imports for V-REP #######
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
from ExpCollector import ExpCollector

# myrobot = NAO_MIMIC(task='RightSalute')
myrobot = C3PO_MIMIC()
myrobot = BAXTER_MIMIC()


from vae_model import *
from CommonObject import *

input_dim = 14  # [Head(2), LeftArm(6), RightArm(6)]
# if myrobot.robot_name == 'NAO_MIMIC':
#     latent_dim = 7
# elif myrobot.robot_name == 'BAXTER_MIMIC':
#     latent_dim = 12

latent_dim = np.prod(myrobot.action_space.shape)
model = VAE(input_dim, latent_dim=latent_dim, use_batch_norm=False, activation='Tanh')
# model = VAE_MT(input_dim, input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

action_class_list = [22, 23, 31, 37, 38, 39]

str_action_class = 'A' + 'A'.join('{0:03}'.format(action_class_list[i]) for i in range(len(action_class_list)))


def motion_data_augmentation(action_class, total_length=20000, noise_scale=0.00):
    print('action class: ', synthetic_motion_dict[action_class])

    # Load synthetic motion data
    np_load_data, shape = load_synthetic_motion_data(synthetic_motion_dict[action_class])
    print('Load data: ', np_load_data, ' Shape:', shape)

    # Data augmentation by adding noise
    # Head index: 1 ~ 2
    # Left Arm index: 3 ~ 8
    # Right Arm index: 21 ~ 26
    Head = np_load_data[:, 1:3]
    LeftArm = np_load_data[:, 3:9]
    RightArm = np_load_data[:, 21:]
    # print('Head: ', Head, ' shape: ', Head.shape)
    # print('LeftArm: ', LeftArm, ' shape: ', LeftArm.shape)
    # print('RightArm: ', RightArm, ' shape: ', RightArm.shape)

    augUpperBody = np.array([])
    index = 0
    # add noises to the every joint except for the finger
    for i in range(total_length):
        augHead = Head[index, :] + np.random.randn(Head[index, :].size) * noise_scale
        augLeftArm = LeftArm[index, :] + np.append(np.random.randn(LeftArm[index, :].size-1), 0.0) * noise_scale
        augRightArm = RightArm[index, :] + np.append(np.random.randn(RightArm[index, :].size-1), 0.0) * noise_scale

        # print('augHead: ', augHead, ' shape: ', augHead.shape)
        # print('augLeftArm: ', augLeftArm, ' shape: ', augLeftArm.shape)
        # print('augRightArm: ', augRightArm, ' shape: ', augRightArm.shape)

        # Make an augmented upper body array: [Head(2), LeftArm(6), RightArm(6)], Total 14 dim
        if augUpperBody.size:
            augUpperBody = np.vstack((augUpperBody, np.concatenate((augHead, augLeftArm, augRightArm), axis=0)))
        else:
            augUpperBody = np.concatenate((augHead, augLeftArm, augRightArm), axis=0)
        # print('augUpperBody: ', augUpperBody, ' shape: ', augUpperBody.shape)

        index += 1
        if index >= np_load_data.shape[0]:
            index = 0

    return augUpperBody



def motion_data_collect_vrep():
    print('NAO Motion data Collection')
    portNum = 50000
    num_samples = 20000
    traj_step = 1
    env = ExpCollector(myrobot, portNum, None, motion_sampling=True)

    gui_on = True
    prev_time = time.time()
    traj = np.array([])
    if env.simStart(gui_on=gui_on, autoStart=True, autoQuit=True, epiNum=10000) != -1:
        print('Data Collection Start!')
        for i in range(num_samples):
            new_traj = np.expand_dims(env.robot.venv.traj_sampling(traj_step), axis=0)
            traj = np.concatenate((traj, new_traj), axis=0) if traj.size else new_traj
            print('i: ', i, ' shape: ', traj.shape, ' time: ', time.time() - prev_time)
            prev_time = time.time()

    return traj, traj.shape


def save_motion_data(motion_data, str_action_class):
    dim_list = []
    [dim_list.append(str(motion_data.shape[i])) for i in range(len(motion_data.shape))]
    dim_str = "_".join(dim_list)
    # file_name = myrobot.task + '_traj_data(' + dim_str + ').txt'

    # save as text file
    save_name = str_action_class + 'TrainMotionData' + '(' + dim_str + ')'

    with open('./data/aug_motion/' + save_name + '.txt', 'w') as f:
        for slice_2d in motion_data:
            np.savetxt(f, slice_2d)

    print(save_name + ' is saved..')
    return save_name


def load_motion_data(file_name):
    shape = file_name[file_name.find('(') + 1:file_name.find(')')]
    shape = list(map(lambda i: int(i), shape.split('_')))

    full_path = os.path.join('./data/aug_motion', file_name + '.txt')
    load_data = np.loadtxt(full_path).reshape(shape)  # os.path.join('./data', file_name + '.txt')
    print('load data shape: ', load_data.shape)
    return load_data, load_data.shape


###########################################################################
############################# Encoding Functions ##########################
###########################################################################


def train(epoch, train_loader):
    model.train()
    model.cuda()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):   # Tensor, [batch_size, col_vec]
        data = data.to(device)  # 'cpu' or 'cuda'
        optimizer.zero_grad()

        # print('data shape: ', data[:, :, :].squeeze(1).shape)
        # exit()
        # recon_batch, mu, logvar = model(data[:, :, :].squeeze(1))   # forward
        recon_batch, mu, logvar = model(data)  # forward
        loss = loss_function(recon_batch, data, mu, logvar, input_dim)
        loss.backward()     # calc gradient
        train_loss += loss.item()   # .item(): to get a python number from tensors
        optimizer.step()            # actual parameter update
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == '__main__':
    """
    0: learning from seed motion data with augmentation
    1: learning from already created data in txt file
    2: learning from superset data
    """
    mode = 2

    save_name = 'Motion_vae_' + make_class_prefix(action_class_list)
    if mode == 0:
        motion_data = motion_data_augmentation(action_class, total_length=20000, noise_scale=0.05)
        # motion_data, _ = motion_data_collect_vrep()
        save_name = save_motion_data(motion_data, str_action_class)
    elif mode == 1:   # individual VAE learning
        all_train_motor_data = np.array([])
        # Load already generated NAO motion data
        for i in action_class_list:
            np_motion_data, _ = load_motion_data(motion_train_dict[i])
            if all_train_motor_data.size == 0:
                all_train_motor_data = np_motion_data
            else:
                all_train_motor_data = np.vstack((all_train_motor_data, np_motion_data))
            print('Finished the frame data Load of: ', i)
    elif mode == 2:   # superset learning
        # Set Robot Name: [NAO, Baxter, C3PO]
        robot_name = 'C3PO'
        # load superset data
        superset_file_path = os.path.join('trained_models', 'ppo', 'phase1')
        superset_file_name = '[AUG]' + robot_name + '_motion_superset.txt'

        all_train_motor_data = np.loadtxt(os.path.join(superset_file_path, superset_file_name))
        save_name = '[' + robot_name + ']Motion_vae_Superset'

    print(all_train_motor_data.shape, all_train_motor_data)
    np.random.shuffle(all_train_motor_data)
    print(all_train_motor_data.shape, all_train_motor_data)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(all_train_motor_data).float(),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    args.epochs = 10
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader)

    # save the trained model
    save_path = os.path.join('./trained_models', 'vae')
    print(save_name + '.pt')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save(model.state_dict(), os.path.join(save_path, save_name + '.pt'))
    print('finish!')


