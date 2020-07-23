from CommonObject import *
import numpy as np
from Skeleton import Skeleton, Category, Body_struct, Joint_struct
import os, glob, random

# [22, 23, 31, 37, 38, 39]
action_class_list = [22, 23, 31, 37, 38, 39]

def load_and_save_raw_data():
    train_skeleton_file_dict = {}

    for i in action_class_list:
        A = 'A' + '{0:03}'.format(i)  # Action class label
        my_path = '/dd/'  # [Important!!] change the '/dd/' to your own path
        category = Category()
        file_path = os.path.join(my_path, 'NTU_DB', category.daily_action, A)

        # Read skeleton exception file list
        skeleton_file_list = sorted(glob.glob(file_path + "/*.skeleton"))
        # exclude exception files
        with open(A + '_exeception_skeleton_list.txt', 'r') as f_skel:
            exception_skel_list = f_skel.readlines()
        for j in range(len(exception_skel_list)):
            exc_file = exception_skel_list[j].replace('\n', '')
            if exc_file in skeleton_file_list: skeleton_file_list.remove(exc_file)
        # train_skeleton_file_list += skeleton_file_list
        train_skeleton_file_dict[i] = skeleton_file_list

    print('skeleton file list: ', train_skeleton_file_dict[22])

    for i in action_class_list:
        # Data load from NTU DB and save
        # Training Data Set
        train_data, total_frame, fc_list = skeleton_motion_data_collect(train_skeleton_file_dict[i])
        np_train_data = skeleton_coordinate_transform(train_data, total_frame, coord_transform=False)
        # print('fc_list: ', fc_list, 'np_train_data: ', np_train_data, ' shape: ', np_train_data.shape)
        save_name = save_skeleton_data(np_train_data, [i], fc_list, data_type='Raw')


# prepare the training and test data separately from the skeleton files
# fetch the skeleton data from skeleton file, except for the exception files
def prepare_training_test_data():
    train_skeleton_file_dict = {}
    test_skeleton_file_dict = {}
    train_skeleton_file_list = []
    test_skeleton_file_list = []
    for i in action_class_list:
        A = 'A' + '{0:03}'.format(i)  # Action class label
        my_path = '/dd/'  # [Important!!] change the '/dd/' to your own path
        category = Category()
        file_path = os.path.join(my_path, 'NTU_DB', category.daily_action, A)

        # Read skeleton exception file list
        skeleton_file_list = sorted(glob.glob(file_path + "/*.skeleton"))
        # exclude exception files
        with open(A + '_exeception_skeleton_list.txt', 'r') as f_skel:
            exception_skel_list = f_skel.readlines()
        for j in range(len(exception_skel_list)):
            exc_file = exception_skel_list[j].replace('\n', '')
            if exc_file in skeleton_file_list: skeleton_file_list.remove(exc_file)
        # train_skeleton_file_list += skeleton_file_list
        train_skeleton_file_dict[i] = skeleton_file_list

    # Divide the skeleton list into Train and Test list.
    for i in action_class_list:
        nTest = round(len(train_skeleton_file_dict[i]) * 0.1)  # number of test files [10% of total files]
        random.shuffle(train_skeleton_file_dict[i])
        for j in range(nTest):
            test_skeleton_file_list.append(train_skeleton_file_dict[i].pop())
        test_skeleton_file_dict[i] = test_skeleton_file_list
        test_skeleton_file_list = []

    for i in action_class_list:
        print('Class Num: ', i)
        print('Train Length: ', len(train_skeleton_file_dict[i]), train_skeleton_file_dict[i])
        print('Test Length: ', len(test_skeleton_file_dict[i]), test_skeleton_file_dict[i])

        # Save the Exception File List
        save_path = './data/skeleton'
        data_type = 'Train'
        save_name = make_class_prefix([i]) + data_type + 'FileList' + '(' + str(len(train_skeleton_file_dict[i])) + ')'
        full_path = os.path.join(save_path, save_name + '.txt')
        print('Train full path: ', full_path)
        with open(full_path, 'w') as f:
            for item in train_skeleton_file_dict[i]:
                f.write("%s\n" % item)

        data_type = 'Test'
        save_name = make_class_prefix([i]) + data_type + 'FileList' + '(' + str(len(test_skeleton_file_dict[i])) + ')'
        full_path = os.path.join(save_path, save_name + '.txt')
        print('Test full path: ', full_path)
        with open(full_path, 'w') as f:
            for item in test_skeleton_file_dict[i]:
                f.write("%s\n" % item)

        print('----- finish -----')

    return train_skeleton_file_dict, test_skeleton_file_dict


# Load every frame of Skeleton Data from a file list
# And make it a list
def skeleton_motion_data_collect(skeleton_file_list):
    train_data = []
    total_frame = 0
    fc_list = []    # frame count list
    index = 0
    while index < len(skeleton_file_list):
        # -----------------------
        # -----[ Data Load ]-----
        # -----------------------
        print('list number: ', index)
        print('skeleton: ', skeleton_file_list[index])

        file = open(skeleton_file_list[index])

        frameCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]
        skel = Skeleton()
        for f in range(frameCount):  # frameCount
            bodyCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]

            body = Body_struct()
            for b in range(bodyCount):
                bodyInfo = np.fromstring(file.readline(), sep=' ', dtype='float64')
                jointCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]
                body.set_body_info(bodyInfo, jointCount, f*1.0/frameCount)

                for j in range(jointCount):
                    jointInfo = np.fromstring(file.readline(), sep=' ', dtype='float64')
                    joint = Joint_struct(jointInfo)
                    body.joints.append(joint)
                skel.append_body(f, b, body)
            # print('phase: ', body.phase)
        train_data.append(skel)
        file.close()
        # print(skel.print_skeleton_info())
        index += 1
        total_frame += len(skel.iBody)
        fc_list.append(len(skel.iBody))
    print('total number of frames: ', total_frame)
    print('train data len: ', len(train_data))

    return train_data, total_frame, fc_list


# Coordinate transform from original axis to the NAO's
def skeleton_coordinate_transform(train_data, total_frame, coord_transform=True):
    step = 3
    cnt = 0
    np_train_data = np.zeros([total_frame, step * 25 + 1])
    for i in range(len(train_data)):  # num train data, each frame element
        print('i: ', i)
        for f in range(len(train_data[i].iBody)):  # num frames
            # body numbers should be checked..
            for b in range(len(train_data[i].iBody[f])):  # num bodies
                # 2 is the torso joint, which is the new origin
                if not train_data[i].iBody[f][b]:
                    continue
                jTorso3D = np.array([train_data[i].iBody[f][b].joints[1].x,
                                     train_data[i].iBody[f][b].joints[1].y,
                                     train_data[i].iBody[f][b].joints[1].z])

                # define the first vector v
                u = np.array([train_data[i].iBody[f][b].joints[0].x,
                               train_data[i].iBody[f][b].joints[0].y,
                               train_data[i].iBody[f][b].joints[0].z])
                u -= jTorso3D
                u = u / np.linalg.norm(u)

                # define v vector by two shoulder joints(from right to left).
                lv = np.array([train_data[i].iBody[f][b].joints[4].x,
                              train_data[i].iBody[f][b].joints[4].y,
                              train_data[i].iBody[f][b].joints[4].z])
                rv = np.array([train_data[i].iBody[f][b].joints[8].x,
                               train_data[i].iBody[f][b].joints[8].y,
                               train_data[i].iBody[f][b].joints[8].z])
                v = lv-rv
                # v = np.ones([3])
                # v[2] = (-u[0]*v[0] - u[1]*v[1]) / u[2]
                v = v / np.linalg.norm(v)    # normalize

                # define u', forward vector of torso
                u2 = np.cross(u, v)

                w = np.cross(u2, v)
                w = w / np.linalg.norm(w)

                # make DCM(direction cosine matrix), raw -> skel coordinate
                m = np.stack((u2, v, w), axis=1)     # column-wise stack
                DCM = np.eye(3, 3)@m

                # Rotation Matrix, skel -> NAO coordinate
                #!!! the following rotation matrix is not used anymore. !!!
                # alpha = np.deg2rad(-90)
                # beta = 0
                # gamma = np.deg2rad(90)
                # Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
                # Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
                # Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
                # rot = Rx@Ry@Rz

                # print('jTorso3D: ', jTorso3D, ' jTorso2D: ', jTorso2D)
                for j in range(len(train_data[i].iBody[f][b].joints)):  # num joints(25)
                    # print('joint value: ', train_data[i].iBody[f][b][0].joints[j].x)
                    j3D = np.array([train_data[i].iBody[f][b].joints[j].x,
                                    train_data[i].iBody[f][b].joints[j].y,
                                    train_data[i].iBody[f][b].joints[j].z])

                    if coord_transform:
                        # origin translation & rotate inversely using DCM
                        newj = j3D - jTorso3D
                        newp = np.transpose(DCM)@newj  # before: rot@(np.transpose(DCM)@newj)
                    else:
                        newj = j3D
                        newp = newj  # before: rot@(np.transpose(DCM)@newj)

                    # 3D
                    # print('cnt: ', cnt, ' j: ', j)
                    np_train_data[cnt][j * step + 0] = newp[0]
                    np_train_data[cnt][j * step + 1] = newp[1]
                    np_train_data[cnt][j * step + 2] = newp[2]

                    # phase is added to the last element
                    np_train_data[cnt][-1] = train_data[i].iBody[f][b].phase

                # define Ryy matrix for refinement of the yaw
                # !!! the following rotation matrix is not used anymore. !!!
                # rs = np.array([np_train_data[cnt][4 * step + 0],
                #                np_train_data[cnt][4 * step + 1],
                #                np_train_data[cnt][4 * step + 2]])
                # ls = np.array([np_train_data[cnt][8 * step + 0],
                #                np_train_data[cnt][8 * step + 1],
                #                np_train_data[cnt][8 * step + 2]])
                # yaw = np.arccos(np.dot((rs - ls), [0, 1, 0]) / (np.linalg.norm(rs - ls) * np.linalg.norm([0, 1, 0])))
                # yaw *= -1.0 if (rs-ls)[0] < 0.0 else 1.0
                # # print('f: ', f, ' cnt: ', cnt, 'yaw: ', yaw * (180 / np.pi), ' x:', rs - ls)
                # Ryz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

                for k in range(25):  # num joints(25)
                    p = np.array([np_train_data[cnt][k * step + 0],
                                  np_train_data[cnt][k * step + 1],
                                  np_train_data[cnt][k * step + 2]])

                    # newp = Ryz@p
                    newp = p
                    np_train_data[cnt][k * step + 0] = newp[0]
                    np_train_data[cnt][k * step + 1] = newp[1]
                    np_train_data[cnt][k * step + 2] = newp[2]

                cnt += 1
    return np_train_data


def save_skeleton_data(np_data, class_list, fc_list, data_type='Raw'):
    shape_str = ''
    for i in range(len(np_data.shape)):
        sep = '_' if i < len(np_data.shape) - 1 else ''
        shape_str += str(np_data.shape[i]) + sep

    fc_list = np.array(fc_list).reshape(1, len(fc_list))

    # save as text file
    save_path = './data/skeleton'
    type_list = ['Raw', 'Train', 'Test']
    flag = 0
    for type in type_list:
        flag += 1 if type == data_type else 0

    if flag <= 0:
        raise ValueError

    action_class_name = make_class_prefix(class_list)
    save_name = action_class_name + data_type + 'SkeletonFrame' + '(' + shape_str + ')'
    np.savetxt(os.path.join(save_path, save_name + '.txt'), fc_list.astype(int),
               fmt='%i', delimiter=' ')   # save file count list first!

    with open(os.path.join(save_path, save_name + '.txt'), 'a') as f:
        np.savetxt(f, np_data)
    f.close()
    return save_name


###########################################################################
############################# Encoding Functions ##########################
###########################################################################

from vae_model import *

input_dim = 75
output_dim = 75
model = VAE_SK(input_dim, output_dim, use_batch_norm=False, activation='ReLU')
# model = VAE_SKA(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)


def train(epoch, train_loader):
    model.train()
    model.cuda()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):   # Tensor, [batch_size, col_vec]
        data = data.to(device)  # 'cpu' or 'cuda'
        optimizer.zero_grad()

        # print('data shape: ', data[:, :-1].shape)
        # exit()
        recon_batch, mu, logvar = model(data)   # forward
        loss = loss_function(recon_batch, data, mu, logvar, output_dim)
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
    from_raw_data = False

    if from_raw_data == True:
        # Training data processing
        train_skeleton_dict, test_skeleton_dict = prepare_training_test_data()
        # load_and_save_raw_data()

        for i in action_class_list:
            # Data load from NTU DB and save
            # Training Data Set
            train_data, total_frame, fc_list = skeleton_motion_data_collect(train_skeleton_dict[i])
            np_train_data = skeleton_coordinate_transform(train_data, total_frame)
            print('fc_list: ', fc_list, 'np_train_data: ', np_train_data, ' shape: ', np_train_data.shape)
            save_name = save_skeleton_data(np_train_data, [i], fc_list, data_type='Train')
            train_skeleton_data, _, fc_list = load_skeleton_data(save_name)

            # Test Data Set
            test_data, total_frame, fc_list = skeleton_motion_data_collect(test_skeleton_dict[i])
            np_test_data = skeleton_coordinate_transform(test_data, total_frame)
            print('fc_list: ', fc_list, 'np_test_data: ', np_test_data, ' shape: ', np_test_data.shape)
            save_name = save_skeleton_data(np_test_data, [i], fc_list, data_type='Test')
            # test_skeleton_data, _, fc_list = load_skeleton_data(save_name)
    else:
        all_train_skeleton_data = np.empty((0, 76))  #np.array([])
        # Load already generated skeleton data
        for i in action_class_list:
            np_train_skeleton_data, _, fc_list = load_skeleton_data(skeleton_train_dict[i])
            # print('len: ', len(fc_list), ' sum: ', sum(fc_list), fc_list)
            print('i: ', i, ' shape: ', np_train_skeleton_data.shape, len(fc_list))
            all_train_skeleton_data = np.vstack((all_train_skeleton_data, np_train_skeleton_data))
            print('Finished the frame data Load of: ', i)

    print(all_train_skeleton_data.shape, all_train_skeleton_data)
    # np.random.shuffle(all_train_skeleton_data)
    # print(all_train_skeleton_data.shape, all_train_skeleton_data)

    # from SkeletonDataManager import SkeletonDataManager
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import time
    # sdm = SkeletonDataManager(skeleton_data, fc_list)
    # # plot code
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(40000):
    #     frame = sdm.get_random_scene_frame()
    #     for j in range(25):
    #         x = frame[j*3+0]
    #         y = frame[j*3+1]
    #         z = frame[j*3+2]
    #         ax.scatter(x, y, z)
    #         # print('frame: ', frame)
    #
    #     plt.xlim(-1, 1)
    #     plt.ylim(-1, 1)
    #     plt.draw()
    #     plt.pause(0.03)
    #     ax.cla()
    # exit()

    # Remove the time phase
    train_src = all_train_skeleton_data[:, :-1]
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(train_src).float(),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    args.epochs = 10
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader)
        # test(epoch, test_loader)

    # save the trained model
    save_path = os.path.join('./trained_models', 'vae')
    # save_name = 'Skel_vae_' + make_class_prefix(action_class_list)
    save_name = 'Skel_vae_Superset'
    print(save_name + '.pt')
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save(model.state_dict(), os.path.join(save_path, save_name + '.pt'))

    print('finish!')


