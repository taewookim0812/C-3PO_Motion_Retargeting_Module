"""
Description: Commonly used functions, variables and a class
Author: Tae-woo Kim
Contact: twkim0812@gmail.com
"""

import numpy as np
import socket, json, os
import torch


def r2d(rad):
    return rad * (180.0/np.pi)


def d2r(deg):
    return deg * (np.pi/180.0)


# Number - Action class name pair dictionary of the NTU-DB
dicActionClass = {22: "cheer_up",
                  23: "hand_waving",
                  31: "point_to_something_with_finger",
                  37: "wipe_face",
                  38: "salute",
                  39: "put_the_palms_together"}

skeleton_train_dict = {
    22: 'A022TrainSkeletonFrame(37613_76)',
    23: 'A023TrainSkeletonFrame(37228_76)',
    31: 'A031TrainSkeletonFrame(28296_76)',
    37: 'A037TrainSkeletonFrame(42172_76)',
    38: 'A038TrainSkeletonFrame(29458_76)',
    39: 'A039TrainSkeletonFrame(27994_76)'
}

skeleton_test_dict = {
    22: 'A022TestSkeletonFrame(4132_76)',
    23: 'A023TestSkeletonFrame(4253_76)',
    31: 'A031TestSkeletonFrame(3265_76)',
    37: 'A037TestSkeletonFrame(4820_76)',
    38: 'A038TestSkeletonFrame(3401_76)',
    39: 'A039TestSkeletonFrame(3253_76)'
}

skeleton_test_file = {
    # unknown motion classes
    # upper body motion
    1: 'A001TestFileList',      # drink water
    20: 'A020TestFileList',     # put on a hat/cap
    10: 'A010TestFileList',     # clapping

    # whole body motion
    8: 'A008TestFileList',      # sitting down
    24: 'A024TestFileList',     # kicking something
    27: 'A027TestFileList',     # jump up


    22: 'A022TestFileList(59)',
    23: 'A023TestFileList(58)',
    31: 'A031TestFileList(56)',
    37: 'A037TestFileList(43)',
    38: 'A038TestFileList(56)',
    39: 'A039TestFileList(55)'
}

synthetic_motion_dict = {
    22: 'A022SyntheticMotion',
    23: 'A023SyntheticMotion',
    31: 'A031SyntheticMotion',
    37: 'A037SyntheticMotion',
    38: 'A038SyntheticMotion',
    39: 'A039SyntheticMotion'
}

motion_train_dict = {
    22: 'A022TrainMotionData(20000_14)',
    23: 'A023TrainMotionData(20000_14)',
    31: 'A031TrainMotionData(20000_14)',
    37: 'A037TrainMotionData(20000_14)',
    38: 'A038TrainMotionData(20000_14)',
    39: 'A039TrainMotionData(20000_14)',
}


# Input: list of class number ex) [22, 23, ...]
# Output: prefix of class list ex) 'A022A023...'
def make_class_prefix(action_class_list):
    action_class_prefix = 'A' + 'A'.join('{0:03}'.format(action_class_list[i]) for i in range(len(action_class_list)))
    return action_class_prefix


def load_skeleton_data(file_name):
    # Retrieve a file shape from a file name
    shape = file_name[file_name.find('(') + 1:file_name.find(')')]
    shape = list(map(lambda i: int(i), shape.split('_')))

    # read only first row that records the number of frames of each video clip
    full_path = os.path.join('./data/skeleton/', file_name + '.txt')
    with open(full_path, 'r') as f:
        fc_list = f.readline()  # fc means frame count of each video
    fc_list = list(map(int, fc_list.split(' ')))    # string to int list

    # Load actual skeleton frames
    load_data = np.loadtxt(full_path, skiprows=1).reshape(shape)

    return load_data, load_data.shape, fc_list


def load_synthetic_motion_data(file_name):
    full_path = os.path.join('./data/synthetic_motion/', file_name + '.txt')
    load_data = np.loadtxt(full_path)
    print('load data shape: ', load_data.shape)
    return load_data, load_data.shape


def load_aug_motion_data(file_name):
    shape = file_name[file_name.find('(') + 1:file_name.find(')')]
    shape = list(map(lambda i: int(i), shape.split('_')))

    full_path = os.path.join('./data/aug_motion/', file_name + '.txt')
    load_data = np.loadtxt(full_path).reshape(shape)  # os.path.join('./data', file_name + '.txt')
    print('load data shape: ', load_data.shape)
    return load_data


class SocketCom:
    def __init__(self, ipAddr, portNum):
        self.ipAddr = ipAddr
        self.portNum = portNum
        self.sock = None
        self.addr = ''

        self.read_buffer = ''

    def open_host(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ipAddr, self.portNum))
        self.sock.listen(4)  # total number of clients to access this server.
        conn, addr = self.sock.accept()
        self.sock = conn
        self.addr = addr

    def socket_connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ipAddr, self.portNum))

    def write_socket(self, objects):
        data = json.dumps(objects).encode('utf-8')
        self.sock.sendall(data)

    def read_socket(self, cut_start=None, cut_end=None):
        frac = ''
        while len(self.read_buffer) < 4096:
            if not cut_start or not cut_end:
                frac = self.read_buffer
                self.read_buffer = ''
                break
            try:
                i1 = self.read_buffer.index(cut_start)
                i2 = self.read_buffer.index(cut_end)
                frac = self.read_buffer[i1:i2+1]
                self.read_buffer = self.read_buffer[i2+1:]
                break
            except ValueError:
                self.read_buffer += self.sock.recv(2048).decode('utf-8')  # Python3.x: byte, Python2.x: string, decode from byte to unicode
                continue
        # print('type: ', type(frac), 'len: ', len(frac), 'frac: ', frac)
        # print('type: ', type(self.read_buffer), 'len: ', len(self.read_buffer), 'buffer: ', self.read_buffer)
        return json.loads(frac)

    def read_socket2(self, cut_start=None, cut_end=None):
        frac = ''
        while len(self.read_buffer) < 5000:
            self.read_buffer += json.loads(self.sock.recv(1024).decode(
                'utf-8'))  # Python3.x: byte, Python2.x: string, decode from byte to unicode
            if not cut_start or not cut_end:
                frac = self.read_buffer
                break
            try:
                i1 = self.read_buffer.index(cut_start)
                i2 = self.read_buffer.index(cut_end)
                frac = self.read_buffer[i1:i2 + 1]
                self.read_buffer = self.read_buffer[i2 + 1:]
                break
            except ValueError:
                continue
        # print('type: ', type(frac), 'len: ', len(frac), 'frac: ', frac)
        # print('type: ', type(self.read_buffer), 'len: ', len(self.read_buffer), 'buffer: ', self.read_buffer)
        return frac

    def flush(self):
        self.read_buffer = ''

    def socket_close(self):
        if self.sock:
            self.sock.close()
