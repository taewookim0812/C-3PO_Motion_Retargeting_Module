"""
NTU_DB Data Loader

@author: Tae-woo Kim
e-mail: twkim0812@gmail.com
"""

from CommonObject import *
from Skeleton import Skeleton, Category, Body_struct, Joint_struct

"""
****************************************************************************************************
Before you run this code, Your NTU database files should be organized as following.
Example)
/*your own path */NTU_DB/Daily_Actions/A022
/*your own path */NTU_DB/Mutual_Conditions/A050
...

The pairs of video clip and skeleton file corresponding to each action class should be in each sub-folder.
Example)
In /*your own path */NTU_DB/Daily_Actions/A022, it contains..
S001C001P001R001A022_rgb.avi, S001C001P001R001A022.skeleton, ...
****************************************************************************************************  

NTU-DB motion classes
http://rose1.ntu.edu.sg/datasets/actionrecognition.asp

(learned six)
[22: cheer up, 23: hand waving, 31: pointing to something with finger, 
 37: wipe face, 38: salute, 39: put palms together]

(unlearned upper body)
[1: drink water, 10: clapping, 20: put on a hat/cap]

(unlearned whole body)
[8: sit down, 24: kicking something, 27: jump up]
"""

evalDict = {'learned_six': [22, 23, 31, 37, 38, 39],
            'unlearned_upper_body': [1, 10, 20],
            'unlearned_whole_body': [8, 24, 27]}
evalTarget = 'learned_six'  # choose target motion class name within the "evalDict"

try:
    action_class_list = evalDict[evalTarget]
except KeyError:
    print('Unknown motion classes..')
    exit()

shuffled_eval = False
socket_comm = True

sync_stream = False              # Synchronized stream for phase 3 teaching
eval_train_or_test = False       # True: Eval. train data, False: Eval. test data
tcp_port = 4000

# display RGB image with skeleton
import numpy as np
import cv2
import os, glob, random
print('OpenCV version: ', cv2.__version__)

# NTU DB were captured by Kinect-v2 with 30Hz
S = 'S' + '{0:03}'.format(1)  # Setup Number
C = 'C' + '{0:03}'.format(1)  # Camera ID
P = 'P' + '{0:03}'.format(1)  # Performer ID
R = 'R' + '{0:03}'.format(1)  # Replication Number


multi_video_file_list = []
multi_skeleton_file_list = []

per_class_video_file_list = []
per_class_skeleton_file_list = []
for i in action_class_list:
    print('i: ', i)
    # Test Data Load
    A = 'A' + '{0:03}'.format(i)  # Action class label
    path = './data/skeleton'
    file_path = os.path.join(path, skeleton_test_file[i] + '.txt')
    test_skel_file = np.genfromtxt(file_path, dtype='str').tolist()

    # for class by method
    per_class_skeleton_file_list.append(test_skel_file)
    per_class_video_file_list.append([w.replace('.skeleton', '_rgb.avi') for w in test_skel_file])

    multi_skeleton_file_list += test_skel_file
    multi_video_file_list += [w.replace('.skeleton', '_rgb.avi') for w in test_skel_file]

    #### File filtering
    # my_path = '/dd/'                                # TODO, change the '/dd/' to your own path
    # category = Category()
    # file_path = os.path.join(my_path, 'NTU_DB', category.daily_action, A)
    #
    # video_file_list = sorted(glob.glob(file_path + "/*.avi"))
    # skeleton_file_list = sorted(glob.glob(file_path + "/*.skeleton"))
    # # exclude exception files
    # with open(A+'_exeception_avi_list.txt', 'r') as f_avi:
    #     exception_avi_list = f_avi.readlines()
    # with open(A + '_exeception_skeleton_list.txt', 'r') as f_skel:
    #     exception_skel_list = f_skel.readlines()
    #
    # for j in range(len(exception_avi_list)):
    #     exc_file = exception_avi_list[j].replace('\n', '')
    #     if exc_file in video_file_list: video_file_list.remove(exc_file)
    # for j in range(len(exception_skel_list)):
    #     exc_file = exception_skel_list[j].replace('\n', '')
    #     if exc_file in skeleton_file_list: skeleton_file_list.remove(exc_file)
    # multi_video_file_list += video_file_list
    # multi_skeleton_file_list += skeleton_file_list

print(len(multi_video_file_list), multi_video_file_list)
print(len(multi_skeleton_file_list), multi_skeleton_file_list)

print('-------------------')
print(len(per_class_video_file_list), per_class_video_file_list)
print(len(per_class_skeleton_file_list), per_class_skeleton_file_list)

nClass = len(per_class_video_file_list)
# d = [None]*sum([len(per_class_video_file_list[i]) for i in range(nClass)])

# cross order: [22, 23, 31, 37, 38, 39,  22, 23, ...]
crossOrder = []
p = []
for i in range(nClass):
    tmp = list(zip(per_class_video_file_list[i], per_class_skeleton_file_list[i]))
    p.append(tmp)

while sum([not p[k] for k in range(len(p))]) < nClass:
    for i in range(nClass):
        try:
            crossOrder.append(p[i].pop(0))
        except IndexError:
            continue

# segment order : [22, 22, 22,  23, 23, 23,  31, 31, 31, ...]
segmentOrder = list(zip(multi_video_file_list, multi_skeleton_file_list))

if shuffled_eval:
    random.shuffle(crossOrder)

multi_video_file_list, multi_skeleton_file_list = zip(*crossOrder)

print(len(multi_video_file_list), multi_video_file_list)
print(len(multi_skeleton_file_list), multi_skeleton_file_list)


# MATLAB based joint connecting index. -1 should be added for pythonic index.
connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]


class Indexer:
    def __init__(self, cherry_pick_list, normal_list):
        self.cherry_pick_list = cherry_pick_list
        self.normal_list = normal_list
        self.index = -1

    def get_prev_index(self):   # for backward
        prev_index = None
        if self.cherry_pick_list:
            try:
                self.index -= 1
                prev_index = self.cherry_pick_list[self.index]
            except IndexError:
                self.index = 0
                prev_index = self.cherry_pick_list[self.index]
        else:
            self.index -= 1
            if self.index < 0:
                self.index = 0
            prev_index = self.index

        return prev_index

    def get_next_index(self):
        next_index = None
        # cherry picking mode
        if self.cherry_pick_list:
            try:
                self.index += 1
                next_index = self.cherry_pick_list[self.index]
            except IndexError:
                self.index = 0
                next_index = self.cherry_pick_list[self.index]

        # Normal mode
        else:
            self.index += 1
            if self.index >= len(self.normal_list):
                self.index = 0
            next_index = self.index

        return next_index


"""
CherryPick Ex] cherryPickList = [4, 6, 40, 9, 10, 11, 32, 21, 81, 49, 55, 73, 35, 37]
Or, if you want the normal indexing, set empty list to cherry_pick_list=
"""
idx = Indexer(cherry_pick_list=[], normal_list=multi_video_file_list)

# -----[ Video Control Parameters ]-----
index = idx.get_next_index()   # initial index
backward = False
stop = False
frame_rate = 20                # Hz
period = int((1.0/frame_rate) * 1000)

# -----[ For socket Communication, Server ] -----
if socket_comm:
    from skeleton_encoding import skeleton_coordinate_transform

    conn = SocketCom('localhost', tcp_port)
    # host = 'localhost'
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((host, tcp_port))
    # s.listen(4)  # total number of clients to access this server.
    # conn, addr = s.accept()
    conn.open_host()
    ignition_msg = conn.read_socket('{', '}')
    print('Client is connected!! ', ignition_msg)

while True:
    # index < len(multi_video_file_list)
    # -----------------------
    # -----[ Data Load ]-----
    # -----------------------
    print('list number: ', index)
    print('video: ', multi_video_file_list[index])
    print('skeleton: ', multi_skeleton_file_list[index])

    id = multi_skeleton_file_list[index].find('/A')
    class_num = int(multi_skeleton_file_list[index][id+2:id+5])
    print('class number: ', class_num)

    cap = cv2.VideoCapture(multi_video_file_list[index])
    file = open(multi_skeleton_file_list[index])

    frameCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]
    skel = Skeleton()
    for f in range(frameCount):  # frameCount
        bodyCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]

        for b in range(bodyCount):
            body = Body_struct()
            bodyInfo = np.fromstring(file.readline(), sep=' ', dtype='float64')
            jointCount = np.fromstring(file.readline(), sep=' ', dtype='int32')[0]
            body.set_body_info(bodyInfo, jointCount, f*1.0/frameCount)

            for j in range(jointCount):
                jointInfo = np.fromstring(file.readline(), sep=' ', dtype='float64')
                joint = Joint_struct(jointInfo)
                body.joints.append(joint)
            skel.append_body(f, b, body)
    file.close()

    if socket_comm:
        skel_data = skeleton_coordinate_transform([skel], frameCount)   # numpy data

    # ----------------------------------------
    # -----[ Display Image and Skeleton ]-----
    # ----------------------------------------
    fCount = 0  # frame count
    while(cap.isOpened()):
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Drawing Skeleton
        for i in range(len(skel.iBody[fCount])):    # num bodies
            fBody = skel.iBody[fCount][i]
            for j in range(len(fBody.joints)):      # num joints
                # draw joints
                cx0 = int(fBody.joints[j].colorX+0.5)
                cy0 = int(fBody.joints[j].colorY+0.5)
                cv2.circle(frame, (cx0, cy0), 4, (0, 255, 0), -1)

                # draw lines
                p = connecting_joint[j] - 1
                cx1 = int(fBody.joints[p].colorX+0.5)
                cy1 = int(fBody.joints[p].colorY+0.5)
                cv2.line(frame, (cx0, cy0), (cx1, cy1), (0, 0, 255), 2)
                # print('cj: ', connecting_joint[j]-1)

        # vertical flip
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(period)
        if key == ord('q'):     # backward
            # if index > 0: index -= 1
            index = idx.get_prev_index()
            backward = True
            break
        elif key == ord('e'):   # forward
            break
        elif key == 27:         # ESC
            stop = True
            break

        # send skeleton data through socket
        # numpy -> list -> bytes -> sendalltraj
        if socket_comm:
            conn.write_socket({'header': 'skeleton_raw', 'skeleton': skel_data[fCount].tolist() + [class_num]})
            if sync_stream:
                recv = conn.read_socket('{', '}')
                print('r: ', recv)
                while recv['header'] != 'ContNTU':
                    pass

        fCount += 1
        phase = fCount * 1.0 / len(skel.iBody)
        # print('frame phase: ', phase, ' fc: ', len(skel.iBody))
        if fCount >= len(skel.iBody):
            break

    file.close()
    cap.release()
    if stop == True: break

    if backward == True:
        backward = False
    else:
        index = idx.get_next_index()
        # index += 1
        # # circular
        # if index >= len(multi_video_file_list):
        #     index = 0


cv2.destroyAllWindows()

# socket close
if socket_comm:
    conn.socket_close()
