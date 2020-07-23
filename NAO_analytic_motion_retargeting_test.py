"""
Description: Analytic Motion Retargeting Test
Python Ver: 3.6 ~
Reference: "A Simple and Fast Geometric Kinematic Solution for Imitation of Human Arms by a NAO Humanoid Robot"
Author: Taewoo Kim
Contact: twkim0812@gmail.com
"""

from CommonObject import *
from NAO_AMR import NAO_AMR

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def draw_skeleton_frame(ax, frame, title=''):
    for j in range(25):
        x = frame[j*3+0]
        y = frame[j*3+1]
        z = frame[j*3+2]
        ax.scatter(x, y, z, s=5)
        p = connecting_joint[j]-1    # pair joint index

        xp = frame[p * 3 + 0]
        yp = frame[p * 3 + 1]
        zp = frame[p * 3 + 2]
        ax.plot([x, xp], [y, yp], [z, zp])
        # print('frame: ', frame)
    ax.title.set_text(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)


if __name__ == "__main__":
    #### Test skeleton data reading
    f = open('test_skeleton_pose.txt', 'r')
    sk_poses = []
    pose = []
    while True:
        temp = f.readline()
        pose.append(temp.rstrip())
        if temp.find(']') >= 0:
            sk_poses.append(list(map(float, ''.join(pose).strip()[1:-1].split(','))))
            pose = []
        if not temp: break
    f.close()

    print('N of skeleton poses: ', len(sk_poses))
    print('Choose the skeleton pose index i(i<N) ::')
    pose_index = 0

    #### Skeleton visualization
    skeleton = sk_poses[pose_index]
    draw_skeleton_frame(ax, frame=skeleton, title='Original')
    plt.draw()
    plt.show(block=True)

    #### Motion Retargeting
    nao = NAO_AMR()
    # nao.disp_mat()    # Show Tr matrices of NAO's Arm
    lMotors2 = nao.left_arm_solve(skeleton)
    rMotors2 = nao.right_arm_solve(skeleton)
    print('lMotors(deg): ', list(map(r2d, lMotors2)))
    print('rMotors(deg): ', list(map(r2d, rMotors2)))
