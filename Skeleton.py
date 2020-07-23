
class Joint_struct:
    def __init__(self, joint_vec):
        if len(joint_vec) is not 12:
            raise ValueError('joint vector length is not valid..')

        # 3D location of the joint j
        self.x = joint_vec[0]
        self.y = joint_vec[1]
        self.z = joint_vec[2]

        # 2D location of the joint in corresponding depth/IR frame
        self.depthX = joint_vec[3]
        self.depthY = joint_vec[4]

        # 2D location of the joint in corresponding RGB frame
        self.colorX = joint_vec[5]
        self.colorY = joint_vec[6]

        # The quaternion orientation of the joint j
        self.orientationW = joint_vec[7]
        self.orientationX = joint_vec[8]
        self.orientationY = joint_vec[9]
        self.orientationZ = joint_vec[10]

        # The tracking state of the joint j
        self.trackingState = joint_vec[11]

    def display_joint_info(self):
        pass

class Body_struct:
    def __init__(self):
        self.bodyID = None
        self.clippedEdges = None
        self.handLeftConfidence = None
        self.handLeftState = None
        self.handRightConfidence = None
        self.handRightState = None
        self.isRestricted = None
        self.leanX = None
        self.leanY = None
        self.trackingState = None
        self.jointCount = None

        self.joints = []

    def set_body_info(self, info_vec, jointCount, phase):
        if len(info_vec) is not 10:
            raise ValueError('body infomation vector length is not valid..')

        self.bodyID = info_vec[0]
        self.clippedEdges = int(info_vec[1])
        self.handLeftConfidence = int(info_vec[2])
        self.handLeftState = int(info_vec[3])
        self.handRightConfidence = int(info_vec[4])
        self.handRightState = int(info_vec[5])
        self.isRestricted = int(info_vec[6])
        self.leanX = info_vec[7]
        self.leanY = info_vec[8]
        self.trackingState = int(info_vec[9])

        self.jointCount = jointCount
        self.phase = phase

    def display_body_info(self):
        print('---------- Body information ------------')
        print('body ID: ', self.bodyID)
        print('clipped edges: ', self.clippedEdges)
        print('hand left confidence: ', self.handLeftConfidence)
        print('hand left state: ', self.handLeftState)
        print('hand right confidence: ', self.handRightConfidence)
        print('hand right state: ', self.handRightState)
        print('isRestricted: ', self.isRestricted)
        print('lean X: ', self.leanX)
        print('lean Y: ', self.leanY)
        print('tracking state: ', self.trackingState)
        print('joint count: ', self.jointCount)


class Skeleton:
    def __init__(self):
        self.iBody = []

    def append_body(self, fIdx, bIdx, body):    # frame index, body index
        if isinstance(body, Body_struct)==False:
            raise ValueError('instance type error')

        if len(self.iBody) <= fIdx:
            [self.iBody.append([]) for i in range(fIdx - len(self.iBody)+1)]

        if len(self.iBody[fIdx]) <= bIdx:
            [self.iBody[fIdx].append([]) for i in range(bIdx - len(self.iBody[fIdx])+1)]

        # self.iBody[fIdx][bIdx].append(body)
        self.iBody[fIdx][bIdx] = body

    def get_body(self, fIdx, bIdx):
        return self.iBody[fIdx][bIdx]

    def get_frame_count(self):
        return len(self.iBody)

    def print_skeleton_info(self):
        print('-------- Skeleton information ----------')
        for f in range(len(self.iBody)):
            print('Frame Index', f, ' nBody: ', len(self.iBody[f]))
            for b in range(len(self.iBody[f])):
                self.iBody[f][b][0].display_body_info()

        print('\n')
        print('---------- Joint information ------------')
        print('')


class Category:
    def __init__(self):
        self.daily_action = 'Daily_Actions'
        self.medical_conditions = 'Medical_Conditions'
        self.mutual_conditions = 'Mutual_Conditions'

