
import numpy as np
from gym.spaces.box import Box
from vecEnv import VecEnv
import vrep
import torch
import os

class NAO_MIMIC(VecEnv):
    def __init__(self, task='', rModel=None):
        VecEnv.__init__(self,
                        observation_space=Box(low=float('-inf'), high=float('inf'), shape=(1, 7), dtype=np.float32),
                        state_space=Box(low=float('-inf'), high=float('inf'), shape=(1, 7), dtype=np.float32),
                        action_space=Box(low=-1, high=1, shape=(7,), dtype=np.float32),
                        )

        self.clientID = -1
        self.robot_name = 'NAO_MIMIC'
        self.task = task
        self.frame_skip = 1
        dirpath = os.path.dirname(__file__)

        self.vrep_sampling_scene_path = os.path.join(dirpath, 'vrep', 'nao_mimic_sampling.ttt').replace("\\", "/")
        self.vrep_learning_scene_path = os.path.join(dirpath, 'vrep', 'nao_mimic.ttt').replace("\\", "/")

        self.vision_handle1 = None
        self.vision_handle2 = None

        # NAO decoder
        self.rModel = rModel

    def get_robot_name(self):
        return self.robot_name

    def initialize_robot(self, clientID):   # set client ID, get vision sensor handles
        self.clientID = clientID

        # retrieve vision sensor handle
        if self.clientID != -1:
            res1, v1 = vrep.simxGetObjectHandle(clientID, 'NAO_vision1', vrep.simx_opmode_oneshot_wait)
            res2, v2 = vrep.simxGetObjectHandle(clientID, 'NAO_vision2', vrep.simx_opmode_oneshot_wait)

            self.vision_handle1 = v1 if res1 == vrep.simx_return_ok else None
            self.vision_handle2 = v2 if res2 == vrep.simx_return_ok else None

    def calc_reward(self, skel, outFloats, ampConst=-2.0):
        # outFloats(NAO) : [lsx,lsy,lsz, lex,ley,lez, lwx,lwy,lwz, rsx,rsy,rsz, rex,rey,rez, rwx,rwy,rwz]
        # skel: [x1,y1,z1, x2,y2,z2, ...]
        # In skeleton, left arm: [5, 6, 7], -> [4*3+0, 4*3+1, 4*3+2]
        # In skeleton, right arm: [9, 10, 11], -> [8*3+0, 9*3+1, 10*3+2]

        # x: measured value
        x = np.array(outFloats[:14]).flatten()

        nao = outFloats[14:]
        step = 3

        # skeleton joint points
        sls = np.array(skel[4*step+0:4*step+3])
        sle = np.array(skel[5*step+0:5*step+3])
        slw = np.array(skel[6*step+0:6*step+3])

        srs = np.array(skel[8*step+0:8*step+3])
        sre = np.array(skel[9*step+0:9*step+3])
        srw = np.array(skel[10*step+0:10*step+3])

        nls = np.array(nao[0*step+0:0*step+3])
        nle = np.array(nao[1*step+0:1*step+3])
        nlw = np.array(nao[2*step+0:2*step+3])

        nrs = np.array(nao[3*step+0:3*step+3])
        nre = np.array(nao[4*step+0:4*step+3])
        nrw = np.array(nao[5*step+0:5*step+3])

        # print('sls: ', sls, ' sle: ', sle, ' slw: ', slw)
        # print('srs: ', srs, ' sre: ', sre, ' srw: ', srw)
        # print('nls: ', nls, ' nle: ', nle, ' nlw: ', nlw)
        # print('nrs: ', nrs, ' nre: ', nre, ' nrw: ', nrw)

        # get direction vector
        sles = (sle - sls) / np.linalg.norm(sle - sls)  # Skeleton Left Elbow to Shoulder
        nles = (nle - nls) / np.linalg.norm(nle - nls)  # Left nao elbow to shoulder
        sres = (sre - srs) / np.linalg.norm(sre - srs)  # Right skeleton elbow to shoulder
        nres = (nre - nrs) / np.linalg.norm(nre - nrs)  # Right nao elbow to shoulder

        slwe = (slw - sle) / np.linalg.norm(slw - sle)  # Left skeleton wrist to elbow
        nlwe = (nlw - nle) / np.linalg.norm(nlw - nle)  # Left nao wrist to elbow
        srwe = (srw - sre) / np.linalg.norm(srw - sre)  # Right skeleton wrist to elbow
        nrwe = (nrw - nre) / np.linalg.norm(nrw - nre)  # Right nao wrist to elbow

        ul = np.arccos(np.clip(np.dot(sles, nles) / (np.linalg.norm(sles) * np.linalg.norm(nles)), -1, 1))  # upper left limb
        ll = np.arccos(np.clip(np.dot(slwe, nlwe) / (np.linalg.norm(slwe) * np.linalg.norm(nlwe)), -1, 1))  # lower left limb
        ur = np.arccos(np.clip(np.dot(sres, nres) / (np.linalg.norm(sres) * np.linalg.norm(nres)), -1, 1))  # upper right limb
        lr = np.arccos(np.clip(np.dot(srwe, nrwe) / (np.linalg.norm(srwe) * np.linalg.norm(nrwe)), -1, 1))  # lower right limb
        reward = 0.25*np.exp(ampConst*ul) + 0.25*np.exp(ampConst*ll) +\
                 0.25*np.exp(ampConst*ur) + 0.25*np.exp(ampConst*lr)

        return np.array([reward])

    def reset(self, z, skel=None):
        res, outInts, outFloats, outStrings, outBuffers = vrep.simxCallScriptFunction(clientID=self.clientID,
                                                                                      scriptDescription='NAO',
                                                                                      options=vrep.sim_scripttype_childscript,
                                                                                      functionName="reset",
                                                                                      inputInts=[],
                                                                                      inputFloats=[],
                                                                                      inputStrings=[],
                                                                                      inputBuffer=bytearray(),
                                                                                      operationMode=vrep.simx_opmode_blocking)

        # go to next step of simulation
        for j in range(self.frame_skip):
            ret = vrep.simxSynchronousTrigger(self.clientID)

        res, outInts, outFloats, outStrings, outBuffers = vrep.simxCallScriptFunction(clientID=self.clientID,
                                                                                      scriptDescription='NAO',
                                                                                      options=vrep.sim_scripttype_childscript,
                                                                                      functionName="get_obs",
                                                                                      inputInts=[],
                                                                                      inputFloats=[],
                                                                                      inputStrings=[],
                                                                                      inputBuffer=bytearray(),
                                                                                      operationMode=vrep.simx_opmode_blocking)

        obs = z
        if self.rModel:
            with torch.no_grad():
                mu, logvar = self.rModel.encode(torch.from_numpy(np.array(outFloats)[:14]).unsqueeze(0).float())
                robot_state = self.rModel.reparameterize(mu, logvar)
                robot_state = robot_state.numpy()
        else:
            robot_state = z

        state = robot_state

        # obs = np.array(outFloats)
        # state = np.array(outFloats)
        # obs = np.concatenate((np.array(outFloats).flatten(), z.flatten()), axis=0)
        # state = np.concatenate((np.array(outFloats).flatten(), z.flatten()), axis=0)
        done = outInts[0]
        info = outBuffers

        return obs, state

    def step(self, action, z, skel):
        # start = time.time()
        for i in range(action.shape[0]):
            res, outInts, outFloats, outStrings, outBuffers = vrep.simxCallScriptFunction(clientID=self.clientID,
                                                                                          scriptDescription='NAO',
                                                                                          options=vrep.sim_scripttype_childscript,
                                                                                          functionName="step",
                                                                                          inputInts=[],
                                                                                          inputFloats=action[i, :],
                                                                                          inputStrings=[],
                                                                                          inputBuffer=bytearray(),
                                                                                          operationMode=vrep.simx_opmode_oneshot)
            # go to next step of simulation in a synchronous way
            for j in range(self.frame_skip):
                ret = vrep.simxSynchronousTrigger(self.clientID)

        res, outInts, outFloats, outStrings, outBuffers = vrep.simxCallScriptFunction(clientID=self.clientID,
                                                                                      scriptDescription='NAO',
                                                                                      options=vrep.sim_scripttype_childscript,
                                                                                      functionName="get_obs",
                                                                                      inputInts=[],
                                                                                      inputFloats=[],
                                                                                      inputStrings=[],
                                                                                      inputBuffer=bytearray(),
                                                                                      operationMode=vrep.simx_opmode_blocking)

        # outFloats: NAO state[14 dim joints, 18 dim joint positions], type: list
        # shoulder, elbow, wrist position of both arm. 6*3=18 dim + joint 10 = 28
        rew = self.calc_reward(skel, outFloats)

        # obs = np.array(outFloats)
        # state = np.array(outFloats)
        # obs = np.concatenate((np.array(outFloats).flatten(), z.flatten()), axis=0)
        # state = np.concatenate((np.array(outFloats).flatten(), z.flatten()), axis=0)
        obs = z

        if self.rModel:
            with torch.no_grad():
                mu, logvar = self.rModel.encode(torch.from_numpy(np.array(outFloats)[:14]).unsqueeze(0).float())
                robot_state = self.rModel.reparameterize(mu, logvar)
                robot_state = robot_state.numpy()
        else:
            robot_state = z

        state = robot_state
        done = np.array([outInts[0]])
        info = list(outBuffers)

        return obs, state, rew, done, info
