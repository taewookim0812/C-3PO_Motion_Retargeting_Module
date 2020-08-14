import subprocess
import time
import vrep
from VecNormalize import bVecNormalize
import platform


class ExpCollector:
    def __init__(self, robot, portNum, comm, motion_sampling=False, enjoy_mode=False):
        """
        [TODO] Please change the "os_vrep_script_path" into where your vrep is.
        """
        if platform.system() == 'Linux':
            self.os_vrep_script_path = "/home/twkim/V-REP_PRO_EDU_V3_6_2_Ubuntu16_04/vrep.sh"
        elif platform.system() == 'Windows':
            self.os_vrep_script_path = "C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe"

        self.portNum = portNum
        self.enjoy_mode = enjoy_mode

        self.clientID = -1
        self.comm = comm

        if hasattr(robot, 'task'):
            self.robot_task = robot.task
        else:
            self.robot_task = ''
        self.robot = bVecNormalize(robot, ret=not enjoy_mode)

        if motion_sampling == True:
            self.vrep_scene_path = self.robot.venv.vrep_sampling_scene_path
        else:
            self.vrep_scene_path = self.robot.venv.vrep_learning_scene_path

    def generate_vrep_cmd(self, gui_on=False, autoStart=True, autoRunTime=0, autoQuit=True, runWithAddonScript=True, epiNum=50):
        # options for vrep
        # [-h: headless mode], [-s(period): automatically run simulation],
        # [-g: parameter value], [-q: automatic quit]
        # [-a: run with add-on script, here is for synchronous mode]
        gui_mode = "-h" if gui_on == False else ""
        autoRun_mode = "-s" if autoStart == True else ""
        autoQuit_mode = "-q" if autoQuit == True else ""
        runWithAddonScript_mode = "-a" if runWithAddonScript == True else ""
        autoRunTime_value = str(autoRunTime) if autoRunTime > 0 else ""
        enjoy_mode = 1 if self.enjoy_mode == True else 0
        if int(autoRunTime) > 0:
            autoRun_mode = autoRun_mode + str(autoRunTime)

        arguments = gui_mode + ' ' + autoRun_mode + ' ' \
                  + autoRunTime_value + ' ' + autoQuit_mode + ' ' \
                  + runWithAddonScript_mode + ' ' \
                  + '-g' + str(self.portNum) + ' ' \
                  + '-g' + str(epiNum) + ' ' \
                  + '-g' + str(enjoy_mode) + ' ' \
                  + '-g' + str(self.robot_task) + ' ' \
                  + self.vrep_scene_path   # + '-g' + self.robot_task + ' ' \
        vrep_cmd = [self.os_vrep_script_path] + arguments.split(' ')
        return vrep_cmd

    def simStart(self, gui_on=False, autoStart=True, autoQuit=True, autoRunTime=0, epiNum=50):
        vrep.simxFinish(-1)
        cmd = self.generate_vrep_cmd(gui_on=gui_on, autoStart=autoStart, autoQuit=autoQuit, autoRunTime=autoRunTime, epiNum=epiNum)
        print('cmd: ', cmd)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        # attempts to connect for 30 sec
        start = time.time()
        while self.clientID == -1:
            self.clientID = vrep.simxStart('127.0.0.1', self.portNum, True, True, 5000, 5)  # Connect to V-REP

            if abs(time.time() - start) > 30:
                break
            time.sleep(0.5)

        if self.clientID > -1:
            retSync = vrep.simxSynchronous(self.clientID, True)
            retSim = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        self.robot.initialize_robot(self.clientID)
        return self.clientID

    def simStop(self):
        returnCode = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        return returnCode

    def step(self, action, z, skel):
        return self.robot.step(action, z, skel)

    def step_broadcast(self, action):
        return self.robot.step_broadcast(action)

    def reset(self, z, skel):
        return self.robot.reset(z, skel)

    def reset_broadcast(self):
        return self.robot.reset_broadcast()

    def __del__(self):
        self.simStop()
        vrep.simxFinish(self.clientID)




