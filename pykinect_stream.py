from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import ctypes
import pygame
import sys

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]

from CommonObject import *
import time
from Skeleton import Skeleton, Category, Body_struct, Joint_struct
from skeleton_encoding import skeleton_coordinate_transform

from NAO_AMR import NAO_AMR
from MR_Demo import MR_Demo

# ==================================================
# User Parameters
socket_comm = True
retarget_mode = 'C3PO'  # ['Analytic', 'C3PO']
target_env = 'CHREO'    # ['CHREO', 'VREP']
target_robot = 'NAO'    # ['NAO', 'BAXTER', 'C3PO']
tcp_port = 5007
# ==================================================


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface(
            (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data
        self._bodies = None

        # state
        self._cmd_state = {'state': 'free', 'color': (128, 128, 128)}  # [free, record]
        self.trs_delay = 0

    def cmd_state_transition(self):
        transition = {'from': self._cmd_state['state'], 'to': self._cmd_state['state']}
        duration = time.time() - self.trs_delay
        if duration < 2:
            return transition

        self.trs_delay = time.time()
        if self._cmd_state['state'] == 'free':    # free -> record
            transition['from'] = self._cmd_state['state']
            self._cmd_state = {'state': 'record', 'color': (255, 0, 0)}     # state transition!
            transition['to'] = self._cmd_state['state']
        elif self._cmd_state['state'] == 'record':  # record -> free
            transition['from'] = self._cmd_state['state']
            self._cmd_state = {'state': 'free', 'color': (128, 128, 128)}   # state transition!
            transition['to'] = self._cmd_state['state']
        else:
            raise ValueError

        return transition

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except:  # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);

        # Right Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight,
                            PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight,
                            PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight,
                            PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft,
                            PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft,
                            PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight,
                            PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight,
                            PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def text_objects(self, text, font, color, background=(255, 255, 255)):
        textSurface = font.render(text, True, color, background)
        return textSurface, textSurface.get_rect()

    def draw_circle_trigger(self, joints, jointPoints, pos, radius, width=8):  # pos: (x, y)
        joint0State = joints[PyKinectV2.JointType_Neck].TrackingState

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good
        right_hand_tip_pos = (jointPoints[PyKinectV2.JointType_HandTipRight].x, jointPoints[PyKinectV2.JointType_HandTipRight].y)

        dist = np.sqrt((pos[0]-right_hand_tip_pos[0])**2 + (pos[1]-right_hand_tip_pos[1])**2)
        transition = None
        if dist <= radius:
            transition = self.cmd_state_transition()    # state transition

        # just draw
        font = pygame.font.Font('freesansbold.ttf', 50)
        color = self._cmd_state['color']
        pygame.draw.circle(self._frame_surface, color, pos, radius, width)
        TextSurf, TextRect = self.text_objects(self._cmd_state['state'], font, color=color)
        TextRect.center = pos
        self._frame_surface.blit(TextSurf, TextRect)
        return transition

    def run(self):
        # -----[ For socket Communication, Server ] -----
        if socket_comm:
            conn = SocketCom('localhost', tcp_port)
            print('Port is Opened and Wait for the connection..')

            if retarget_mode == 'Analytic':  # direct connect to the NAO agent
                nao = NAO_AMR()
                conn.socket_connect()
                conn.write_socket({'header': 'start Kinect!', 'data': []})  # start signal
            else:   # indirect connect to the NAO by C-3PO model
                demo = MR_Demo(target_robot=target_robot, target_env=target_env, tcp_port=tcp_port)
                if target_env == 'CHREO': conn.socket_connect()

        # -------- Main Program Loop -----------
        demo_body_idx = -1  # only for the firstly detected human
        socket_count = 0
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            # --- Game logic should go here

            # --- Getting frames and drawing
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                # Get the index of the first body only
                if demo_body_idx < 0:  # no body index
                    # Get new body index
                    for i in range(0, self._kinect.max_body_count):
                        body = self._bodies.bodies[i]
                        if body.is_tracked:
                            demo_body_idx = i
                            break
                else:                   # already have a body index
                    body = self._bodies.bodies[demo_body_idx]
                    if not body.is_tracked:
                        demo_body_idx = -1

                if demo_body_idx >= 0:
                    # print('Demo body index: ', demo_body_idx)
                    body = self._bodies.bodies[demo_body_idx]
                    joints = body.joints
                    joint_points = self._kinect.body_joints_to_color_space(joints)

                    # get a skeleton frame from Kinect
                    skel = Skeleton()
                    body_struct = Body_struct()
                    bodyInfo = np.array([demo_body_idx, 0, 0, 0, 0,
                                         0, 0, 0, 0, body.is_tracked])
                    jointCount = PyKinectV2.JointType_Count
                    body_struct.set_body_info(bodyInfo, jointCount, phase=0)

                    for j in range(jointCount):     # 25 body joints
                        jointInfo = np.array([joints[j].Position.x, joints[j].Position.y, joints[j].Position.z,
                                              0, 0, 0, 0, 0, 0, 0, 0,
                                              body.is_tracked])
                        joint = Joint_struct(jointInfo)
                        body_struct.joints.append(joint)
                    skel.append_body(0, demo_body_idx, body_struct)

                    if socket_comm:
                        socket_count += 1
                        if socket_count >= 1:   # 6, for sync of socket delay
                            socket_count = 0
                            _from = _to = 'None'
                            skel_data = skeleton_coordinate_transform([skel], 1)  # numpy data

                            if retarget_mode == 'Analytic':
                                data = skel_data[0].tolist()[:-1]
                                lMotors = nao.left_arm_solve(data)
                                rMotors = nao.right_arm_solve(data)
                                motors = lMotors + rMotors
                                conn.write_socket({'header': 'Analytic', 'data': motors, 'from': _from, 'to': _to})
                            else:
                                if target_env == 'CHREO':
                                    retargeted_skel = demo.do_retargeting(skel_data[0].tolist()[:-1])
                                    conn.write_socket({'header': 'SetMotor', 'data': retargeted_skel,
                                                       'from': _from, 'to': _to})
                                elif target_env == 'VREP':
                                    ret = demo.do_retargeting_vrep(skel_data[0].tolist()[:-1])
                        conn.flush()
                        print('Send Success')

                    self.draw_body(joints, joint_points, SKELETON_COLORS[demo_body_idx])

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()  # (w: 1920, h: 1080)
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0, 0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();