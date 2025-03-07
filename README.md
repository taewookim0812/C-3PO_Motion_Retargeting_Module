## \[Motion Retargeting Module\] C-3PO Motion Retargeting
This project is a motion retargeting module for human-robot motion retargeting. Human motion
is captured and converted to a skeleton motion by Kinect sensor, where the skeleton consists of
25 joints as shown in <a href="https://arxiv.org/pdf/1604.02808.pdf">this</a>. The skeleton motions are then retargeted to
robot motions frame by frame by our algorithm.

Motion retargeting module provides two methods for motion retargeting,
where the first one is an analytic approach ([paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7886806))
and the second is our C-3PO method which is based on data-driven and reinforcement learning.
Following figure represents the overall structure of our C-3PO motion retargeting system.

![Alt text](./figs/system.png)

### Update
<pre>
* Phase 2 training module is updated
* Phase 3 training module will be updated
</pre>

### Getting Started
<pre>
- Before you start, please download <a href="https://drive.google.com/file/d/1Covr-vyPli7j10EtLmKCN2-WjDzY-uLw/view?usp=sharing">"trained_models.zip"</a>
and extract the file here without any modifications of the file name.

- For training and evaluation, following paths must be modified for your environment.
    * "NTU_DB_PATH" in CommonObject.py
    * "os_vrep_script_path" in ExpCollector.py

- Please read ./robots/vrep/ReadMe.txt for initial remoteAPI setup of the V-REP simulator.

- If you want to use the exception files in ./exception_file_list, you should
modify the first part of the path to your path in accordance with your environment.
For example, in "/dd/NTU_DB/Daily_Actions/A022/S001C002P006R001A022_rgb.avi",
"/dd/" should be changed to your path (ex] /MyDB).
</pre>

* Learning and Evaluation
<pre>
- After setting learning parameters, run the "training.py" for learning.
- For evaluation, you should firstly run "enjoy1.py" and then "enjoy2.py" in order.
</pre>

* Test
<pre>
1) Set the appropriate port numbers and run the "demo.py" file of the robot control module.
2) Set the identical port number to the robot control module.
3) After initialize the arguments and user parameters, run the "pykinect_stream.py" file and enjoy.

You can choose motion retargeting mode(Analytic, C3PO), robots(NAO, Baxter, C-3PO)
and simulation environment(Choregraphy, V-REP) using the user parameters described at the top of the
"pykinect_stream.py".
</pre>

### File descriptions
<pre>
    .
    ├── algo                                    # RL algorithm files ex) PPO,...
    ├── data                                    # Temporary files for training
    │   ├── aug_motion
    │   ├── skeleton
    │   └── synthetic_motion
    ├── exception_file_list                     # Files for noisy skeleton exception
    ├── figs
    │   └── system.png                          # Figure for README file
    ├── robots
    │   |── vrep
    |   |   |── baxter_mimic.ttt                # V-REP project file for BAXTER robot
    |   |   |── c3po_mimic.ttt                  # V-REP project file for C-3PO robot
    |   |   └── nao_mimic.ttt                   # V-REP project file for NAO robot
    │   ├── BAXTER_MIMIC.py                     # BAXTER robot class for simulation based training
    │   ├── C3PO_MIMIC.py                       # C3PO robot class for simulation based training
    │   └── NAO_MIMIC.py                        # NAO robot class for simulation based training
    ├── runs                                    # For tensorboard
    ├── src                                     # baselines
    ├── trained_models                          # !!! This folder should be downloaded at first !!!
    │   |── ppo
    |   |   |── phase1                          # Augmented robot motion trajectory dataset
    |   |   |── phase2                          # Trained policy using PPO algorithm
    |   |   └── phase3                          # Fine-tuned policy in phase 3
    │   |── vae
    |   |   |── [Baxter]Motion_vae_Superset.pt  # BAXTER robot motion encoder-decoder network
    |   |   |── [C3PO]Motion_vae_Superset.pt    # C3PO robot motion encoder-decoder network
    |   |   |── [NAO]Motion_vae_Superset.pt     # NAO robot motion encoder-decoder network
    |   |   └── Skel_vae_Superset.pt            # Human skeleton encoder-decoder network
    ├── arguments.py                            # Arguments for learning
    ├── CommonObject.py                         # Commonly used functions and variables
    ├── distributions.py                        # Network class for action sampling
    ├── enjoy1.py                               # First running file for evaluation
    ├── enjoy2.py                               # Second running file for evaluation
    ├── ExpCollector.py                         # Class for V-REP environment
    ├── model.py                                # Policy network model definition
    ├── motion_encoding.py                      # Learning robot motion encoder-decoder network
    ├── MR_Demo.py                              # Demo file for motion retargeting
    ├── NAO_AMR.py                              # Analytic motion retargeting class only for NAO
    ├── NAO_analytic_motion_retargeting_test.py # NAO Analytic motion retargeting test code
    ├── pykinect_stream.py                      # Kinect based motion sampling and retargeting demo
    ├── README.md
    ├── remoteApi.dll                           # Library file for V-REP
    ├── requirements.txt
    ├── Skeleton.py                             # Class for Skeleton management
    ├── skeleton_encoding.py                    # Learning skeleton encoder-decoder network
    ├── SkeletonDataManager.py                  # Skeleton data management
    ├── storage.py                              # Class for storage and management of rollout data
    ├── test_skeleton_pose.txt                  # Skeleton frame for test
    ├── training.py                             # Training file
    ├── utils.py                                # Utility for policy network
    ├── vae_model.py                            # Variational AutoEncoder model
    ├── vecEnv.py                               # Vectorized Environment
    ├── VecNormalize.py                         # Vectorized Environment
    ├── vrep.py                                 # File for V-REP use
    └── vrepConst.py                            # File for V-REP use

[Important!!]
* In order to use the V-REP simulator in PyCharm, you should copy three files of "remoteApi.dll",
"vrep.py" and "vrepConst.py" into the main project folder.
* ASUS mother board does not support the Kinect V2.
</pre>

### External Dependencies
<pre>
* Python 3.6 ~
* <a href="https://www.coppeliarobotics.com/">CoppeliaSim(V-REP)</a>
* <a href="https://www.microsoft.com/en-us/download/details.aspx?id=44559">Kinect Windows Driver</a>
* <a href="https://pytorch.org/">Pytorch >= 1.3.1</a>
</pre>

### Requirements
<pre>
pip install -r requirements.txt
</pre>

### Publication
Please refer to the following paper and video for more detailed information.
* Kim and Lee, C-3PO: Cyclic-Three-Phase Optimization for Human-Robot Motion Retargeting based on
Reinforcement Learning, ICRA 2020,
[[Paper]](https://arxiv.org/abs/1909.11303),
[[YouTube]](https://www.youtube.com/watch?v=C37Fip1X0Y0&t=19s)

<pre>
@article{kim2019c,
  title={C-3PO: Cyclic-Three-Phase Optimization for Human-Robot Motion Retargeting based on Reinforcement Learning},
  author={Kim, Taewoo and Lee, Joo-Haeng},
  journal={arXiv preprint arXiv:1909.11303},
  year={2019}
}
</pre>


### Acknowledgment
This work was supported by the ICT R&D program of MSIP/IITP \[2017-0-00162, Development of Human-care
Robot Technology for Aging Society\].
