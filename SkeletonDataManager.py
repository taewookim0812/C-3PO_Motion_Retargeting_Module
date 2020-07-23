
import numpy as np
import copy, time


class SkeletonDataManager:
    def __init__(self, np_train_data, fc_list):
        self.fc_list = copy.deepcopy(fc_list)
        self.np_train_data = copy.deepcopy(np_train_data)

        np.random.seed(int(time.time()))
        self.rand_scene = np.random.permutation(len(self.fc_list))
        print('rand scene: ', self.rand_scene)
        self.scene_index = 0
        self.frame_index = 0
        self.frame_count = 0
        self.rand_scene_idx = 0

        self.start = 0
        self.end = 0

    def get_random_scene_frame(self):
        first_sub_scene = False
        # check the frame count
        if self.frame_count <= 0:
            # get new scene
            self.rand_scene_idx = self.rand_scene[self.scene_index]
            self.frame_count = self.fc_list[self.rand_scene_idx]

            self.start = sum(self.fc_list[:self.rand_scene_idx])
            self.end = self.start + self.frame_count-1
            first_sub_scene = True

        index = self.start + self.frame_index

        last_sub_scene = False
        if index < self.end:
            self.frame_index += 1
            self.frame_count -= 1
        else:   # end of frame
            last_sub_scene = True       # True at the end of the sub-scene!
            self.frame_index = 0
            self.frame_count = 0
            self.scene_index += 1
            if self.scene_index >= len(self.rand_scene):    # end of a scene
                self.rand_scene = np.random.permutation(len(self.fc_list))
                self.scene_index = 0

        return self.np_train_data[index, :], last_sub_scene, first_sub_scene

    def frame_reset(self):
        self.frame_index = 0

    def scene_reset(self):
        self.scene_index = 0

    def reset_all(self):
        self.frame_reset()
        self.scene_reset()


