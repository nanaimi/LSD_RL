import numpy as np
import cv2
import glog as log
class Reward():
    '''
    define different type reward function
    '''
    def __init__(self, setting):
        self.reward_factor = setting['reward_factor']
        self.reward_th = setting['reward_th']
        self.dis2target_last = 0

    # IN: object mask delivered by unrealcv client, mask, and pose
    # OUT: reward
    def reward_mask(self, mask,
                    factor=200,
                    right_shift=1.5):
        reward = 0
        height, width = mask.shape
        tot_num_pixels = height*width
        fov_score = (cv2.sumElems(mask)[0] / 255) / tot_num_pixels
        log.warn("FOV Score: {}".format(fov_score))

        reward = factor*np.tanh(2*np.pi*fov_score-right_shift*np.pi)
        log.warn("Reward for FOV: {}".format(reward))

        return reward, fov_score

    def reward_height(self, pose,
                      scale=150,
                      stretch=1000):
        reward = 0
        height = pose[2]
        interim = scale*np.tanh((1/stretch)*height)
        log.warn("Interim Height: {}".format(interim))

        reward  = (-1)*np.max(np.asarray([0,interim]))
        log.warn("Reward Height: {}".format(reward))

        return reward

    def reward_mask_height(self, mask, pose, done_thr,
                           factor=300,
                           right_shift=1.5,
                           scale=50,
                           stretch=1000,
                           success_thr=0.9):
        done = False
        success = False
        reward = 0
        reward_fov, fov_score = self.reward_mask(mask, factor, right_shift)
        reward_height = self.reward_height(pose, scale, stretch)

        reward = reward_fov + reward_height

        if pose[2] < done_thr:
            done = True
            if fov_score > success_thr:
                reward += 500
                success = True
            else:
                reward -= 500

        log.warn("Reward Total: {}".format(reward))
        return reward, done, success

    def reward_depth(self, depth):
        pass

    def reward_distance(self, dis2target_now):
        reward = (self.dis2target_last - dis2target_now) / max(self.dis2target_last, 100)
        self.dis2target_last = dis2target_now
        return reward

    def reward_bbox(self, boxes):
        reward = 0
        # summarize the reward of each detected box
        for box in boxes:
            reward += self.get_bbox_reward(box)

        if reward > self.reward_th:
            # get ideal target
            reward = min(reward * self.reward_factor, 10)
        elif reward == 0:
            # false trigger
            reward = -1
        else:
            # small target
            reward = 0

        return reward, boxes

    def get_bbox_reward(self, box):
        # get reward of single box considering the size and position of box
        (xmin, ymin), (xmax, ymax) = box
        boxsize = (ymax - ymin) * (xmax - xmin)
        x_c = (xmax + xmin) / 2.0
        x_bias = x_c - 0.5
        discount = max(0, 1 - x_bias ** 2)
        reward = discount * boxsize
        return reward
