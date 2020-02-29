"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os

import ai2thor.controller
import numpy as np
from skimage import transform
from collections import defaultdict

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_robothor.image_processing import rgb2gray
from gym_robothor.utils import read_config
import gym_robothor.tasks

import torch
from visualpriors.transforms import multi_representation_transform
from visualpriors.transforms import max_coverage_featureset_transform
import json
import cv2
import math
import random

ALL_POSSIBLE_ACTIONS = ['LookUp', 'RotateLeft', 'MoveAhead', 'MoveBack', 'RotateRight', 'LookDown']
# ALL_POSSIBLE_ACTIONS = ['LookUp', 'RotateLeft', 'MoveAhead', 'RotateRight', 'LookDown']


class RoboThorEnv(gym.Env):
    """
    Wrapper base class
    """
    def __init__(self, seed=None, config_file='config_files/NavTaskTrain.json', config_dict=None):
        """
        :param seed:         (int)   Random seed
        :param config_file:  (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: config_dict: (dict)  Overrides specific fields from the input configuration file.
        """

        # Loads config settings from file
        self.config = read_config(config_file, config_dict)
        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)

        # priors vision settings
        self.use_priors = self.config['use_priors']
        if self.use_priors:
            self.priors = self.config['priors']
            self.mode = self.config['manual_mode']
            self.k = self.config['k_max_cover']
        
        # changable params
        self.scene = self.config['scene']
        self.init_pos = None
        self.init_ori = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = None
       

        # Image settings
        self.event = None


        # Create task from config
        try:
            self.task = getattr(gym_robothor.tasks, self.config['task']['task_name'])(**self.config)
        except Exception as e:
            raise ValueError('Error occurred while creating task. Exception: {}'.format(e))
        
        # Start ai2thor
        '''
        params stored to env.controller.initialization_parameters >>>>>>>>>>>>>>>>>>>>
        'cameraY': 0.675, 'fieldOfView': 90.0, 'rotateStepDegrees': 30, 'visibilityDistance': 1.0, 'gridSize': 0.25, 
        'agentType': 'stochastic', 'agentMode': 'bot', 'continuousMode': True, 'snapToGrid': False, 
        'applyActionNoise': True, 'renderDepthImage': False, 'renderClassImage': False, 'renderObjectImage': False
        '''
        self.controller = ai2thor.controller.Controller(width=self.config['width'], height=self.config['height'], **self.config['initialize'])
        
        if self.config.get('build_file_name'):
            # file must be in gym_ai2thor/build_files
            self.build_file_path = os.path.abspath(os.path.join(__file__, '../../build_files',
                                                                self.config['build_file_name']))
            print('Build file path at: {}'.format(self.build_file_path))
            if not os.path.exists(self.build_file_path):
                raise ValueError('Unity build file at:\n{}\n does not exist'.format(
                    self.build_file_path))
            self.controller.local_executable_path = self.build_file_path


    def step(self, action, verbose=True, return_event=False):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        action_str = self.action_names[action]

        # visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]

        # if/else statements below for dealing with up to 13 actions
        if action_str.startswith('Rotate'):
            self.event = self.controller.step(dict(action=action_str))
        
        elif action_str.startswith('Move') or action_str.startswith('Look'):
            # Move and Look actions
            self.event = self.controller.step(dict(action=action_str))
        
        elif action_str == 'Stop':
            self.event = self.controller.step(dict(action=action_str))
        
        else:
            raise NotImplementedError('action_str: {} is not implemented'.format(action_str))
        
        
        target_obj = self.event.get_object(self.task.target_id)
        cur_pos = self.event.metadata['agent']['position']
        cur_ori = self.event.metadata['agent']['rotation']
        tgt_pos = target_obj['position']
        state_image, bear = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos)

        reward, done = self.task.transition_reward(self.event)
        
        if return_event:
            info = [self.event, bear]
        else:
            info = bear
        
        return state_image, reward, done, info

    
    def preprocess(self, img, cur_pos, cur_ori, tgt_pos):
        """
        Compute image operations to generate state representation
        """
        # TODO: replace scikit image with opencv
        # input shape: width,  height, 3
        img = transform.resize(img, self.config['resolution'], mode='reflect')
        img = img.astype(np.float32)
        # cv2.imshow('img',img)
        # cv2.waitKey(1)

        
        if self.config['grayscale']:
            img = rgb2gray(img) 
            img = np.moveaxis(img, 2, 0)
            img = torch.from_numpy(img).float()#3 dims  1 channel, tensor
            # img = img / 255. # already done in ai2thor

        elif self.use_priors:
            img = np.moveaxis(img, 2, 0)
            img = torch.Tensor(img).unsqueeze(0)

            if self.priors == 'manual':
                # img = multi_representation_transform(img/255., self.mode)
                img = multi_representation_transform(img, self.mode)
            elif self.priors == 'max_cover':
                # img = max_coverage_featureset_transform(img/255., self.k)
                img = max_coverage_featureset_transform(img, self.k)
            else: raise NotImplementedError
            img = img.squeeze(0)# 3 dims, tensor

        else:
            img = torch.from_numpy(np.moveaxis(img, 2, 0))# 3 dims, 3 channels, tensor
            # img = img / 255. # already done in ai2thor
        # print(img)
        
        # all to be tensor.
        cur_pos = torch.tensor([cur_pos['x'], cur_pos['z']])
        tgt_pos = torch.tensor([tgt_pos['x'], tgt_pos['z']])
        cur_ori = torch.tensor([cur_ori['y']])
        vector = (tgt_pos - cur_pos)
        dir = math.atan2(vector[1], vector[0])/math.pi*180
        
        # -180 ~ 180
        # cur_ori = -cur_ori if cur_ori<180 else 360-cur_ori
        # splm_ori = cur_ori-180 if cur_ori>0 else cur_ori+180
        # dir = dir
        

        # 0 ~ 360
        cur_ori = 360 - cur_ori
        splm_ori = cur_ori + 180 if cur_ori<180 else cur_ori-180
        # print(cur_ori, splm_ori)
        dir = 360+dir if dir < 0 else dir
        # print(dir)

        if splm_ori > cur_ori:
            if dir > cur_ori and dir < splm_ori:# left
                angle = dir-cur_ori
            elif dir < cur_ori: # right
                angle = dir-cur_ori
            else:
                angle = -(360-dir+cur_ori)
        else:
            if dir > splm_ori and dir < cur_ori:# right
                angle = dir-cur_ori
            elif dir < splm_ori:#left
                angle = 360-cur_ori+dir
            else:
                angle= dir-cur_ori

        # print(angle)

        dis = torch.tensor([np.sqrt(np.sum(np.square(np.array(cur_pos)-np.array(tgt_pos))))])
        
        bear = torch.cat([dis/8.0, angle/360.0])
        bear = bear.to(self.device)
        # print(bear)
        '''
        fake_img = torch.stack([i.expand_as(img[0]) for i in bear])
        # print('fake', fake_img) # 3 dims
        
        img = torch.cat([img, fake_img], axis=0).to(self.device) #  cuda if have
        # print(img.shape)
        '''
        return img, bear

    def reset(self):
        print('Resetting environment and starting new episode')
        # resetting scene
        self.event = self.controller.reset(scene=self.scene)

        #resetting pos & ori
        assert self.init_pos != None and self.init_ori != None
        teleport_action = dict(action='TeleportFull')
        teleport_action.update(self.init_pos)
        self.controller.step(action=teleport_action)
        self.controller.step(action=dict(action='Rotate', rotation=dict(y=self.init_ori, horizon=0.0)))
        
        # initialize action must be evolved after changing the pos and ori, cannot reset anymore here!!!
        self.event = self.controller.step(dict(action='Initialize', **self.config['initialize']))
        
        # resetting task
        self.task.reset()

        # state preprocessing
        assert self.task.target_id != None
        target_obj = self.event.get_object(self.task.target_id)
        cur_pos = self.event.metadata['agent']['position']
        cur_ori = self.event.metadata['agent']['rotation']
        tgt_pos = target_obj['position']
        self.task.pre_distance = np.sqrt(np.sum(np.square(np.array([cur_pos['x'], cur_pos['z']])-np.array([tgt_pos['x'], tgt_pos['z']]))))
        state, bear = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos)
        
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(state.shape[0], state.shape[1], state.shape[2]),
                                            dtype=np.uint8)
        return state, bear

    def render(self, mode='human'):
        # raise NotImplementedError
        self.controller.step(dict(action='ToggleMapView'))

    def seed(self, seed=None):
        self.np_random, seed_new = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed_new

    def close(self):
        self.controller.stop()


def env_generator(split:str):
    split_path = os.path.join(os.path.dirname('gym_robothor/data_split/'), split + ".json")
    with open(split_path) as f:
        episodes = json.loads(f.read())
    
    env = RoboThorEnv(config_file='config_files/NavTaskTrain.json')
    
    while True:
        # e = random.choice(episodes)
        e = random.choice(list(filter(lambda x:x['difficulty']=='easy', episodes)))
        env.controller.initialization_parameters['robothorChallengeEpisodeId'] = e['id']
        env.controller.initialization_parameters['shortest_path'] = e['shortest_path']
        
        env.scene = e['scene']
        env.init_pos = e['initial_position']
        env.init_ori = e['initial_orientation']
        env.task.target_id = e['object_id']

        print('>>>>>>> Using scene {}, {} LEVEL'.format(env.controller.initialization_parameters['robothorChallengeEpisodeId'], e['difficulty']))

        yield env


if __name__ == '__main__':
    import time

    for i in env_generator('train'):
        i.render() # render() always at the last
        time.sleep(0.1)