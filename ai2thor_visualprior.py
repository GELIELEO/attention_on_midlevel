import ai2thor.controller as C
import numpy as np
import torch
import torchvision.transforms.functional as TF
from utils.visual_priors_wrapper import visual_priors
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import time
mode = ['autoencoding', 'depth_euclidean',  'reshading', 'keypoints2d', 'edge_occlusion','curvature', 'edge_texture', 'keypoints3d', 'segment_unsup2d', 'segment_unsup25d','normal','segment_semantic', 'denoising' , 'inpainting',
       'class_object',
       'jigsaw', 'room_layout','class_scene', 'egomotion', 'nonfixated_pose','fixated_pose', 'point_matching', 'vanishing_point']

mode = ['curvature', 'edge_occlusion', 'edge_texture']

'''
env = C.Controller(fullscreen=False, headless=False)
env.start(port=8008, start_unity=True)
env.reset('FloorPlan3')
event = env.step(dict(action='Initialize',RenderImage=False, renderDepthImage=False, gridSize=0.01, cameraY=1, visibilityDistance=1.0))
event = env.step({'action':'ToggleMapView'})
event = env.step(dict(action='AddThirdPartyCamera', rotation=dict(x=0, y=0, z=90), position=dict(x=-1.0, z=-2.0, y=1.0)))
for i in range(10):
    event = env.step({'action':'MoveBack'})
    event = env.step({'action':'MoveRight'})
# time.sleep(100)
print(event.frame)
# print(event.frame.dtype) # uint8
'''

'''
action:
0:forward
1:backward
2:right
3:left
4:up
5:down
6:Right
7:Left
8:OpenObject
9:CloseObject
10:PickupObject
11:PutObject
'''

env = AI2ThorEnv()
state = env.reset()
state_dim = state.shape
action_dim = env.action_space.n
# env.render()
for i in range(100):
    state, rewards, done, _ = env.step(0)
    state, rewards, done, _ = env.step(2)
print(state.shape)#3/1, 128, 128
state_priors = visual_priors(state, mode)
print(state_priors[0].shape)#1,8,8,8