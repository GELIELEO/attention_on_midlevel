"""
Example use case of ai2thor wrapper. It runs N_EPISODES episodes in the Environment picking
random actions.
"""
import time

import gym
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

N_EPISODES = 1


if __name__ == '__main__':
    config_dict = {'max_episode_length': 2000}
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    for episode in range(N_EPISODES):
        start = time.time()
        state = env.reset()
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action, return_event=True)


            # print(_.metadata['objects'][0]['receptacleObjectIds'])
            # print(_.metadata['inventoryObjects'])

            # goal_found = list(filter(lambda x: x['name'].startswith('Desk'), _.metadata['objects']))
            # pot_found = list(filter(lambda x: x['name'].startswith('Pot'), _.metadata['objects']))[0]
            # print(goal_found)
            # print(pot_found.keys())
            # print(goal_found['receptacleObjectIds'])
            # if goal_found['receptacleObjectIds'] is not None:
            #     if pot_found['objectId'] in goal_found['receptacleObjectIds']:
            #         print('here!')

            # if goal_found['name'].startswith('Sink') and goal_found['visible'] and goal_found['receptacleObjectIds'] is None:
            #     time.sleep(5)
                
            # for obj in _.metadata['objects']:
            #     if obj['name'].startswith('SinkBasin') and obj['visible']:
            #         print(obj['name'])
            #         print(obj['receptacleObjectIds'])
            #         time.sleep(5)

            if done:
                break

            if step_num + 1 > 0 and (step_num + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1,
                                         (step_num + 1), max_episode_length, time.time() - start))
                start = time.time()

'''
dir(event)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
'__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', 
'__str__', '__subclasshook__', '__weakref__', '_image_depth', 'add_image', 'add_image_classes', 'add_image_depth', 'add_image_depth_meters', 'add_image_flows', 
'add_image_ids', 'add_image_normals', 'add_third_party_camera_image', 'add_third_party_image_classes', 'add_third_party_image_depth', 'add_third_party_image_flows', 
'add_third_party_image_ids', 'add_third_party_image_normals', 'class_detections2D', 'class_masks', 'class_segmentation_frame', 'color_to_object_id', 
'cv2image', 'cv2img', 'depth_frame', 'events', 'flow_frame', 'frame', 'get_object', 'image_data', 'instance_detections2D', 'instance_masks', 
'instance_segmentation_frame', 'metadata', 'normals_frame', 'object_id_to_color', 'objects_by_type', 'pose', 'pose_discrete', 'process_colors', 
'process_colors_ids', 'process_visible_bounds2D', 'screen_height', 'screen_width', 'third_party_camera_frames', 'third_party_class_segmentation_frames', 
'third_party_depth_frames', 'third_party_flows_frames', 'third_party_instance_segmentation_frames', 'third_party_normals_frames']

event.metadata['objects'][0].keys()
dict_keys(['name', 'position', 'rotation', 'cameraHorizon', 'visible', 'receptacle', 'toggleable', 'isToggled', 'breakable', 'isBroken', 
'canFillWithLiquid', 'isFilledWithLiquid', 'dirtyable', 'isDirty', 'canBeUsedUp', 'isUsedUp', 'cookable', 'isCooked', 'ObjectTemperature', 
'canChangeTempToHot', 'canChangeTempToCold', 'sliceable', 'isSliced', 'openable', 'isOpen', 'pickupable', 'isPickedUp', 'mass', 'salientMaterials', 
'receptacleObjectIds', 'distance', 'objectType', 'objectId', 'parentReceptacle', 'parentReceptacles', 'currentTime', 'isMoving', 'objectBounds'])

'''