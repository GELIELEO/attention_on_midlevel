import numpy as np
import torch
from torch.autograd import Variable

from gym_robothor.envs import env_generator
import ai2thor.util.metrics

import torch

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def reset(env, state_shape, device):
    state = env.reset()
    mask_t = torch.tensor(0., dtype=torch.float32).to(device)
    prev_a = torch.tensor(0, dtype=torch.long).to(device)
    obs_t = state
    state_t = torch.zeros(state_shape, dtype=torch.float32).to(device)
    inputs = {"observation": obs_t,
         "memory": {
             "state": state_t,
             "mask": mask_t,
             "action": prev_a
          }
         }
    return inputs

def evaluate_with_spl(model, storage):
    state_shape = storage.h_buf.shape[1]
    device = storage.device

    max_num_env = 10
    standard = 'reward'

    episode_results = []
    reward_list = []
    for i, env in enumerate(env_generator('test_valid_')):
        if i < max_num_env:
            if standard == 'length':
                episode_result = dict(shortest_path=env.controller.initialization_parameters['shortest_path'], success=False, path=[])
                episode_results.append(episode_result)
                episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])

                inputs = reset(env, state_shape, device)
                done = False

                while not done:
                    a_t, logp_t, _, v_t, state_t = model(inputs)
                    
                    with torch.no_grad():
                        state, reward, done, _ = env.step(a_t.item()) # if the data is in cuda, use item to extract it.
                    
                    episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])
                    
                    inputs["observation"] = state
                    inputs["memory"]["state"] = state_t
                    inputs["memory"]["mask"] = torch.tensor((done+1)%2, dtype=torch.float32).to(device)
                    inputs["memory"]["action"] = a_t

                    if done:
                        break
                        target_obj = env.controller.last_event.get_object(env.task.target_id)
                        episode_result['success'] = target_obj['visible']
            else:
                inputs = reset(env, state_shape, device)
                done = False
                
                while not done:
                    a_t, logp_t, _, v_t, state_t = model(inputs)
                    
                    with torch.no_grad():
                        state, reward, done, _ = env.step(a_t.item()) # if the data is in cuda, use item to extract it.
                        reward_list.append(reward)

                    inputs["observation"] = state
                    inputs["memory"]["state"] = state_t
                    inputs["memory"]["mask"] = torch.tensor((done+1)%2, dtype=torch.float32).to(device)
                    inputs["memory"]["action"] = a_t

                    if done:
                        break
        else:
            break
    if standard == 'length':
        spl = ai2thor.util.metrics.compute_spl(episode_results)
        return spl
    else:
        return np.mean(reward_list)


'''
 {
        "difficulty": "medium",
        "id": "Train_1_1_Apple_17",
        "initial_orientation": 90,
        "initial_position": {
            "x": 1.75,
            "y": 0.9009997,
            "z": -4.25
        },
        "object_id": "Apple|+01.98|+00.77|-01.75",
        "object_type": "Apple",
        "scene": "FloorPlan_Train1_1",
        "shortest_path": [
            {
                "x": 1.75,
                "y": 0.0103442669,
                "z": -4.25
            },
            {
                "x": 2.85833335,
                "y": 0.0103442669,
                "z": -3.208334
            },
            {
                "x": 4.025,
                "y": 0.0103442669,
                "z": -2.68333435
            },
            {
                "x": 4.141667,
                "y": 0.0103442669,
                "z": -2.56666756
            },
            {
                "x": 4.025,
                "y": 0.0103442669,
                "z": -2.27500057
            },
            {
                "x": 3.0,
                "y": 0.0103442669,
                "z": -2.0
            }
        ],
        "shortest_path_length": 4.340735893219212,
        "target_position": {
            "x": 1.979,
            "y": 0.7714,
            "z": -1.753
        }
    },
    '''