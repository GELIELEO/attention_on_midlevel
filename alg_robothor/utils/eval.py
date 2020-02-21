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
    for i, env in enumerate(env_generator('test_valid')):
        if i < max_num_env:
            if standard == 'length':
                print('Evaluating scene {}'.format(env.controller.initialization_parameters['robothorChallengeEpisodeId']))
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


