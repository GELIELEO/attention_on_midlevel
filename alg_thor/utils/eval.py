import numpy as np
import torch
from torch.autograd import Variable

from gym_robothor.envs import env_generator
import ai2thor.util.metrics


def evaluate_with_spl(env, policy, cuda, task_config_file):
    episode_results = []
    for env in env_generator('testing'):
        episode_result = dict(shortest_path= e['shortest_path'], success=False, path=[])
        episode_results.append(episode_result)

        agent = policy
        episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])
        
        state = env.reset()
        done = False

        while not done:
            action = agent(state)
            state, reward, done, _ = env.step(action)
            episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])

        if done:
            break
            target_obj = env.controller.last_event.get_object(env.task.target_id)
            episode_result['success'] = target_obj['visible']

    spl = ai2thor.util.metrics.compute_spl(episode_results)
    
    return spl

