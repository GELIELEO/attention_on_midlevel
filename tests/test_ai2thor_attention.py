from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.attention.cbam import CBAM
from algorithms.attention.bam import BAM
import torch
from visualpriors.transforms import multi_representation_transform
from visualpriors.transforms import representation_transform
from visualpriors.transforms import max_coverage_featureset_transform

N_EPISODES = 1
mode = ['autoencoding', 'depth_euclidean']
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    config_dict = {'max_episode_length': 2000, 'use_priors': False, 'agentMode':'thor'}
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    cbam = CBAM(gate_channels=8, reduction_ratio=3).cuda()
    for episode in range(N_EPISODES):
        state = env.reset()
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action, return_event=False)
            state = torch.from_numpy(state).unsqueeze(0).to(default_device) # 4 dim
            # print(state.shape)

            state_priors = representation_transform(state, 'autoencoding', device=default_device)
            # multi_state_priors = multi_representation_transform(state, mode)
            # max_cover_priors = max_coverage_featureset_transform(state,4)
            state_attention = cbam(state_priors.squeeze())
            
            print('state_attention',state_priors.shape) # [1, 8, 19, 19]
            # print('state_attention',multi_state_priors.shape) # [1, 8, 19, 19]
            # print('state_attention',multi_state_priors.shape) # [1, 8, 19, 19]
            print('state_attention',state_attention.shape) # [1, 8, 19, 19]
'''
# interger priors to env
if __name__ == '__main__':
    config_dict = {'max_episode_length': 2000, 'grayscale': False, 'use_priors': True, 'priors':'max_cover', 'k_max_cover':3, 'manual_mode':["autoencoding", "depth_euclidean",'curvature']}
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    state = env.reset()
    obs_dim = env.observation_space.shape
    print(obs_dim)

    for episode in range(N_EPISODES):
        
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action, return_event=False)
            # state = torch.from_numpy(state).unsqueeze(0).to(default_device) # 4 dim
            print(state.shape)
'''