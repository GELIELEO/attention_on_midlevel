from gym_robothor.envs.robothor_env import RoboThorEnv
from alg_robothor.attention.cbam import CBAM
from alg_robothor.attention.bam import BAM
import torch
from visualpriors.transforms import multi_representation_transform
from visualpriors.transforms import representation_transform
from visualpriors.transforms import max_coverage_featureset_transform

import time


N_EPISODES = 1
mode = ['autoencoding', 'depth_euclidean']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    config_dict = {'max_episode_length': 2000, 'use_priors': True, "priors": "max_cover", 'k_max_cover':4}
    env = RoboThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    cbam = BAM(gate_channels=8*config_dict['k_max_cover'], reduction_ratio=1).cuda()
    cbam.eval()
    for episode in range(N_EPISODES):
        state = env.reset()
        
        for step_num in range(max_episode_length):
            action = env.action_space.sample()

            start_time = time.time()
            state, reward, done, _ = env.step(action, return_event=False)
            print('>>>>>>>> used time', time.time() - start_time)
            
            print('input shape',state.shape)
            if torch.is_tensor(state): # priors
                state_attention = cbam(state.unsqueeze(0))
            else: # raw
                state_attention = cbam(torch.from_numpy(state).to(device).unsqueeze(0))
            
            
            # print('state',state.shape)
            # print('state_priors',state_priors.shape) # [1, 8, 19, 19]
            # print('state_attention',state_attention.shape) # [1, 8, 19, 19]

'''
conclusion:
1, input to CBAM, the batch size can be 1, but the input to BAM, batch size must bigger than 1.
2, using input from priors cost less time (3 priors)
'''
