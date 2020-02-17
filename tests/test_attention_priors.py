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
    config_dict = {'max_episode_length': 2000, 'use_priors': False, "priors": "max_cover"}
    env = RoboThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    cbam = BAM(gate_channels=3, reduction_ratio=1).cuda()
    cbam.eval()
    for episode in range(N_EPISODES):
        state = env.reset()
        pre_state = state
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action, return_event=False)
            
            print('input shape',state.shape)

            start_time = time.time()

            if torch.is_tensor(state): # priors
                input = torch.cat([state.unsqueeze(0), pre_state.unsqueeze(0)],dim=0)
                state_attention = cbam(input)
            else: # raw
                state_attention = cbam(torch.from_numpy(state).to(device).unsqueeze(0))
            
            print('>>>>>>>> used time', time.time() - start_time)

            pre_state = state
            
            # print('state',state.shape)
            # print('state_priors',state_priors.shape) # [1, 8, 19, 19]
            # print('state_attention',state_attention.shape) # [1, 8, 19, 19]

'''
conclusion:
1, input to CBAM, the batch size can be 1, but the input to BAM, batch size must bigger than 1.
2, using input from priors cost less time (3 priors)
'''
