import os
import numpy as np
import torch
from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator


def reset(env, state_shape, device):
    state = env.reset()
    mask_t = torch.tensor(0., dtype=torch.float32).to(device)
    prev_a = torch.tensor(0, dtype=torch.long).to(device)
    if not torch.is_tensor(state):
        obs_t = torch.Tensor(state / 255.).to(device)
    else:
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


def worker(worker_id,
           policy,
           storage,
           ready_to_work,
           queue,
           exit_flag,
           use_priors = False,
           task_config_file="config_files/config_example.json"):
    '''
    Worker function to collect experience based on policy and store the experience in storage
    :param worker_id: id used for store the experience in storage
    :param policy: function/actor-critic
    :param storage:
    :param ready_to_work: condition to synchronize work and training
    :param queue: message queue to send episode reward to learner
    :param exit_flag: flag set by leaner to exit the job
    :param task_config_file: the task configuration file
    :return:
    '''

    print(f"Worker with Id:{worker_id} pid ({os.getpid()}) starts ...")

    steps_per_epoch = storage.block_size
    print('>>>>>> steps per epoch executed by this worker', steps_per_epoch)
    state_shape = storage.h_buf.shape[1]
    device = storage.device
    
    episode_rewards, episode_steps = [], []
    reward_sum, step_sum = 0., 0

    # Wait for start job
    ready_to_work.wait()

    for env in env_generator('training'):
        print('using scene {}'.format(env.controller.initialization_parameters['robothorChallengeEpisodeId']))
        if exit_flag.value != 1:
            inputs = reset(env, state_shape, device)
            for i in range(steps_per_epoch):
                with torch.no_grad():
                    a_t, logp_t, _, v_t, state_t = policy(inputs)
            
                # interact with environment
                state, reward, done, _ = env.step(a_t.item())
                # print('state.shape', state.shape, type(state.shape))
                
                reward_sum += reward  # accumulate reward within one rollout.
                step_sum += 1
                r_t = torch.tensor(reward, dtype=torch.float32).to(device)
                
                # save experience
                storage.store(worker_id,
                                inputs["observation"],
                                a_t,
                                r_t,
                                v_t,
                                logp_t,
                                inputs["memory"]["state"],
                                inputs["memory"]["mask"])
                # prepare inputs for next step
                if use_priors:
                    inputs["observation"] = state
                else:
                    inputs["observation"] = torch.Tensor(state/255.).to(device) # 128x128 -> 1x128x128
                # print('inputs["observation"]', inputs["observation"].shape)
                
                inputs["memory"]["state"] = state_t
                inputs["memory"]["mask"] = torch.tensor((done+1)%2, dtype=torch.float32).to(device)
                inputs["memory"]["action"] = a_t
                # check terminal state
                if done: # calculate the returns and GAE and reset environment
                    storage.finish_path(worker_id, 0)
                    # print(f"Worker:{worker_id} {device} pid:{os.getpid()} finishes goal at steps :{i}")
                    episode_rewards.append(reward_sum)
                    episode_steps.append(step_sum)
                    inputs = reset(env, state_shape, device)
                    reward_sum, step_sum = 0., 0
            # env does not reaches end
            if not done:
                _, _, _, last_val, _ = policy(inputs)
                storage.finish_path(worker_id,last_val)
            # print(f"Worker:{worker_id} {device} pid:{os.getpid()} begins to notify Learner Episode done")
            queue.put((episode_rewards,episode_steps, worker_id))
            # print(f"Worker:{worker_id} waits for next episode")
            episode_rewards, episode_steps = [], []
            # inputs = reset(env, state_shape)
            # reward_sum, step_sum = 0., 0
            # Wait for next job
            ready_to_work.clear()
            ready_to_work.wait()
            # print(f"Worker:{worker_id} {device} pid:{os.getpid()} starts new episode")

        else:
            env.close()
            break
    
    print(f"Worker with pid ({os.getpid()})  finished job")


def tester(model, device, n=5, task_config_file="config_files/NavTaskTrain.json"):
    episode_reward = []
    rnn_size = 128
    env = RoboThorEnv(config_file=task_config_file)
    for _ in range(n):
        # Wait for trainer to inform next job
        total_r = 0.
        done = False
        inputs = reset(env, rnn_size, device)
        while not done:
            with torch.no_grad():
                a_t, _, _, _, state_t = model(inputs)
                # interact with environment
                state, reward, done, _ = env.step(a_t.data.item())
                total_r += reward  # accumulate reward within one rollout.
                # prepare inputs for next step
                inputs["observation"] = torch.Tensor(state / 255.).to(device)
                inputs["memory"]["state"] = state_t
                inputs["memory"]["mask"] = torch.tensor((done + 1) % 2, dtype=torch.float32).to(device)
                inputs["memory"]["action"] = a_t

        episode_reward.append(total_r)
        print("Episode reward:", total_r)

    env.close()
    print(f"Average eposide reward ({np.mean(episode_reward)})")
