#! /usr/bin/env python
# noted by chan
#orion -v --debug hunt -c ./orion_config.yaml -n ppo_test ./algorithms/ppo/main.py --epochs~'fidelity(5000, 10000)' --lr~'uniform(1e-5, 1)'
#orion -v --debug hunt -c ./orion_config.yaml -n nav_ppo ./alg_robothor/ppo/main.py --gamma~'uniform(.95,.9995)' --epochs~'fidelity(5000, 10000)' --lr~'loguniform(1e-5, 1.0)'
#orion -v hunt -c ./orion_config.yaml -n nav_ppo_entropy ./alg_robothor/ppo/main.py  --epochs~'fidelity(100,150)' --lr~'loguniform(1e-5, 1.0)' --entropy_coef~'uniform(1e-4, 1.0)'  --max-kl~'uniform(0.01, 0.3)'

import os
import torch
import numpy as np
from torch.multiprocessing import SimpleQueue, Process, Value, Event, Barrier
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from alg_robothor.ppo.core import ActorCritic, PPOBuffer, count_vars
from alg_robothor.ppo.worker import worker
from alg_robothor.ppo.learner import learner
from gym_robothor.envs.robothor_env import RoboThorEnv


def train_ai2thor(model, args, rank=0, b=None):

    seed = args.seed + 10000 *rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # torch.cuda.set_device(rank)
    # device = torch.device(f'cuda:{rank}')
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     os.environ['DISPLAY'] = f':{rank}'

    model = model.to(device)
    model.share_memory()

    # Experience buffer
    storage = PPOBuffer(model.obs_shape, args.steps, args.num_workers, args.state_size, args.gamma, device=device)
    storage.share_memory()
        
    #torch.multiprocessing.set_start_method('spawn')
    # start multiple processes
    ready_to_works = [Event() for _ in range(args.num_workers)]
    exit_flag = Value('i', 0)
    queue = SimpleQueue()

    processes = []
    task_config_file = "config_files/NavTaskTrain.json"
    # start workers
    for worker_id in range(args.num_workers):
        p = Process(target=worker, args=(worker_id, model, storage, ready_to_works[worker_id], queue, exit_flag, task_config_file))
        p.start()
        processes.append(p)

    # start trainer
    train_params = {"epochs": args.epochs,
                    "steps": args.steps,
                    "world_size": args.world_size,
                    "num_workers": args.num_workers
                    }
    ppo_params = {"clip_param": args.clip_param,
                  "train_iters": args.train_iters,
                  "mini_batch_size": args.mini_batch_size,
                  "value_loss_coef": args.value_loss_coef,
                  "entropy_coef": args.entropy_coef,
                  "rnn_steps": args.rnn_steps,
                  "lr": args.lr,
                  "max_kl": args.max_kl
                  }
    
    distributed = False
    if args.world_size > 1:
        if distributed==True:
            distributed = True
            # Initialize Process Group, distributed backend type
            dist_backend = 'nccl'
            # Url used to setup distributed training
            dist_url = "tcp://127.0.0.1:23456"
            print("Initialize Process Group... pid:", os.getpid())
            dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=rank, world_size=args.world_size)
            # Make model DistributedDataParallel
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    else: print('Distribution Forbidden ~')
        
    learner(model, storage, train_params, ppo_params, ready_to_works, queue, exit_flag, rank, distributed, b)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2048)#4 2048  #total number of steps across all processes per epoch in one world
    parser.add_argument('--num-workers', type=int, default=2)#1 4
    parser.add_argument('--mini-batch-size', type=int, default=128)#4 128
    parser.add_argument('--epochs', type=int, default=10000)#100 orion
    parser.add_argument('--train-iters', type=int, default=10)#4 10
    parser.add_argument('--model-path', type=str, default=None)

    parser.add_argument('--state-size', type=int, default=128)#512 128
    parser.add_argument('--hidden-sizes', type=list, default=(64,64))

    parser.add_argument('--rnn-steps', type=int, default=16)#4 16 ?
    parser.add_argument('--gamma', type=float, default=0.99)#
    parser.add_argument('--lr', type=float, default=3e-4)# stable | reward increase
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--value_loss_coef', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--max-kl', type=float, default=0.3)# stable | slow
    
    parser.add_argument('--use-attention', type=bool, default=False)
    parser.add_argument('--attention', type=str, default='BAM')

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    # get observation dimension
    env = RoboThorEnv(config_file="config_files/NavTaskTrain.json")
    env.init_pos = {'x':0, 'y':0, 'z':0}
    env.init_ori = {'x':0, 'y':0, 'z':0}
    env.task.target_id = 'Apple|+01.98|+00.77|-01.75'
    env.reset()
    
    obs_dim = env.observation_space.shape
    # Share information about action space with policy architecture
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['state_size'] = args.state_size
    ac_kwargs['hidden_sizes'] = args.hidden_sizes
    ac_kwargs['use_attention'] = args.use_attention
    ac_kwargs['attention'] = args.attention
    
    env.close()

    # Main model
    print("Initialize Model...")
    # Construct Model
    ac_model = ActorCritic(obs_shape=obs_dim, **ac_kwargs)
    if args.model_path:
        ac_model.load_state_dict(torch.load(args.model_path))
    # Count variables
    var_counts = tuple(count_vars(m) for m in [ac_model.policy, ac_model.value_function, ac_model.feature_base])
    print('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)

    if args.world_size > 1:
        processes = []
        b = Barrier(args.world_size)
        for rank in range(args.world_size):
            p = Process(target=train_ai2thor, args=(ac_model, args, rank, b))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            print("process ", p.pid, " joined") 
    else:
        train_ai2thor(ac_model, args)
    
    print("main exits")
