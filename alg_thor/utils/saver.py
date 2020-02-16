from contextlib import suppress
import os
import torch

class TrainingSaver:
    def __init__(self, networks:list, optimizer=None):
        self.checkpoint_path = 'model/checkpoint-{checkpoint}.pth'
        self.restore_path = 'model/checkpoint-{checkpoint}.pth'
        self.networks = networks
        self.optimizer = optimizer

    # self.saving_period = config.get('saving_period', 10 ** 6 // 5)   
    # def after_optimization(self):
    #     iteration = self.optimizer.get_global_step()
    #     if iteration % self.saving_period == 0:
    #         self.save()

    def save(self, iteration):
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration))
        model = dict()
        
        for i, network in enumerate(self.networks):
            model['network'+str(i)] = network.state_dict()
        if self.optimizer is not None:
            model['optimizer'] = self.optimizer.state_dict()
        
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(filename))
        torch.save(model, open(filename, 'wb'))

    def restore(self, iteration):
        filename = self.restore_path.replace('{checkpoint}', str(iteration))
        dir = os.path.dirname(filename)
        file = os.path.basename(filename)
        

        print('checkpoint dir:', dir) # checkpoint dir: model
        print('checkpoint base name:', file) # checkpoint base name: checkpoint-420.pth

        state = torch.load(open(os.path.join(dir, file), 'rb'))
        
        if 'optimizer' in state and self.optimizer is not None: self.optimizer.load_state_dict(state['optimizer'])
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(state['network'+str(i)])
