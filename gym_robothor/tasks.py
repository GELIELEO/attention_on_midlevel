"""
Different task implementations that can be defined inside an ai2thor environment
"""

from gym_robothor.utils import InvalidTaskParams
import numpy as np


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.max_episode_length = config.get('max_episode_length', 1000)
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class NavTask(BaseTask):
    def __init__(self, **kwargs):
        super(NavTask, self).__init__(kwargs)
        self.target_id = None

    def transition_reward(self, event):
        print(self.step_num)
        reward, done = self.movement_reward, False

        #calculate the distance:
        agent_pos = [event.metadata['agent']['position'][i] for i in event.metadata['agent']['position']]
        target_object = event.get_object(self.target_id)
        target_pos = [target_object['position'][i] for i in target_object['position']]

        dis = np.sqrt(np.sum(np.square(np.array(agent_pos)-np.array(target_pos))))
        
        print('dis',dis)
        print(target_object['name'])
        print(target_object['visible'])
        
        if dis < 1.1 and target_object['visible']:
            reward += 10
            print('>>>>>>>>>>>>>>>> Reached destination')
        
        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True
        
        self.step_num += 1

        return reward, done

    def reset(self):
        self.step_num = 0

       


