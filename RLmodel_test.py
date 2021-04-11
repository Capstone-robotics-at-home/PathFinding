'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-04-09 22:52:28
 # @ Description: A testing file for the accuracy of the pre-trained neural network.
 '''

from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import matplotlib.pyplot as plt 
from Path_Utils.simple_RL_env import CartEnv
from Path_Utils.simple_RL_run import Net
import time 

# Hyper Parameters
N_EPISODES = 300            # Number of total episodes
N_SHOWN = 50
BATCH_SIZE = 32
EPSILON = 0.9              # greedy policy

N_ACTIONS = 3
N_STATES = 5
ENV_A_SHAPE = 0


class DQNnet():
    """ A DQN net loaded with pre-trained model """    
    def __init__(self, model_name):
        self.eval_net = torch.load(model_name)

    def choose_action(self,x): 
        x = torch.unsqueeze(torch.FloatTensor(x),0) 
        # input only one sample 
        if np.random.uniform() < EPSILON:  # greedy 
            actions_value = self.eval_net.forward(x) 
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index        
        else: # random 
            action = np.random.randint(0,N_ACTIONS) 
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) 
        return action 


def mtest(objects, file_name):
    obstacle_ls = objects['Obstacle']
    s_start = objects['Jetbot'][0]
    s_goal = objects['Target'][0]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]
    env = CartEnv(s_start, s_goal, obstacle_ls) 
    dqn = DQNnet(file_name)
    events =np.array([0,0,0,0,0])  # record the events during training: [bound, crash, deviate, reach]
    events_history = events.copy()
    episode_events = np.array([0,0,0,0,0]) # record the events during each episode
    accuracy_history = [0.0] # record the accuracy 


    print("--------------------------------Finished initalizing------------------------")
    start_time = time.time()
    for i_episode in range(N_EPISODES):
        s = env.reset(objects['Jetbot'][0],objects['Grabber'][0])
        ep_r = 0 
        step = 0 
        while True: 
            step += 1 
            a = dqn.choose_action(s)                 
            s_,r,done,info = env.step(a) 

            ep_r += r 
            if done: 
                print('Episode information number: ', info)
                events[info-1] += 1  # update the event records
                if info == 1:
                    print('================================ Path Found =================================')
                    print(' Episode: {0} | Reward: {1} | time: {2} sec'.format(
                        i_episode,round(ep_r,2), int(time.time()- start_time)))
                    # plt.pause(1)
                # env.show_plot([]) 
                break

            s = s_
        
        if i_episode % N_SHOWN == 0:  # show the results in N_shown episode
            episode_events = events - events_history
            events_history = events.copy()
            accuracy = episode_events[0] / sum(episode_events)
            print('\n Results: Reached: {0} Obstacle: {1}, Crashed: {2}, Deviation: {3}, Missing: {4} \n'.format(
                episode_events[0], episode_events[1], episode_events[2], episode_events[3],episode_events[4]))
            print('ACCURACY: {0}'.format(accuracy))
            accuracy_history.append(accuracy)
    accuracy_history = accuracy_history[2:]
    plt.plot(accuracy_history)
    plt.ylim((0,1))
    plt.grid(True)
    plt.title('AVERAGE ACCURACY: {0}'.format(np.average(accuracy_history)))
    # plt.savefig('Tested_result')

    print('\n Results: Reached: {0} Boundary: {1}, Crashed: {2}, Deviation: {3}, Missing: {4} \n'.format(
    episode_events[0], episode_events[1], episode_events[2], episode_events[3],episode_events[4]))
    print('AVERAGE ACCURACY: {0}'.format(np.average(accuracy_history)))
    plt.show()




if __name__ == '__main__':
    # objects = {'Jetbot': [(161, 146), 109, 213, 222, 70], 
    #             'Obstacle': [(508, 223), 465, 551, 293, 153], 
    #             # 'Obstacle': [(0,0), 0,1,0,1],
    #             'Target': [(780, 364), 756, 804, 412, 316], 
    #             'Grabber': [(214, 191), 186, 242, 232, 150]}
    objects = {'Jetbot': [(211, 169), 153, 270, 265, 74], 
                'Obstacle': [(508, 186), 472, 544, 246, 126], 
                'Target': [(758, 335), 733, 783, 383, 288], 
                'Grabber': [(262, 223), 237, 288, 261, 185]}
    mtest(objects, 'DQNnet.pkl')



