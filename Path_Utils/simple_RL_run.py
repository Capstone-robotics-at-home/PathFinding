'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-31 16:11:45
 # @ Description: A Reinforcement Training program using simple Reinforcement learning environment.
 '''

from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import matplotlib.pyplot as plt 
from Path_Utils.simple_RL_env import CartEnv 
# from Testing import realtime_search
import time 


# Hyper Parameters
N_EPISODES = 3000            # Number of total episodes
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9              # greedy policy
GAMMA = 0.99               # reward discount, the larger, the longer sight. 
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 100

N_ACTIONS = 3
N_STATES = 5
ENV_A_SHAPE = 0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 15)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(15, 15)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(15, N_ACTIONS) 
        self.out.weight.data.normal_(0,0.1) 

    def forward(self, x):
        x = self.fc1(x) 
        x = F.relu(x)  # activation 
        x = self.fc2(x) 
        x = F.relu(x)  # activation 
        actions_value = self.out(x) 
        return actions_value


class DQN():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0 
        self.memory_counter = 0 
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES * 2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = LR)
        self.loss_func = nn.MSELoss()

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

    def store_transition(self, s,a,r,s_):
        transition = np.hstack((s,[a,r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY 
        self.memory[index, :] = transition 
        self.memory_counter += 1 

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1 

        # sample batch transitions 
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) 
        b_memory = self.memory[sample_index,:] 
        b_s = torch.FloatTensor(b_memory[:,:N_STATES])
        b_a = torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:,N_STATES+1:N_STATES +2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience 
        q_eval = self.eval_net(b_s).gather(1,b_a)  # shape:(batch, 1)
        q_next = self.target_net(b_s_).detach()    # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)   # shape (batch, 1)
        loss = self.loss_func(q_eval,q_target) 

        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()

def main(objects):
    plt.ion()
    obstacle_ls = objects['Obstacle']
    s_start = objects['Jetbot'][0]
    s_goal = objects['Target'][0]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]
    env = CartEnv(s_start, s_goal, obstacle_ls) 
    dqn = DQN()
    events =np.array([0,0,0,0,0])  # record the events during training: [bound, crash, deviate, reach]
    events_history = events.copy()
    episode_events = np.array([0,0,0,0,0]) # record the events during each episode
    accuracy_history = [0.0] # record the accuracy 
    N_shown = 100  # show results for N steps

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
            dqn.store_transition(s,a,r,s_)
            if dqn.memory_counter > MEMORY_CAPACITY: 
                dqn.learn() 


            ep_r += r 
            if done: 
                events[info-1] += 1  # update the event records
                if info == 1:
                    print('\n================================ Good =================================')
                    print(' Episode: {0} | Reward: {1} | Step: {2} | Memory: {3} | time: {4} sec'.format(
                        i_episode,round(ep_r,2), step, dqn.memory_counter, int(time.time()- start_time)))
                    # plt.pause(1)
                # env.show_plot(astar_sol) 
                break

            s = s_
        
        if i_episode % N_shown == 0:  # show the results in N_shown episode
            episode_events = events - events_history
            events_history = events.copy()
            accuracy = episode_events[0] / sum(episode_events)
            print('\n Results: Reached: {0} Obstacle: {1}, Crashed: {2}, Deviation: {3}, Missing: {4} \n'.format(
                episode_events[0], episode_events[1], episode_events[2], episode_events[3],episode_events[4]))
            print('ACCURACY: {0}'.format(accuracy))
            accuracy_history.append(accuracy)

            # # show the accuracy results
            # plt.plot(accuracy_history[2:])
            # plt.ylim((0,1))
            # plt.grid(True)
            # plt.title('Results: Reached: {0} Boundary: {1}, Crashed: {2}, Deviation: {3}, Missing: {4} \n'.format(
            #     episode_events[0], episode_events[1], episode_events[2], episode_events[3],episode_events[4]) + 'Accuracy: {0}'.format(accuracy))
            plt.pause(0.5)
    plt.ioff()
    plt.plot(accuracy_history[2:])
    plt.ylim((0,1))
    plt.grid(True)
    plt.title('RL_results.jpg')
    # plt.savefig('RL_results_{0}.jpg'.format((round(accuracy_history[-1],2))))

    print('\n Results: Reached: {0} Boundary: {1}, Crashed: {2}, Deviation: {3}, Missing: {4} \n'.format(
    episode_events[0], episode_events[1], episode_events[2], episode_events[3],episode_events[4]))
    print('ACCURACY HISTORY: {0}'.format(accuracy_history))
    torch.save(dqn.eval_net, 'DQNnet_{0}.pkl'.format(round(accuracy_history[-1],2)))
    plt.show()




if __name__ == '__main__':
    objects = {'Jetbot': [(161, 146), 109, 213, 222, 70], 
                'Obstacle': [(508, 223), 465, 551, 293, 153], 
                # 'Obstacle': [(0,0), 0,1,0,1],
                'Target': [(780, 364), 756, 804, 412, 316], 
                'Grabber': [(214, 191), 186, 242, 232, 150]}
    main(objects)



