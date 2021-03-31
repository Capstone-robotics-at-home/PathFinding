'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-31 16:11:45
 # @ Description: A Reinforcement Training program using simple Reinforcement learning environment.
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import matplotlib.pyplot as plt 
from simple_RL_env import CartEnv 


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.02                   # learning rate
EPSILON = 0.8               # greedy policy
GAMMA = 0.99                # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 200

N_ACTIONS = 3
N_STATES = 3
ENV_A_SHAPE = 0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 15)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(15, N_ACTIONS) 
        self.out.weight.data.normal_(0,0.1) 

    def forward(self, x):
        x = self.fc1(x) 
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
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random 
            action = np.random.randint(0,N_ACTIONS) 
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) 

        return action 

    def store_transition(self, s,a,r,s_):
        transition = np.hstack((s,[a,r], s_))
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
        q_next = self.target_net(b_s_).detach() 
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)
        loss = self.loss_func(q_eval,q_target) 

        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step()

def main(objects):
    
    obstacle_ls = objects['Obstacle']
    s_start = objects['Jetbot'][0]
    s_goal = objects['Target'][0]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]
    env = CartEnv(s_start, s_goal, obstacle_ls) 
    dqn = DQN()

    plt.ion()
    
    for i_episode in range(300):
        plt.cla()   
        s = env.reset(objects['Jetbot'][0],objects['Grabber'][0])
        ep_r = 0 
        step = 0 
        while True: 
            step += 1 
            a = dqn.choose_action(s) 
            
            s_,r,done = env.step(a) 

            dqn.store_transition(s,a,r,s_)

            ep_r += r 
            if dqn.memory_counter > MEMORY_CAPACITY: 
                dqn.learn() 
                if done: 
                    print('Epsilon: {0} | Reward: {1} | Step: {2}'.format(
                        i_episode,round(ep_r,2), step))

            if step == 1000: 
                break 

            if done: 
                break 

            # if done or step % MEMORY_CAPACITY == 0: 
            #     dqn.learn()
            #     print('Epsilon: {0} | Reward: {1} | Step: {2}'.format(
            #         i_episode,round(ep_r,2), step))
            #     break

            s = s_
        # print(env.cart.x,env.cart.y,env.cart.theta) # print the result 
        
        env.show_plot()
        plt.pause(0.5)

if __name__ == '__main__':
    objects = {'Jetbot': [(210, 462), 107, 314, 577, 347],
               'Obstacle': [(758, 292), 693, 823, 388, 180],
            #    'Obstacle': [(0,0), 0,1,0,1],
               'Target': [(1070, 199), 1036, 1105, 256, 143],
               'Grabber': [(250, 320), 141, 207, 660, 523]}

    main(objects)
