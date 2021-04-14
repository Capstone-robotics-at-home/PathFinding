
'''
 # @ Author: Zion Deng
 # @ Description: Run RL in real-time: train, load and predict 
 '''
from Path_Utils.simple_RL_env import CartEnv
from Path_Utils.simple_RL_train import Net,train
from Path_Utils.RLmodel_test import mtest,DQNnet 


def RL_search(objects):
    """ Path finding methods with Reinforcement Learning 
    return: the recorded commands to the goal"""   

    # generate env
    obstacle_ls = objects['Obstacle']
    s_start = objects['Jetbot'][0]
    s_goal = objects['Target'][0]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]
    env = CartEnv(s_start, s_goal, obstacle_ls) 

    # initialize RL brain
    dqn = DQNnet('DQNnet.pkl') 
    info = 0  # initialize info with 0 means nothing happened

    # try until find solution
    while info is not 1: 
        s = env.reset(objects['Jetbot'][0],objects['Grabber'][0])
        ep_r = 0 
        step = 0 
        # while info is not 1: 
        while True:
            step += 1 
            a = dqn.choose_action(s)                 
            s_,r,done,info = env.step(a) 

            ep_r += r 
            if done: 
                print('Episode information number: ', info)
                if info == 1:
                    print('================================ Path Found =================================')
                    return env.decider.cmd_record
                break
            s = s_
    if info is not 1:
        print('================================ERROR: path not found================================')
        return [0]
    
    return []


if __name__ == '__main__':
    if_trained = False
    objects = {'Jetbot': [(126, 403), 100, 153, 435, 372], 
                'Obstacle': [(279, 386), 224, 334, 437, 335], 
                'Target': [(385, 188), 352, 418, 224, 152], 
                'Grabber': [(158, 407), 140, 177, 436, 378]}
    train(objects)
    if_trained = True
    if if_trained:
        mtest(objects,'DQNnet.pkl')
    sol = RL_search(objects)
    print('solution is: ',sol)
