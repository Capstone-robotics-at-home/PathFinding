'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-04-01 15:53:22
 # @ Description: Test file for Astar search
 '''

from matplotlib import pyplot as plt
from RLmodel_test import DQNnet
from simple_RL_env import CartEnv
from JetbotPy import Decider
from Astar import Astar
import plotting
from simple_RL_train import Net


Ratio = 1 
def Astar_search(objects, decider):
    """ Searching conducting
    :return: path points """
    obstacle_ls = objects['Obstacle']
    s_start = decider.get_position()
    s_goal = objects['Target'][0]
    jetbot_size = objects['Jetbot'][-4:]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]

    global Ratio 
    Path_Found = False
    while not Path_Found:  # Error might take place when scale changes 
        try:
            astar = Astar(s_start, s_goal, obstacle_ls, jetbot_size, Ratio)
            path_sol, visited = astar.searching()
            Path_Found = True
        except UnboundLocalError:
            Ratio -= 0.1
            print('Error, try change the size, ratio = ', Ratio)
    return path_sol


def get_obs_set(obstacle_list, margin_size):
    # the input style: [[(623, 165), 546, 700, 288, 42]], [834, 1073, 636, 287]
    margin_x = (margin_size[1] - margin_size[0]) // 2
    margin_y = (margin_size[2] - margin_size[3]) // 2
    if obstacle_list == []:
        raise ValueError('Obstacle list is empty')
    obs = set()
    for o in obstacle_list:
        # the 4 parameters of the obstacle 'box'
        left, right, top, bottom = o[-4:]
        for x in range(left - margin_x, right+1 + margin_x):
            for y in range(bottom - margin_y, top+1 + margin_y):
                obs.add((x, y))

    return obs


def realtime_search(objects):
    """ Path finding with realtime Astar algorithm
    return command at each step and the original Astar path solution"""

    decider = Decider(False)
    # plot the original path
    jetbot_pos, jetbot_size = objects['Jetbot'][0], objects['Jetbot'][-4:]
    grab_pos = objects['Grabber'][0]
    decider.reinit(jetbot_pos, grab_pos)

    obstacle_ls = objects['Obstacle']
    s_start = objects['Jetbot'][0]
    s_goal = objects['Target'][0]
    if type(obstacle_ls[0]) == type(()):  # if there is only one obstacle:
        obstacle_ls = [obstacle_ls]

    global Ratio 
    Path_Found = False
    while not Path_Found:  # Error might take place when scale changes 
        try:
            astar = Astar(s_start, s_goal, obstacle_ls, jetbot_size, Ratio)
            Original_path, visited = astar.searching()
            Path_Found = True
        except UnboundLocalError:
            Ratio -= 0.1
            print('Error, try change the size, ratio = ', Ratio)

    plot = plotting.Plotting(s_start, s_goal, obstacle_ls)
    plot.animation(Original_path, visited, 'AStar')

    # define path and obstacle set
    path = Original_path
    obs_set = get_obs_set(obstacle_ls, jetbot_size)

    # start iteration
    while len(path) > decider.Horizon:
        decider.jetbot_step(path, obs_set)
        path = Astar_search(objects, decider)

    # get the result
    trajectory = decider.get_trajectory()
    print('Terminate, Total number of movements is: %d' % len(trajectory))
    plot.plot_traj(Original_path, trajectory)
    return decider.cmd_record, Original_path


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
    objects = {'Jetbot': [(161, 146), 109, 213, 222, 70], 
                'Obstacle': [(508, 223), 465, 551, 293, 153], 
                # 'Obstacle': [(0,0), 0,1,0,1],
                'Target': [(780, 364), 756, 804, 412, 316], 
                'Grabber': [(214, 191), 186, 242, 232, 150]}
    # objects = {
    #     'Jetbot': [(829, 278), 695, 964, 485, 71], 'Obstacle': [(972, 588), 898, 1047, 718, 458], 'Target': [(1559, 727), 1517, 1602, 819, 636], 'Grabber': [(962, 377), 920, 1004, 447, 308]}
    # realtime_search(objects)
    sol = RL_search(objects)
    print(sol)
