'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-04-01 15:53:22
 # @ Description: Test file for Astar search
 '''

from Path_Utils.JetbotPy import Decider
from Path_Utils.Astar import Astar
from Path_Utils import plotting


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
    return decider.cmd_record


if __name__ == '__main__':

    objects = {'Jetbot': [(210, 462), 107, 314, 577, 347],
               'Obstacle': [(758, 292), 693, 823, 388, 180],
               'Target': [(1070, 199), 1036, 1105, 256, 143],
               'Grabber': [(174, 591), 141, 207, 660, 523]}
    # objects = {
    #     'Jetbot': [(829, 278), 695, 964, 485, 71], 'Obstacle': [(972, 588), 898, 1047, 718, 458], 'Target': [(1559, 727), 1517, 1602, 819, 636], 'Grabber': [(962, 377), 920, 1004, 447, 308]}
    realtime_search(objects)
