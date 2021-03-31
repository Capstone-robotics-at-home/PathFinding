'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-31 16:11:18
 # @ Description: A simple Reinforcement Learning Environment for DQN
 '''


# import gym
from JetbotPy import Decider
from Astar import Astar
from Path_Utils import plotting


class CartEnv():
    def __init__(self, start, goal, obs):
        self.decider = Decider()
        self.REWARD_STEP = 0.01
        self.REWARD_CRASH = -50
        self.REWARD_REACH = 1000
        self.astar = Astar(start, goal, obs)
        self.plot = plotting.Plotting(start, goal, obs)

    def step(self, action):
        """ 
        action 0: forward
        action 1: left
        action 2: right
        """
        if action == 0:
            self.decider.forward()
        elif action == 1:
            self.decider.left()
        elif action == 2:
            self.decider.right()
        else:
            print("Action Wrong!")

        next_state = self.decider.visited[-1]
        # self.update()
        done = False
        info = []
        reward = self.REWARD_STEP

        if self.check_crash() == True:
            done = True
            info.append('Crashed into Obstacle')
            reward = self.REWARD_CRASH

        if self.check_reach() == True:
            done = True
            info.append('Reached the Goal')
            reward = self.REWARD_REACH

        return next_state, reward, done, info

    def show_plot(self):
        astar_sol, visited = self.astar.searching()
        self.plot.plot_traj(astar_sol, self.decider.get_trajectory())

    def reset(self, p_start, p_grabber):
        self.decider = Decider() 
        self.decider.reinit(p_start, p_grabber)
        return self.decider.position + [self.decider.heading]

    def check_crash(self):  # check if it crashed into obstacle
        if self.decider.get_position() in self.astar.obs:
            return True
        else:
            return False

    def check_reach(self):  # check if it reached the goal
        x, y = self.decider.get_position()
        goal = self.astar.s_goal
        if abs(x - goal[0]) + abs(y - goal[1]) < self.astar.Dstop:
            return True
        else:
            return False
