'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-31 16:11:18
 # @ Description: A simple Reinforcement Learning Environment for DQN
 '''

from Path_Utils.JetbotPy import Decider
from Path_Utils.Astar import Astar
from Path_Utils import plotting



class CartEnv():
    def __init__(self, start, goal, obs):
        self.decider = Decider()
        self.decider.reinit(start, goal)
        self.astar = Astar(start, goal, obs)
        self.plot = plotting.Plotting(start, goal, obs)
        # self.astar_sol = self.astar.searching()[0]

        # self.REWARD_STEP = 0.01
        self.REWARD_REACH = 100
        self.REWARD_BOUND = -10
        self.REWARD_CRASH = -10
        self.REWARD_DEVIATION = -10
        self.REWARD_MISSING = -10
        self.MAX_STEPS = 100
        self.BOUNDS = [1200, 800]
        self.TOTAL_DISTANCE = self.count_distance()  # the Original distance between goal and start 

    def step(self, action):
        """ 
        action 0: forward
        action 1: left
        action 2: right
        return info -> int: 
            0: Nothing happened
            1: Reach the goal 
            2: Go to the boundary
            3: Crash in the obstacle 
            4: Deviate from the ideal path
            5: Too many steps, missing 
        """
        current_distance = self.count_distance() # the current distance between goal and start
        if action == 0:
            self.decider.forward()
        elif action == 1:
            self.decider.left()
        elif action == 2:
            self.decider.right()
        else:
            print("Action Wrong!")

        next_state = self.decider.visited[-1] + self.count_delta()
        done = False
        next_distance = self.count_distance() 
        delta_distance = current_distance - next_distance  # positive reward means get closer to the target 
        # The reward function is an important part.
        # reward = delta_distance / current_distance   # get reward only when it moves forward
        reward = 0 
        info = 0 

        if self.check_reach() == True:
            done = True
            # print('Reached the Goal')
            reward = self.REWARD_REACH
            info = 1
            return next_state, reward, done, info 

        if self.check_boundary() == True:
            done = True
            # print('Boundary hit')
            reward = self.REWARD_BOUND
            info = 2
            return next_state, reward, done, info

        if self.check_crash() == True:
            done = True
            # print('Crashed into Obstacle')
            reward = self.REWARD_CRASH
            info = 3
            return next_state, reward, done, info 

        if self.check_deviation() == True:
            done = True
            # print('Deviate from Ideal path')
            reward = self.REWARD_DEVIATION
            info = 4
            return next_state, reward, done, info 

        if len(self.decider.visited) > self.MAX_STEPS:
            done = True
            # print('Too many steps')
            reward = self.REWARD_MISSING
            info = 5
            return next_state, reward, done, info

        return next_state, reward, done, info 

    def show_plot(self,solution):
        self.plot.plot_traj(solution, self.decider.get_trajectory())

    def reset(self, p_start, p_grabber):
        self.decider = Decider()
        self.decider.reinit(p_start, p_grabber)
        return self.decider.position + [self.decider.heading] + self.count_delta()

    def check_crash(self):  
        """ check if it crashed into obstacle """
        if self.decider.get_position() in self.astar.obs:
            return True
        else:
            return False

    def check_reach(self):  
        """ check if it reached the goal """
        if self.count_distance() <= self.decider.Horizon * self.decider.StepLen:
            return True
        else:
            return False

    def check_boundary(self):  
        """ check if it hits the boundary """
        x, y = self.decider.get_position()
        if x < 0 or x > self.BOUNDS[0] or y < 0 or y > self.BOUNDS[1]:
            return True
        else:
            return False
    
    def check_deviation(self): 
        """ check if the current distance is longer than original """
        distance = self.count_distance() 
        return True if distance > self.TOTAL_DISTANCE + 100 else False

    def count_distance(self):
        """ Get the distance from the jetbot to target """
        x, y = self.decider.get_position()
        goal = self.astar.s_goal
        return abs(x - goal[0]) + abs(y - goal[1])

    def count_delta(self):
        """ Count the relative position to the goal. """
        x, y = self.decider.get_position()
        goal = self.astar.s_goal
        return [goal[0] - x, goal[1] - y]