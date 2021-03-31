'''
 # @ Author: Zion Deng
 # @ Create Time: 2021-03-26 14:55:40
 # @ Description: copied form Weibo 
 '''

import gym
from numpy import core
import polytope as pt
import numpy as np
from math import sin, cos

from scipy.integrate._ivp.ivp import solve_ivp


def cart_ode(t, x, v, w):
    """ state dynamic ordinary differential equation
    x' = v * cos(theta) 
    y' = v * sin(theta)
    theta' = w """
    return [v * cos(x[2]), v * sin(x[2]), w]


class Cart(object):
    def __init__(self, x=0, y=0, theta=0,
                 width=0.1, lf=0.04, lr=0.16):
        self.x = x
        self.y = y
        self.theta = theta
        self.lf = lf
        self.lr = lr
        self.width = width

    def step(self, v, w, time):
        y0 = [self.x, self.y, self.theta]
        sol = solve_ivp(cart_ode, [0, time], y0, args=(v, w), rtol=1e-8)
        self.x = sol.y[0, :][-1]
        self.y = sol.y[1, :][-1]
        self.theta = sol.y[2, :][-1]

    def corners_posistion(self):
        corner1 = np.array([self.lf, self.width / 2])
        corner2 = np.array([self.lf, -self.width / 2])
        corner3 = np.array([-self.lr, -self.width / 2])
        corner4 = np.array([-self.lr, self.width / 2])

        corners = np.vstack([corner1, corner2, corner3, corner4]).T
        translation = np.array([[self.x], [self.y]])
        rotation_matrix = np.array([[cos(self.theta), -sin(self.theta)],
                                    [sin(self.theta),
                                     cos(self.theta)]])
        corners = rotation_matrix @ corners + translation

        return corners


class CartEnv(gym.Env):
    def __init__(self, step_time=0.01):
        self.cart = Cart()
        self.obstacles = list()
        self.cart_poly = pt.qhull(self.cart.corners_posistion().T)
        self.goal = pt.qhull(
            np.array([[1.3, 1.3], [1.3, 2], [2, 1.3], [2, 2]]))
        self.frame = pt.qhull(
            np.array([[-0.5, -0.5], [2, -0.5], [-0.5, 2], [2, 2]]))
        self.step_time = step_time

    def update_cart_polytope(self):
        self.cart_poly = pt.qhull(self.cart.corners_posistion().T)

    def add_obstacle(self, corners):
        p1 = pt.qhull(np.asarray(corners))
        self.obstacles.append(p1)

    def check_goal(self):
        A = self.goal.A
        b = self.goal.b
        for point in pt.extreme(self.cart_poly):
            if not np.all(A@point - b <= 0):
                return False
        return True

    def check_frame(self):
        A = self.frame.A
        b = self.frame.b
        for point in pt.extreme(self.cart_poly):
            if (not (np.all(A @ point - b <= 0))):
                return False
        return True

    def check_crash(self):
        for o in self.obstacles:
            A = o.A
            b = o.b
            for point in pt.extreme(self.cart_poly):
                if (np.all(A@point - b <= 0)):
                    return False

            A = self.cart_poly.A
            b = self.cart_poly.b
            for point in pt.extreme(o):
                if np.all(A @ point - b <= 0):
                    return False

        return True

    def step(self, action):
        """ 
        action 0: forward
        action 1: left
        action 2: right
        """
        if action == 0:
            v = 0.5
            w = 0

        if action == 1:
            v = 0
            w = 0.5
        if action == 2:
            v = 0
            w = -0.5

        self.cart.step(v, w, self.step_time)
        next_state = [self.cart.x, self.cart.y, self.cart.theta]
        self.update_cart_polytope()
        done = False
        info = []
        reward = -0.01
          # tuning argument here

        if self.check_crash() == False:
            done = True
            info.append('crashed into obstacle')
            reward = -50

        if self.check_frame() == False:
            done = True
            info.append('out of frame')
            reward = - 50

        if self.check_goal() == True:
            done = True
            info.append('reach the goal')
            reward = 1000
        return next_state, reward, done, info

    def reset(self):
        self.cart.x = 0
        self.cart.y = 0
        self.cart.theta = 0
        return [0, 0, 0]
