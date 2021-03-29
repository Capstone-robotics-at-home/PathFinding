import numpy as np 

class Env:
    def __init__(self, obstacles, margin_size = [0,0,0,0], modification = 1):
        """ 
        The environment class for Astar
        The modificaiton added because of the noise during detecting 
        might cause the searching error 
        """

        # self.motions = [(2,0),(-2,0),(0,2),(0,-2),
        #                 (1,1),(1,-1),(-1,1),(-1,-1)]
        self.motions = 6 * np.array(
            [(5,0),(-5,0),(0,5),(0,-5),
            (3,4),(3,-4),(-3,4),(-3,4),                   
            (4,3),(4,-3),(-4,3),(-3,4),]
        )
        self.ratio = modification
        self.obs = self.obs_map_mod(obstacles, margin_size)

    def update_obs(self, obs):
        self.obs = obs

    def obs_map_mod(self, obs_ls, margin_size = [0,0,0,0]):
        ''' modify the map by the detected obstacles
        :return: map of obstacles  '''

        # the input style: [[(623, 165), 546, 700, 288, 42]], [834, 1073, 636, 287]
        margin_x = int((margin_size[1] - margin_size[0]) // 2 * self.ratio)
        margin_y = int((margin_size[2] - margin_size[3]) // 2 * self.ratio)
        if obs_ls == []:
            raise ValueError('Obstacle list is empty')
        obs = set() 
        for o in obs_ls:
            left, right, top, bottom = o[-4:]  # the 4 parameters of the obstacle 'box'
            for x in range(left - margin_x, right+1 + margin_x):
                for y in range(bottom - margin_y, top+1 + margin_y):
                    obs.add((x,y))

        return obs 

        
