class Env:
    def __init__(self, obstacles = []):
        self.x_range = 51  # size of the background 
        self.y_range = 31 
        self.motions = [(-1,0),(-1,1),(0,1),(1,1),
                        (1,0),(1,-1),(0,-1),(-1,-1)]
        if obstacles == []:
            self.obs = self.obs_map()
        else: 
            self.obs = self.obs_map_mod(obstacles)

    def update_obs(self, obs):
        self.obs = obs

    def obs_map_mod(self, obs_ls):
        ''' modify the map by the detected obstacles
        :return: map of obstacles  '''
        if obs_ls == []:
            raise ValueError('Obstacle list is empty')
        obs = set() 
        Size = [4,4]  # the size of the 'box' of obstacle, [row, column]
        Row, Column = Size[0], Size[1] 
        for o in obs_ls:
            ox, oy = o[0], o[1] 
            for x in range(ox -Row, ox + Row):
                for y in range(oy - Column, oy + Column):
                    obs.add((x,y))

        return obs 

    def obs_map(self):
        """  
        Initialize the obstacles' positions
        return: map of obstacles 
        """
        x = self.x_range 
        y = self.y_range 
        obs = set()

        # margins 
        for i in range(x):
            obs.add((i,0))
        for i in range(x):
            obs.add((i,y-1))

        for i in range(y):
            obs.add((0,i))
        for i in range(y):
            obs.add((x-1,i))

        # walls 
        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs 




        
