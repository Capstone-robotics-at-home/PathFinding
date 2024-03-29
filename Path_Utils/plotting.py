import matplotlib.pyplot as plt
import os,sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/Path_Utils")
import env
from PIL import ImageDraw
from math import sin, cos


class Plotting:
    def __init__(self, xI, xG, obs):
        """ 
        xI: start point
        xG: end point 
        """
        self.xI, self.xG = xI, xG
        self.env = env.Env(obs)
        self.obs = self.env.obs_map_mod(obs)

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name):
        self.plot_grid(name)
        # self.plot_visited(visited)
        self.plot_path(path)
        plt.show()

    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], 'bs')
        plt.plot(self.xG[0], self.xG[1], 'gs')
        plt.plot(obs_x, obs_y, 'sk')
        plt.title(name)
        # plt.axis('equal')

    def plot_traj(self, ori_path, traj):
        """ plot the original path and the trajectory of the Jetbot
        :ori_path: the Original path solved by Astar 
        :traj: the trajectory of the Jetbot in real-time  """
        self.plot_grid('Trajectory')
        self.plot_path(ori_path)

        # plot the center and the grabber
        Len = 30  # distance from center to grabber
        for i in traj:  # [x,y,heading]
            x, y, theta = i
            center = [x, y]
            c_grabber = [x + Len * cos(theta), y + Len * sin(theta)]
            x_temp = [center[0], c_grabber[0]]
            y_temp = [center[1], c_grabber[1]]
            plt.plot(x_temp, y_temp, 'b--')

        plt.pause(0.1)

        plt.show()

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)
        if self.xG in visited:
            visited.remove(self.xG)

        count = 0
        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            # why do we need length here??
            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) / 3:
                length = 30
            else:
                length = 40
            if count % length == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], 'bs')
        plt.plot(self.xG[0], self.xG[1], 'gs')

        plt.pause(0.1)

    def plot_image_path(self, img, path, N_horizon = 0):
        ''' draw the path on the image 
        return: painted image '''
        start = path[-1]
        end = path[0]
        draw = ImageDraw.Draw(img)
        draw.line(path, fill=(255, 0, 0), width=20)
        draw.line(path[-N_horizon:], fill = (0,0,255), width = 15)
        del draw

        return img

    @staticmethod
    def color_list():
        cl_v = ['silver',
                'wheat',
                'lightskyblue',
                'royalblue',
                'slategray']
        cl_p = ['gray',
                'orange',
                'deepskyblue',
                'red',
                'm']
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        cl = ['silver',
              'steelblue',
              'dimgray',
              'cornflowerblue',
              'dodgerblue',
              'royalblue',
              'plum',
              'mediumslateblue',
              'mediumpurple',
              'blueviolet',
              ]
        return cl
