import math
import heapq
from Path_Utils import env, plotting

 
class Astar:
    '''Astar set the cost + heuristics as the priority 
    '''

    def __init__(self, s_start, s_goal, obs = [], heuristic_type = 'manhattan'):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.Env = env.Env() if obs == [] else env.Env(obs) # class env
        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        self.OPEN = []  # priority queque / OPENset
        self.ClOSED = []  # visited points
        self.PARENT = dict()  # the recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """ 
        A* searching 
        :return: path, visited order 
         """
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0  # why 0?
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.ClOSED.append(s)

            if s == self.s_goal:
                break  # stop condition

            for s_n in self.get_neighbor(s):
                # why cost is these 2?
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:
                    # condition for update the cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.ClOSED

    def extract_path(self, PARENT):
        """ extract the path based on the parent set
        :return: the planning path """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def get_neighbor(self, s):
        """ find neighbours of state s not in obstacles
        :param s: state 
        :return: neighbours """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """ Calculate cost for this motion 
        :param s_start: starting node 
        :param s_goal: end note
        :return: Cost for this motion
        note: this could be more complex! """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """ check the line segment(s_start, s_end) is collision 
        :param s_start: start node 
        :param s_end: end node 
        :return: True/ False  """

        # XXX confused

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """ 
        f = g + h (cost to come + heuristic value)
        :param s: current state
        :return: f """

        return self.g[s] + self.heuristic(s)

    def heuristic(self, s):
        """ 
        Calculate heuristic
        :param s: current node(state) 
        :return: heuristic function value """

        heuristic_type = self.heuristic_type
        goal = self.s_goal

        if heuristic_type == 'manhattan':
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


if __name__ == '__main__':
    s_start = (5, 5)
    s_goal = (45, 25)
    obstacle = [(20,5),(30,3),(25,17),(25,13),(40,25)]
    astar = Astar(s_start, s_goal, obstacle)
    path, visited = astar.searching()

    plot = plotting.Plotting(s_start, s_goal, obstacle)
    plot.animation(path,visited,'AStar')
    # print(path[::-1])
