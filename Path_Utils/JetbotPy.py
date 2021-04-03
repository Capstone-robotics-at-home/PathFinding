""" 
Author: ZionDeng 
The decider class is used for decision making for Jetbot based on current position and env
Decisions: left, right, forward 
"""

from math import sin, cos, atan, pi


class Decider():
    '''
    Decider class for jetbot motion decision 
    '''

    def __init__(self, if_write=False):
        self.position = [0,0]
        self.heading = 0
        self.visited = []
        self.Horizon = 10  # A tuning parameter
        self.Angle = pi / 12  # Turning angle at each step
        self.StepLen = 20  # Step Length, should be shorter than Horizon
        self.if_write = if_write  # if you want to write the command to target dir
        self.Target_path = '../Capstone_Simulation/gym_rev/cmd.txt'  # path of cmd.txt
        self.cmd = '0'
        self.cmd_record = [] # record the command history

    def reinit(self, position, grabber_p):
        self.position = list(position)
        self.heading = self.checkHeading(grabber_p, position)

    def right(self):
        self.heading -= self.Angle if self.heading > -pi else (self.Angle - pi)
        self.visited.append(self.position + [self.heading])
        self.cmd = 'left'  # should be left not right 
        self.cmd_record.append(2)
        if self.if_write:
            self.send_cmd()

    def left(self):
        self.heading += self.Angle if self.heading < pi else (self.Angle - pi) 
        self.visited.append(self.position + [self.heading])
        self.cmd = 'right'  # should be right not left 
        self.cmd_record.append(1)
        if self.if_write:
            self.send_cmd() 

    def forward(self):
        x, y = self.position
        x += int(self.StepLen * cos(self.heading))
        y += int(self.StepLen * sin(self.heading))
        self.position = [x, y]
        self.visited.append([x, y, self.heading])
        self.cmd = 'forward'
        self.cmd_record.append(0)
        if self.if_write: 
            self.send_cmd()

    def send_cmd(self):
        ''' A tool to transmit cmd line to jetbot
        warning: the counting is the mirror of the real world pic -> left right inverse '''
        cmd_file = open(self.Target_path, 'w')
        cmd_file.write(self.cmd)
        cmd_file.close()

    def jetbot_step(self, path, obs_set):
        """ A step function for jetbot simulation 
        First turn to the desired direction 
        Then move forward in that direction 
        :cmd: the list of path solved by Astar 
        :obs_ls: the list of obstacles """
        target_point = path[-self.Horizon]
        target_heading = self.checkHeading(target_point, self.position)
        while abs(self.heading - target_heading) > 0.27:
            print('Position: {} || Target point: {} ||'.format(self.position,target_point),
                'Heading: %.2f || Target Heading: %.2f' %(self.heading*180/pi, target_heading*180/pi))
            # turn until it reaches a desired angle,
            # the angle should be slightly larger than self.Angle
            if target_heading > self.heading:
                self.left()
                return
            else:
                self.right()
                return

        self.forward()
        print('Position: {} || Target point: {} ||'.format(self.position,target_point),
            'Heading: %.2f || Target Heading: %.2f' %(self.heading*180/pi, target_heading*180/pi))


    def get_trajectory(self):
        return self.visited

    def get_position(self):
        return tuple(self.position)

    def checkHeading(self, pHead, pBody):
        ''' check where the jetbot is heading for
        :return: the theta of heading angle '''
        Pi = 3.1415926
        # pHead = objs['Target'][0]
        # pBody = objs['Jetbot'][0]
        x1, y1 = pHead[0], pHead[1]
        x2, y2 = pBody[0], pBody[1]
        dx, dy = x1-x2, y1-y2
        if dx == 0:
            return Pi if dy > 0 else -Pi
        if dx > 0:  # first quadrant and forth quadrant
            return atan((y1-y2)/(x1-x2))
        if dx < 0 and dy >= 0:  # second quadrant
            return atan((y1-y2)/(x1-x2)) + Pi
        if dx < 0 and dy < 0:  # third quadrant
            return atan((y1-y2)/(x1-x2)) - Pi

    def can_step(self, obs):
        """ check if jetbot can go in that direction
        Since one step further can collide
        :obs_set: the Obstacle set (same idea as that in Astar env) """
        x_test, y_test = self.position
        x_test += int(self.StepLen * cos(self.heading))
        y_test += int(self.StepLen * sin(self.heading))
        return False if (x_test, y_test) in obs else True

    def obj_isvalid(self, objs):
        """ check if the objects are valid """
        # objects = {'Jetbot': [(757, 437), 663, 851, 549, 325],
        #             'Obstacle': [(1315, 496), 1251, 1379, 605, 387],
        #             'Target': [[(1701, 440), 1666, 1736, 511, 369], [(712, 1036), 690, 735, 1072, 1001]],
        #             'Grabber': [(654, 398), 625, 683, 444, 352]}
        if not len(objs) == 4:
            return False
        if not len(objs['Jetbot'][0]) == 2:
            return False
        if not len(objs['Target'][0]) == 2:
            return False
        if not len(objs['Grabber'][0]) == 2:
            return False
        return True


if __name__ == '__main__':
    decider = Decider()
    objects = {'Jetbot': [(475, 454), 370, 581, 602, 307], 
    'Obstacle': [(971, 447), 904, 1039, 565, 329], 
    'Target': [[(1335, 454), 1289, 1381, 537, 371], [(711, 1038), 688, 735, 1078, 999]], 
    'Grabber': [(585, 591), 540, 631, 667, 515]}
    print(decider.obj_isvalid(objects))
