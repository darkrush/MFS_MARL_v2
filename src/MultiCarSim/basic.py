# properties of agent entities
class AgentProp(object):
    def __init__(self,agent_prop = None):
        self.R_safe     = 0.2  # minimal distance not crash
        self.R_reach    = 0.1  # maximal distance for reach target
        self.L_car      = 0.3  # length of the car
        self.W_car      = 0.2  # width of the car
        self.L_axis     = 0.25 # distance between front and back wheel
        self.R_laser    = 4    # range of laser
        self.N_laser    = 360  # number of laser lines
        self.K_vel      = 1    # coefficient of back whell velocity control
        self.K_phi      = 30   # coefficient of front wheel deflection control

        if agent_prop is not None:
            for k,v in agent_prop.items():
                self.__dict__[k] = v

        self.N_laser = int(self.N_laser)
        if 'true_N' not in agent_prop.keys():
            self.true_N = self.N_laser

class AgentState(object):
    def __init__(self):
        #center point position in x,y axis
        self.x = 0
        self.y = 0
        #linear velocity of back point
        self.vel_b = 0
        # direction of car axis
        self.theta = 0
        # Deflection angle of front wheel
        self.phi = 0
        self.enable = True
        # Movable
        self.movable = True
        self.crash = False
        self.reach = False
        # target x coordinate
        self.target_x   = 1
        # target y coordinate
        self.target_y   = 1