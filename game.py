from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import random
import numpy as np
from enum import Enum
client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)
class Direction(Enum):
    FORWARD = 1
    BACKWARD = 0

motor_left = sim.getObjectHandle('/Main_plate/motor_left')
motor_right = sim.getObjectHandle('/Main_plate/motor_right')
chassis = sim.getObjectHandle('/Main_plate')

class Game:

    def __init__(self, sim = sim):
        self.sim = sim
        self.sim.startSimulation()
        print("Bot initialized")
        sim.startSimulation()
        self.reset()


    def reset(self):
        self.sim.setJointTargetVelocity(motor_left, 0.0)
        self.sim.setJointTargetVelocity(motor_right, 0.0)
        self.sim.setObjectPosition(chassis, [-0.04,-0.06,0.090])
        self.sim.setObjectOrientation(chassis, [0.0, random.uniform(-0.05, 0.05), 0.0])
        self.speed = 0.0
        self.score = 0
        self.frame_iteration = 0
        self.previous_speed = 0.0

    def move(self, action):
        # if np.array_equal(action, [0,1]):
        #     self.speed -= 1
        #     self.sim.setJointTargetVelocity(motor_left, self.speed)
        #     self.sim.setJointTargetVelocity(motor_right, self.speed)
        #     print("Backward")
        # elif np.array_equal(action, [1,0]):
        #     self.speed += 1
        #     self.sim.setJointTargetVelocity(motor_left, self.speed)
        #     self.sim.setJointTargetVelocity(motor_right, self.speed)
        #     print("Forward")

        ## 9 actions (-8 to 0 to 8)

        if np.array_equal(action, [0,0,0,0,0,0,0,0,0]):
            self.speed = 0
        elif np.array_equal(action, [0,0,0,0,0,1,0,0,0]):
            self.speed = 1
        elif np.array_equal(action, [0,0,0,0,0,0,1,0,0]):
            self.speed = 2
        elif np.array_equal(action, [0,0,0,0,0,0,0,1,0]):
            self.speed = 4
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,1]):
            self.speed = 5
        elif np.array_equal(action, [0,0,0,0,1,0,0,0,0]):
            self.speed = -1
        elif np.array_equal(action, [0,0,0,1,0,0,0,0,0]):
            self.speed = -2
        elif np.array_equal(action, [0,0,1,0,0,0,0,0,0]):
            self.speed = -4
        elif np.array_equal(action, [0,1,0,0,0,0,0,0,0]):
            self.speed = -5

        self.sim.setJointTargetVelocity(motor_left, self.speed)
        self.sim.setJointTargetVelocity(motor_right, self.speed)


        
    def is_imbalanced(self):
        self.orientation = self.sim.getObjectOrientation(chassis)[1]
        self.position = self.sim.getObjectPosition(chassis)[0]
        if abs(self.orientation) > 0.418 or abs(self.position) > 1:
            return True
        if abs(self.previous_speed - self.speed) > 3:  ## If the speed changes too fast
            return True

        self.previous_speed = self.speed
        return False
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        self.move(action)
        self.sim.step()
        # Check if game over
        reward = 1
        self.score += 1
        terminated = False
        if self.is_imbalanced():
            terminated = True

        return reward, terminated, self.score  ## Return reward and terminated status
        
    def get_state(self):
        self.linear_velocity, self.angular_velocity = self.sim.getObjectVelocity(chassis)
        self.orientation = self.sim.getObjectOrientation(chassis)[1]    
        self.position = self.sim.getObjectPosition(chassis)[0]    

        state = [
            # Orientation and position
            self.orientation,         # theta
            self.angular_velocity[1], # theta_dot
            self.position,            # x
            self.linear_velocity[0],  # x_dot
            ]
        return np.array(state, dtype=np.float32)