import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # assigns mild penalties from default +1 reward (for continuing episode), for deviations from target positions
        # assigns severe penalty for crashes
        # assigns strong reward for hitting exact target
        # assigns moderately large reward for being within 5 coordinate points of target along z axis (considered OK if wind knocks it slightly off course horizontally)
        distance = abs(self.sim.pose[:3] - self.target_pos)
        height = self.target_pos[2]
        largest_region = np.array([height, height, height])
        target = ((distance <= (0.02*largest_region)).sum() == 3)
        region05 = ((distance <= (0.05*largest_region)).sum() == 3)
        region10 = (((region05 == False) & (distance <= (0.1*largest_region))).sum() == 3)
        region15 = (((region10 == False) & (distance <= (0.15*largest_region))).sum() == 3)
        region20 = (((region15 == False) & (distance <= (0.20*largest_region))).sum() == 3)
        region30 = (((region20 == False) & (distance <= (0.20*largest_region))).sum() == 3)
        region40 = (((region30 == False) & (distance <= (0.20*largest_region))).sum() == 3)
        region50 = (((region40 == False) & (distance <= (0.20*largest_region))).sum() == 3)
        # reward = 1 - 0.001*(distance).sum() - 100*(self.sim.pose[2]==0) + 10*(region10) + 3*(region15) + 1*(region20)# + 0.1*(region30) + 0.01*(region40) + 0.001*(region50) # sort-of worked but unlearned
        # reward = 1 - 0.001*(distance).sum() - 100*(self.sim.pose[2]==0) + 20*(region05) + 10*(region10) + 3*(region15) + 1*(region20) + 0.25*(region30) + 0.05*(region40) + 0.01*(region50) # sort-of works but forgets and drone doesn't hover in place
        # reward = 1 - 0.001*(distance).sum() - 100*(self.sim.pose[2]==0) + 20*(region05) + 10*(region10) - 1*(region05 & (sum(self.sim.v)>0.5)) - 0.5*(region10 & (sum(self.sim.v)>0.5)) - 0.05*(region15 & (sum(self.sim.v)>0.5)) - 0.01*(region20 & (sum(self.sim.v)>0.5)) # eventually ends up low; doesn't really hover
        # reward = 1 - 0.001*(distance).sum() - 100*(self.sim.pose[2]==0) + 20*(region05) + 30*(abs(self.sim.pose[2] - height) < 5) - 1*(region05 & (sum(self.sim.v)>0.5)) - 0.5*(region10 & (sum(self.sim.v)>2)) - 0.05*(region15 & (sum(self.sim.v)>5)) - 0.01*(region20 & (sum(self.sim.v)>10)) # never gets off the ground
        # reward = 1 - 0.001*(distance).sum() - 1000*(self.sim.pose[2]==0) + 50*(region05) + 20*(abs(self.sim.pose[2] - height) < (0.5*height)) - 1*(region05 & (sum(self.sim.v)>2)) # doesn't work at all
        # reward = 1 - 0.001*(distance).sum() - 1000*(self.sim.pose[2]==0) + 50*(region05) #- 0.5*(region05 & (abs(self.sim.v).sum()>2)) # may have learned, but forgot quite quickly
        # reward = 1 + 0.03*(self.sim.pose[2]) - 0.0001*(abs(self.sim.pose[0] - self.sim.init_pose[0])) - 0.0001*(abs(self.sim.pose[1] - self.sim.init_pose[1])) - 0.01*(self.sim.v[2]<0)
        # reward = 1 - 0.003*(distance).sum() - 300*(self.sim.pose[2]==0) + 10*(region05) - 0.0003*(abs(self.sim.v[:3])).sum() # sort-of works, but always ends up too high; doesn't really hover - does not unlearn, either, however
        # reward = 1 - 0.003*(distance).sum() - 300*(self.sim.pose[2]==0) + 10*(target) + 1*(region10 | region05) - 0.003*(abs(self.sim.v[:3])).sum() # didn't work
        reward = 1 - 0.005*(distance).sum() - 2000*(self.sim.pose[2]==0) + 100*(target) + 2*(region05) + 1*(region10) -  0.005*(abs(self.sim.v[:3])).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
