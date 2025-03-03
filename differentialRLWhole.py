from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import datetime

time.sleep(1)

class PioneerEnv(gym.Env):
    def __init__(self, port=23000):
        super(PioneerEnv, self).__init__()
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.require('sim')
        self.sim.setStepping(True)
        self.rightmotor = self.sim.getObject('/PioneerP3DX/rightMotor')
        self.leftMotor = self.sim.getObject('./PioneerP3DX/leftMotor')
        self.robot = self.sim.getObject('./PioneerP3DX')
        self.target = self.sim.getObject('./Target')
        #
        self.control = 0
        self.errorDist = 0
        self.errorAngle = 0
        self.stdControl = 0
        #
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.sim.startSimulation()
        self.l = 0.15
        self.r = 0.1
        self.xd = (np.random.rand() - 0.5)*2
        self.yd = (np.random.rand() - 0.5)*2
        self.reset(0)
        

    def reset(self,seed=0):
        self.sim.stopSimulation()
        time.sleep(1)
        self.sim.startSimulation()
        self.sim.setObjectPosition(self.robot, (0, 0, 0.5), -1)
        self.xd = (np.random.rand() - 0.5)*2
        self.yd = (np.random.rand() - 0.5)*2
        self.sim.setObjectPosition(self.target, (self.xd, self.yd, 0.1), -1)
        self.z = 0
        return self._get_obs(), {"A":0}

    def step(self, action):
        kTheta, kDistance = action
        rightmotor = self.sim.getObject('/PioneerP3DX/rightMotor')
        leftMotor = self.sim.getObject('./PioneerP3DX/leftMotor')
        robot = self.sim.getObject('./PioneerP3DX')
    
        self.sim.setObjectPosition( robot,(0.5,-0.5,0.5),-1)
        (x,y,x) = self.sim.getObjectPosition(robot,-1)

        while (t := self.sim.getSimulationTime()) < 20:
            # print(f'Simulation time: {t:.2f} [s]')
            (x,y,z) = self.sim.getObjectPosition(robot,-1)
            (dummy,beta,theta) = self.sim.getObjectOrientation(robot,-1)
            dx = xd-x
            dy = yd-y
            distance =math.sqrt(dx*dx + dy*dy)
            (x,y,z) = self.sim.getObjectPosition(robot,-1)
            angle = -math.degrees(theta)+math.degrees(math.atan2(dy,dx))
            vel = distance*kDistance
            angVel = angle*kTheta
            print(vel,angVel)
            if distance > 0.1:
                rl = (2*vel+angVel*self.l)/(2*self.r)
                ll = (2*vel-angVel*self.l)/(2*self.r)
            else:
                rl = 0
                ll = 0    
            self.sim.setJointTargetVelocity(rightmotor,rl)
            self.sim.setJointTargetVelocity(leftMotor,ll)
            self.sim.step()
        self.sim.stopSimulation()

        obs = self._get_obs()
        reward = float(self._compute_reward(obs))
        done = self._is_done(obs)
        truncated = self.isTruncated(obs)
        
        
        return obs, reward, done,truncated, {}
    def isTruncated(self,obs):
        if obs[2] > 10:
            return True
        else:
            return False
    def _get_obs(self):
        control = self.control
        errorDist = self.errorDist
        errorAngle = self.errorAngle
        stdControl = self.stdControl
        return np.array([control, errorDist, 
                         errorAngle, stdControl], dtype=np.float32)

    def _compute_reward(self, obs):
        dist = -obs[1]
        angle = -obs[2]
        std = -obs[3]
        return 0.01*(dist + std + angle)

    def _is_done(self, obs):
        distance = obs[2]
        if distance < 0.2:
            terminated = True
        else:
            terminated = False
        return terminated

    def close(self):
        self.sim.stopSimulation()

xd = 1
yd = 1
env = PioneerEnv()
check_env(env)

log_dir = "./TD3Coppelia/"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2))

try:
    model = TD3.load("TD3CartCoppelia.zip",env=env, verbose=1,
            learning_rate=0.0005,batch_size=32,
            tensorboard_log=log_dir, gamma=0.97,
            action_noise=action_noise)
    print("Model loaded")
except:
    print("Unable to load TD3CartCoppelia.zip, creating new model")
    model = TD3("MlpPolicy", env, verbose=1,learning_rate=0.0005,
            batch_size=32,tensorboard_log=log_dir, gamma=0.99,
            action_noise=action_noise)

dtime = time.time()
model.learn(total_timesteps=25_000)

obs,info = env.reset(0)
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs,info = env.reset(0)

env.close()
model.save("TD3CartCoppelia")
print((dtime-time.time())/60)