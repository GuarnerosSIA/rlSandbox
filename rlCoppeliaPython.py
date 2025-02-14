from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

time.sleep(1)

class PioneerEnv(gym.Env):
    def __init__(self):
        super(PioneerEnv, self).__init__()
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.sim.setStepping(True)
        self.rightmotor = self.sim.getObject('/PioneerP3DX/rightMotor')
        self.leftMotor = self.sim.getObject('./PioneerP3DX/leftMotor')
        self.robot = self.sim.getObject('./PioneerP3DX')
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.sim.startSimulation()
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
        self.z = 0
        return self._get_obs(), {"A":0}

    def step(self, action):
        rl, ll = action
        self.sim.setJointTargetVelocity(self.rightmotor, float(rl*2))
        self.sim.setJointTargetVelocity(self.leftMotor, float(ll*2))
        self.sim.step()
        obs = self._get_obs()
        reward = float(self._compute_reward(obs))
        done = self._is_done(obs)
        truncated = self.isTruncated(obs)
        return obs, reward, done,truncated, {}
    def isTruncated(self,obs):
        if obs[2] > 10:
            return True
        elif self.sim.getSimulationTime()>10:
            return True
        else:
            return False
    def _get_obs(self):
        x, y, z = self.sim.getObjectPosition(self.robot, -1)
        self.z = abs(z)
        _, beta, theta = self.sim.getObjectOrientation(self.robot, -1)
        dx = xd - x
        dy = yd - y
        distance = math.sqrt(dx * dx + dy * dy + (z*z)*0.1)
        angle = -math.degrees(theta) + math.degrees(math.atan2(dy, dx))
        return np.array([dx, dy, distance, angle], dtype=np.float32)

    def _compute_reward(self, obs):
        d = obs[2]
        angle = obs[3]
        distance = -d
        angle = -abs(angle)
        fall = -self.z
        return 0.01*(distance + fall + angle)

    def _is_done(self, obs):
        distance = obs[2]
        if distance < 0.1:
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

model = TD3("MlpPolicy", env, verbose=1,learning_rate=0.001,
            batch_size=256)
dtime = time.time()
model.learn(total_timesteps=50_000)

obs,info = env.reset(0)
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs,info = env.reset(0)

env.close()

print(dtime-time.time())