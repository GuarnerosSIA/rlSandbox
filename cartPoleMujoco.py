import gymnasium as gym
from stable_baselines3 import TD3
import torch as th

import numpy as np
from typing import Any, Dict
import datetime

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


env = gym.make('InvertedPendulum-v4',render_mode = "human")

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128], qf=[128]))

log_dir = "./td3_cartpole_mujoco/"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

model = TD3("MlpPolicy",env,learning_rate=0.005,policy_kwargs=policy_kwargs, verbose=False,
            tensorboard_log=log_dir)


model.learn(total_timesteps=1_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs,deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")