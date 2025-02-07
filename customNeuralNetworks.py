from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import gymnasium as gym

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import TD3
from importingNN import importMatlabNN

actorNN, criticNN = importMatlabNN(mode='ConvertModel')

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy,self).__init__(*args, **kwargs)
    def make_actor(self, *args, **kwargs):
        actor,critic = importMatlabNN(mode='ConvertModel')    
        return actor
    def make_critic(self, *args, **kwargs):
        actor,critic = importMatlabNN(mode='ConvertModel')
        return critic




env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")
model = TD3(CustomTD3Policy, env, verbose=1)


model.learn(total_timesteps=10_000)
