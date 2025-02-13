from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import gymnasium as gym

from stable_baselines3 import TD3,PPO
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.td3.policies import Actor, TD3Policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
import scipy.io
import datetime


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4,128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128,128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128,1)
        self.output = nn.Tanh()

        self.actorWeights()
    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        out1 = self.fc1(features)
        act1 = self.act1(out1)
        out2 = self.fc2(act1)
        act2 = self.act2(out2)
        out3 = self.fc3(act2)
        logits = self.output(out3)
        return logits
    
    def actorWeights(self):
        data = scipy.io.loadmat("actorParams.mat")

        fc1_W = data["fc_1_W"]
        fc1_b = data["fc_1_b"]

        fc2_W = data["fc_2_W"]
        fc2_b = data["fc_2_b"]

        fc3_W = data["fc_3_W"]
        fc3_b = data["fc_3_b"]

        fc1_W_tensor = th.from_numpy(fc1_W).float()
        fc1_b_tensor = th.from_numpy(fc1_b[:,0]).float()
        
        fc2_W_tensor = th.from_numpy(fc2_W).float()
        fc2_b_tensor = th.from_numpy(fc2_b[:,0]).float()

        fc3_W_tensor = th.from_numpy(fc3_W).float()
        fc3_b_tensor = th.from_numpy(fc3_b[:,0]).float()

        self.fc1.weight = th.nn.Parameter(fc1_W_tensor)
        self.fc1.bias = th.nn.Parameter(fc1_b_tensor)
        
        self.fc2.weight = th.nn.Parameter(fc2_W_tensor)
        self.fc2.bias = th.nn.Parameter(fc2_b_tensor)

        self.fc3.weight = th.nn.Parameter(fc3_W_tensor)
        self.fc3.bias = th.nn.Parameter(fc3_b_tensor)
   

class customCritic(nn.Module):
    def __init__(self,path):
        super(customCritic, self).__init__()
        
        self.fcin1 = nn.Linear(4,128)
        self.fcin2 = nn.Linear(1,128)

        self.reluBody = nn.ReLU()

        self.fcBody = nn.Linear(256,128)
        self.fcBodyOutput = nn.ReLU()

        self.output = nn.Linear(128,1)

        self.loadMyWeights(path)
    def forward(self, x):
        x1,x2 = th.split(x,[4,1],dim=1)
        fc1 = self.fcin1(x1)
        fc2 = self.fcin2(x2)
        concat = th.cat((fc1,fc2),1)
        reluBody = self.reluBody(concat)
        fcBody = self.fcBody(reluBody)
        bodyOutput = self.fcBodyOutput(fcBody)
        logits = self.output(bodyOutput)
        return logits*0.01

    def loadMyWeights(self,path):
        data = scipy.io.loadmat(path)

        fc1_W = data["fc1_W"]
        fc1_b = data["fc1_b"]

        fc2_W = data["fc2_W"]
        fc2_b = data["fc2_b"]

        fcbody_W = data["fcBody_W"]
        fcbody_b = data["fcBody_b"]

        fcoutput_W = data["fcoutput_W"]
        fcoutput_b = data["fcoutput_b"]

        fc1_W_tensor = th.from_numpy(fc1_W).float()
        fc1_b_tensor = th.from_numpy(fc1_b[:,0]).float()

        fc2_W_tensor = th.from_numpy(fc2_W).float()
        fc2_b_tensor = th.from_numpy(fc2_b[:,0]).float()

        fcbody_W_tensor = th.from_numpy(fcbody_W).float()
        fcbody_b_tensor = th.from_numpy(fcbody_b[:,0]).float()

        fcoutput_W_tensor = th.from_numpy(fcoutput_W).float()
        fcoutput_b_tensor = th.from_numpy(fcoutput_b[:,0]).float()

        self.fcin1.weight = th.nn.Parameter(fc1_W_tensor)
        self.fcin1.bias = th.nn.Parameter(fc1_b_tensor)

        self.fcin2.weight = th.nn.Parameter(fc2_W_tensor)
        self.fcin2.bias = th.nn.Parameter(fc2_b_tensor)

        self.fcBody.weight = th.nn.Parameter(fcbody_W_tensor)
        self.fcBody.bias = th.nn.Parameter(fcbody_b_tensor)

        self.output.weight = th.nn.Parameter(fcoutput_W_tensor)
        self.output.bias = th.nn.Parameter(fcoutput_b_tensor)


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            # q_net = nn.Sequential(nn.Flatten(),
            #                     nn.Linear(5,128),
            #                     nn.ReLU(),
            #                     nn.Linear(128,128),
            #                     nn.ReLU(),
            #                     nn.Linear(128,1),
            #                     )
            path = "criticParams"+str(idx+1)+".mat"
            q_net = customCritic(path)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs,self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs,self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


env = gym.make('InvertedPendulum-v4',render_mode = "human")
log_dir = "./td3CartBatch512Critic0_01/"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

model = TD3(CustomTD3Policy, env, verbose=1,learning_rate=0.005,
            tensorboard_log=log_dir, batch_size=256)

model.learn(10_000)
