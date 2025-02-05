import gymnasium as gym
from stable_baselines3 import TD3
import torch as th

import numpy as np
from typing import Any, Dict
import datetime

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video



class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True




env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128], qf=[128]))

log_dir = "./td3_cartpole_mujoco/"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

model = TD3("MlpPolicy",env,learning_rate=0.005,policy_kwargs=policy_kwargs, verbose=False,
            tensorboard_log=log_dir)
video_recorder = VideoRecorderCallback(gym.make("InvertedPendulum-v4",render_mode='rgb_array'), render_freq=10000)


model.learn(total_timesteps=10_000,callback=video_recorder)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs,deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")