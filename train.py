from stable_baselines3 import PPO
import gym

from drone_2d_env import *
from gym.envs.registration import register

register(
    id='drone-2d-custom-v0',
    entry_point='drone_2d_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
            'initial_throw': False}
)

env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=True, render_shade=True,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=60000)
model.save('new_agent')