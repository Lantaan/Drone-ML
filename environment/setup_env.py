from gymnasium import make
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def setup_env(render_sim, initial_throw, config):
    return make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
                shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True,
                initial_throw=initial_throw, wind_intensity=config["wind_intensity"], 
                wind_len_scale=config["wind_len_scale"], render_wind=config["render_wind"])

def setup_vec_env(n_envs, render_sim, initial_throw, config):
    env_kwargs = {'render_sim': render_sim, 'render_path': True, 'render_shade': True, 
                  'shade_distance': 70, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': True, 
                  'initial_throw': initial_throw, 'wind_intensity': config["wind_intensity"], 
                  'wind_len_scale': config["wind_len_scale"], 'render_wind': config["render_wind"]}
    return make_vec_env('drone-2d-custom-v0', n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
