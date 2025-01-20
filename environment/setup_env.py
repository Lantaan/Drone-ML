from environment.drone_2d_env import *
from gym.envs.registration import register


def setup_env(render_sim):
    register(
        id='drone-2d-custom-v0',
        entry_point='environment.drone_2d_env:Drone2dEnv',
        kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
                'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
                'initial_throw': False}
    )
    return gym.make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
                    shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True)
