from gym.envs.registration import register


register(
    id='drone-2d-custom-v0',
    entry_point='environment.drone_2d_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
            'initial_throw': True, 'wind_intensity': 0.0, 'wind_len_scale': 100, 'render_wind': True}
)