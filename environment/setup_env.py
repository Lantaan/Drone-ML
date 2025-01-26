import gym


def setup_env(render_sim, initial_throw, config):
    return gym.make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
                    shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True,
                    initial_throw=initial_throw, wind_intensity=config["wind_intensity"], 
                    wind_len_scale=config["wind_len_scale"], render_wind=config["render_wind"])
