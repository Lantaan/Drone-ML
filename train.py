from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from environment.setup_env import setup_env, setup_vec_env
from default_config import default_config


def train(config, agent_name):
    vec_env = setup_vec_env(8, False, False, config)

    logger = configure(f"logs/{agent_name}/", config["train_log_formats"])

    model = PPO("MlpPolicy", vec_env, learning_rate=config["learning_rate"], verbose=config["train_log_policy"])

    model.set_logger(logger)
    model.learn(total_timesteps=config["training_time"], progress_bar=True)
    model.save(f'trained/{agent_name}')


if __name__ == "__main__":
    config_property_to_change = "wind_intensity"
    base_config = default_config
    del base_config[config_property_to_change]

    for value in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        config = base_config.copy()
        config[config_property_to_change] = value

        train(config, f"agent_wind-{int(10*value)}")
